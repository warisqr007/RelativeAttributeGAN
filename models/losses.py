"""
All loss functions for training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNetLoss(nn.Module):
    """Pairwise ranking loss for attribute rankers"""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, score_i, score_j, label):
        """
        Args:
            score_i, score_j: (batch,) - scores for two items
            label: (batch,) - 1 if i > j, -1 if j > i
            
        Returns:
            loss: scalar
        """
        score_diff = self.temperature * (score_i - score_j)
        # For label=1 (i>j), we want sigmoid(score_diff) close to 1
        # For label=-1 (j>i), we want sigmoid(score_diff) close to 0
        target = (label + 1) / 2  # Convert {-1, 1} to {0, 1}
        loss = F.binary_cross_entropy_with_logits(score_diff, target)
        return loss



class CombinedGANLoss(nn.Module):
    """
    Combined loss for generator with multiple components:
    1. Adversarial loss
    2. Attribute matching loss (uses BOTH discriminator AND rankers)
    3. Distance control loss
    4. Smoothness loss
    """
    def __init__(
        self,
        lambda_adv=1.0,
        lambda_attr=1.0,
        lambda_dist=1.0,
        lambda_smooth=0.1,
        lambda_ranker=0.5,
        lambda_gmm_prior=0.1,
        lambda_cycle=0.5,      # NEW
        lambda_ortho=0.1,      # NEW
        use_wasserstein=False
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_attr = lambda_attr
        self.lambda_dist = lambda_dist
        self.lambda_smooth = lambda_smooth
        self.lambda_ranker = lambda_ranker  # NEW
        self.lambda_gmm_prior = lambda_gmm_prior  # NEW
        self.lambda_ortho = lambda_ortho  # NEW
        self.lambda_cycle = lambda_cycle  # NEW
        self.use_wasserstein = use_wasserstein
    
    def adversarial_loss(self, fake_logits):
        """Standard GAN loss for generator"""
        if self.use_wasserstein:
            return -fake_logits.mean()
        else:
            return F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.ones_like(fake_logits)
            )
    
    def attribute_matching_loss(self, predicted_deltas, target_deltas):
        """
        Ensure the generator applied the requested attribute changes.
        Uses discriminator's prediction.
        
        Args:
            predicted_deltas: (batch, num_attributes) - what discriminator predicts changed
            target_deltas: (batch, num_attributes) - what we asked generator to change
        """
        return F.mse_loss(predicted_deltas, target_deltas)
    
    def ranker_based_attribute_loss(
        self,
        rankers,
        original_embeddings,
        transformed_embeddings,
        target_deltas,
        attributes
    ):
        """
        NEW: Use pretrained attribute rankers to verify attribute changes.
        This provides an independent signal beyond the discriminator.
        
        Args:
            rankers: MultiAttributeRanker with frozen weights
            original_embeddings: (batch, embedding_dim)
            transformed_embeddings: (batch, embedding_dim)
            target_deltas: (batch, num_attributes) - requested changes
            attributes: List of attribute names
            
        Returns:
            loss: Scalar measuring if ranker scores changed in expected direction
        """
        total_loss = 0.0
        
        with torch.no_grad():  # Rankers are frozen, no gradients needed
            # Get ranker scores for original and transformed
            for i, attr in enumerate(attributes):
                # Score original embeddings
                score_orig = rankers.score(original_embeddings, attr)  # (batch,)
                
                # Score transformed embeddings
                score_transformed = rankers.score(transformed_embeddings, attr)  # (batch,)
                
                # The change in score should correlate with target_delta
                actual_change = score_transformed - score_orig
                target_change = target_deltas[:, i]
                
                # Loss: penalize if actual change doesn't match target direction
                # We want: sign(actual_change) == sign(target_change)
                # And magnitude should be proportional
                
                # Method 1: MSE on normalized changes
                # Normalize to [-1, 1] range for stable training
                actual_change_norm = torch.tanh(actual_change)
                target_change_norm = torch.tanh(target_change)
                
                attr_loss = F.mse_loss(actual_change_norm, target_change_norm)
                total_loss += attr_loss
        
        return total_loss / len(attributes)
    
    def distance_control_loss(self, transformed, original, lambda_anon):
        """
        Control how far transformed embeddings are from originals based on anonymization level.
        
        Args:
            transformed: (batch, embedding_dim)
            original: (batch, embedding_dim)
            lambda_anon: (batch, 1) - target anonymization level [0, 1]
        """
        # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
        cos_sim = F.cosine_similarity(transformed, original, dim=1)
        
        # Target similarity decreases linearly with anonymization
        target_sim = 1.0 - lambda_anon.squeeze()
        
        # Penalize deviation from target similarity
        loss = F.mse_loss(cos_sim, target_sim)
        
        return loss
    
    def smoothness_loss(self, embeddings_sequence):
        """
        Ensure smooth interpolation between attribute values.
        
        Args:
            embeddings_sequence: List of (batch, embedding_dim) for interpolated steps
        """
        if len(embeddings_sequence) < 2:
            return torch.tensor(0.0, device=embeddings_sequence.device)
        
        # Compute consecutive similarities
        similarities = []
        for i in range(len(embeddings_sequence) - 1):
            sim = F.cosine_similarity(
                embeddings_sequence[i],
                embeddings_sequence[i + 1],
                dim=1
            )
            similarities.append(sim)
        
        similarities = torch.stack(similarities)  # (num_steps, batch)
        
        # Penalize variance - we want consistent similarity between steps
        variance = similarities.var(dim=0).mean()
        
        return variance
    
    def gmm_prior_loss(self, transformed_embeddings, gmm_prior):
        nll = gmm_prior.nll(transformed_embeddings)
        nll_normalized = nll / transformed_embeddings.size(1)
        loss = nll_normalized.mean()
        return loss
    
    def orthogonality_loss(self, generator, embeddings, deltas, lambda_anon):
        """
        Encourage orthogonality: ⟨∂e'/∂Δ_i, ∂e'/∂Δ_j⟩ ≈ 0 for i≠j
        """
        embeddings = embeddings.detach()
        deltas = deltas.clone().requires_grad_(True)
        
        # Forward pass
        e_transformed = generator(embeddings, deltas, lambda_anon)
        
        # Compute gradients w.r.t. deltas
        grads = torch.autograd.grad(
            e_transformed.sum(),  # Scalar output for backward
            deltas,
            create_graph=True,
            retain_graph=True
        )[0]  # Shape: (batch, num_attributes)
        
        # Compute pairwise dot products
        # We want different attributes to be orthogonal
        ortho_loss = 0.0
        num_attrs = grads.shape[1]
        
        for i in range(num_attrs):
            for j in range(i+1, num_attrs):
                # Dot product between gradient of attr_i and attr_j
                dot = (grads[:, i] * grads[:, j]).pow(2).mean()
                ortho_loss += dot
        
        # Normalize by number of pairs
        num_pairs = (num_attrs * (num_attrs - 1)) / 2
        return ortho_loss / num_pairs if num_pairs > 0 else ortho_loss


    def cycle_consistency_loss(self, generator, embeddings, deltas, lambda_anon):
        """
        Cycle: e -> G(e, Δ, λ) -> G(e', -Δ, λ*0.5) ≈ e
        """
        # Forward
        e_transformed = generator(embeddings, deltas, lambda_anon)
        
        # Backward (reverse deltas, smaller lambda)
        e_reconstructed = generator(
            e_transformed.detach(),  # Stop gradients through first transform
            -deltas,
            lambda_anon * 0.5  # Smaller anonymization for reverse
        )
        
        # Reconstruction loss
        loss = F.mse_loss(e_reconstructed, embeddings)
        return loss

    
    def forward(
        self,
        fake_logits,
        predicted_attr_deltas,
        target_attr_deltas,
        transformed_embeddings,
        original_embeddings,
        lambda_anon,
        attribute_rankers=None,
        attributes=None,
        gmm_prior=None,
        generator=None,
        interpolated_embeddings=None
    ):
        """
        Compute combined generator loss.
        """
        device = transformed_embeddings.device
        
        # 1. Adversarial loss
        loss_adv = self.adversarial_loss(fake_logits) if self.lambda_adv > 0 else torch.tensor(0.0, device=device)
        
        # 2. Attribute matching loss (from discriminator)
        loss_attr = 0.0
        if self.lambda_attr > 0 and predicted_attr_deltas is not None:
            loss_attr = self.attribute_matching_loss(
                predicted_attr_deltas,
                target_attr_deltas
            )
        
        # 3. Ranker-based attribute loss
        loss_ranker = 0.0
        if self.lambda_ranker > 0 and attribute_rankers is not None and attributes is not None:
            loss_ranker = self.ranker_based_attribute_loss(
                attribute_rankers,
                original_embeddings,
                transformed_embeddings,
                target_attr_deltas,
                attributes
            )
        
        # 4. Distance control loss
        loss_dist = 0.0
        if self.lambda_dist > 0:
            loss_dist = self.distance_control_loss(
                transformed_embeddings,
                original_embeddings,
                lambda_anon
            )
        
        # 5. Smoothness loss
        loss_smooth = 0.0
        if self.lambda_smooth > 0 and interpolated_embeddings is not None:
            loss_smooth = self.smoothness_loss(interpolated_embeddings)
        
        # 6. GMM prior loss (CORRECTED scaling)
        loss_gmm = 0.0
        if self.lambda_gmm_prior > 0 and gmm_prior is not None:
            loss_gmm = self.gmm_prior_loss(transformed_embeddings, gmm_prior)
        
        # 7. NEW: Cycle consistency loss
        loss_cycle = 0.0
        if self.lambda_cycle > 0 and generator is not None:
            loss_cycle = self.cycle_consistency_loss(
                generator,
                original_embeddings,
                target_attr_deltas,
                lambda_anon
            )
        
        # 8. NEW: Orthogonality loss
        loss_ortho = 0.0
        if self.lambda_ortho > 0 and generator is not None:
            loss_ortho = self.orthogonality_loss(
                generator,
                original_embeddings,
                target_attr_deltas,
                lambda_anon
            )
        
        # Total loss
        total_loss = (
            self.lambda_adv * loss_adv +
            self.lambda_attr * loss_attr +
            self.lambda_ranker * loss_ranker +
            self.lambda_dist * loss_dist +
            self.lambda_smooth * loss_smooth +
            self.lambda_gmm_prior * loss_gmm +
            self.lambda_cycle * loss_cycle +
            self.lambda_ortho * loss_ortho
        )
        
        # Return individual components for logging
        return {
            'total': total_loss,
            'adversarial': loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv,
            'attribute': loss_attr.item() if isinstance(loss_attr, torch.Tensor) else loss_attr,
            'ranker': loss_ranker.item() if isinstance(loss_ranker, torch.Tensor) else loss_ranker,
            'distance': loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist,
            'smoothness': loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth,
            'gmm_prior': loss_gmm.item() if isinstance(loss_gmm, torch.Tensor) else loss_gmm,
            'cycle': loss_cycle.item() if isinstance(loss_cycle, torch.Tensor) else loss_cycle,
            'orthogonality': loss_ortho.item() if isinstance(loss_ortho, torch.Tensor) else loss_ortho
        }



class DiscriminatorLoss(nn.Module):
    """Loss for discriminator training"""
    def __init__(self, use_wasserstein=False, gradient_penalty_weight=10.0):
        super().__init__()
        self.use_wasserstein = use_wasserstein
        self.gp_weight = gradient_penalty_weight
    
    def forward(self, real_logits, fake_logits):
        """Standard discriminator loss"""
        if self.use_wasserstein:
            # Wasserstein loss
            return fake_logits.mean() - real_logits.mean()
        else:
            # Standard GAN loss
            real_loss = F.binary_cross_entropy_with_logits(
                real_logits,
                torch.ones_like(real_logits)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.zeros_like(fake_logits)
            )
            return real_loss + fake_loss
    
    def gradient_penalty(self, discriminator, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        
        # Interpolate between real and fake
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # Get discriminator output
        d_interpolates, _ = discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.gp_weight * penalty
