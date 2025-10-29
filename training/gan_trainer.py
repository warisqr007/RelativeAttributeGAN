import torch
import pytorch_lightning as pl
from models.generator import AttributeControlGenerator
from models.discriminator import MultiTaskDiscriminator
from models.attribute_ranker import MultiAttributeRanker
from models.losses import CombinedGANLoss, DiscriminatorLoss
from models.gmm import GMMNLLPrior


class RelativeAttributeGAN(pl.LightningModule):
    """
    Complete GAN system with manual optimization for generator and discriminator.
    Now properly integrates pretrained attribute rankers.
    """
    def __init__(
        self,
        embedding_dim=192,
        num_attributes=4,
        attributes=['age', 'gender', 'pitch', 'voice_quality'],  # NEW: attribute names
        generator_hidden_dim=512,
        discriminator_hidden_dim=512,
        num_residual_blocks=6,
        lr_g=1e-4,
        lr_d=4e-4,
        lambda_attr=1.0,
        lambda_dist=1.0,
        lambda_smooth=0.1,
        lambda_ranker=0.5,  # NEW: Weight for ranker loss
        lambda_gmm_prior=0.5,  # NEW
        ranker_checkpoint_dir=None,  # NEW: Path to pretrained rankers
        gmm_checkpoint_path=None,
        gmm_num_components=64,
        curriculum_schedule=None,
        d_steps_per_g_step=2
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Enable manual optimization for GANs
        self.automatic_optimization = False
        
        # Models
        self.generator = AttributeControlGenerator(
            embedding_dim=embedding_dim,
            num_attributes=num_attributes,
            hidden_dim=generator_hidden_dim,
            num_residual_blocks=num_residual_blocks
        )
        
        self.discriminator = MultiTaskDiscriminator(
            embedding_dim=embedding_dim,
            num_attributes=num_attributes,
            hidden_dim=discriminator_hidden_dim
        )
        
        # NEW: Load pretrained attribute rankers (FROZEN)
        self.attribute_rankers = MultiAttributeRanker(
            embedding_dim=embedding_dim,
            attributes=attributes
        )
        
        if ranker_checkpoint_dir:
            self._load_rankers(ranker_checkpoint_dir, attributes)

        
        self.gmm_prior = GMMNLLPrior(emb_dim=embedding_dim, num_components=gmm_num_components)
        if gmm_checkpoint_path:
            self._load_gmm(gmm_checkpoint_path)
        
        # Freeze rankers - they provide supervisory signal only
        for param in self.attribute_rankers.parameters():
            param.requires_grad = False
        self.attribute_rankers.eval()
        
        # Losses
        self.g_loss_fn = CombinedGANLoss(
            lambda_attr=lambda_attr,
            lambda_dist=lambda_dist,
            lambda_smooth=lambda_smooth,
            lambda_ranker=lambda_ranker,  # NEW
            lambda_gmm_prior=lambda_gmm_prior  # NEW
        )
        
        self.d_loss_fn = DiscriminatorLoss()
        
        # Progressive training curriculum
        self.curriculum = curriculum_schedule or self._default_curriculum()
        self.current_epoch_in_curriculum = 0
    
    def _load_rankers(self, checkpoint_dir, attributes):
        """Load pretrained ranker checkpoints"""
        import os
        
        for attr in attributes:
            checkpoint_path = os.path.join(checkpoint_dir, attr, 'best.ckpt')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                # Extract ranker state dict from Lightning checkpoint
                state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith(f'ranker.{attr}'):
                        new_key = key.replace(f'ranker.{attr}.', '')
                        state_dict[new_key] = value
                
                self.attribute_rankers.rankers[attr].load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded pretrained ranker for {attr}")
            else:
                print(f"⚠ Warning: Ranker checkpoint not found for {attr} at {checkpoint_path}")
    
    def _load_gmm(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert self.gmm_prior.num_components == ckpt["pi"].numel(), \
            "GMM component count mismatch."
        assert self.gmm_prior.emb_dim == ckpt["mu"].size(1), \
            "GMM embedding dimension mismatch."
        
        with torch.no_grad():
            self.gmm_prior.load_gmm(ckpt["pi"], ckpt["mu"], ckpt["var"])

        print(f"✓ Loaded GMM prior from {checkpoint_path}")

    def _default_curriculum(self):
        """Default progressive training schedule"""
        return {
            'phase1': {  # Epochs 0-20: Learn basic transformations
                'epochs': 20,
                'lambda_attr': 0.0,  # No attribute matching yet
                'lambda_ranker': 0.0,  # NEW: No ranker loss yet
                'lambda_dist': 1.0,
                'lambda_smooth': 0.0,
                'delta_range': (-0.3, 0.3),  # Small changes
                'lambda_anon_range': (0.0, 0.3)
            },
            'phase2': {  # Epochs 20-50: Add attribute control
                'epochs': 30,
                'lambda_attr': 0.5,
                'lambda_ranker': 0.3,  # NEW: Start using ranker loss
                'lambda_dist': 1.0,
                'lambda_smooth': 0.0,
                'delta_range': (-0.6, 0.6),
                'lambda_anon_range': (0.0, 0.6)
            },
            'phase3': {  # Epochs 50+: Full training with smoothness
                'epochs': 50,
                'lambda_attr': 1.0,
                'lambda_ranker': 0.5,  # NEW: Full ranker loss
                'lambda_dist': 1.0,
                'lambda_smooth': 0.1,
                'delta_range': (-1.0, 1.0),
                'lambda_anon_range': (0.0, 1.0)
            }
        }
    
    def get_current_phase(self):
        """Determine which curriculum phase we're in"""
        epoch = self.current_epoch
        cumulative = 0
        
        for phase_name, phase_config in self.curriculum.items():
            cumulative += phase_config['epochs']
            if epoch < cumulative:
                return phase_name, phase_config
        
        # If beyond all phases, use last phase config
        return 'phase3', self.curriculum['phase3']
    
    def forward(self, embeddings, attribute_deltas, lambda_anon):
        """Generate transformed embeddings"""
        return self.generator(embeddings, attribute_deltas, lambda_anon)
    
    def training_step(self, batch, batch_idx):
        """
        Manual optimization for GAN training.
        We train discriminator multiple times per generator step.
        """
        g_opt, d_opt = self.optimizers()
        
        # Get current curriculum phase
        phase_name, phase_config = self.get_current_phase()
        
        # Update loss weights based on curriculum
        self.g_loss_fn.lambda_attr = phase_config['lambda_attr']
        self.g_loss_fn.lambda_ranker = phase_config.get('lambda_ranker', 0.0)  # NEW
        self.g_loss_fn.lambda_dist = phase_config['lambda_dist']
        self.g_loss_fn.lambda_smooth = phase_config['lambda_smooth']
        self.g_loss_fn.lambda_gmm_prior = phase_config.get('lambda_gmm_prior', 0.5)
        
        # Unpack batch
        real_embeddings = batch['embedding']
        batch_size = real_embeddings.shape[0]
        
        # Sample attribute deltas and anonymization levels from curriculum range
        delta_range = phase_config['delta_range']
        lambda_range = phase_config['lambda_anon_range']
        
        attribute_deltas = torch.FloatTensor(
            batch_size, self.hparams.num_attributes
        ).uniform_(delta_range[0], delta_range[1]).to(self.device)
        
        lambda_anon = torch.FloatTensor(
            batch_size, 1
        ).uniform_(lambda_range[0], lambda_range[1]).to(self.device)
        
        ##########################
        # (1) Update Discriminator
        ##########################
        for _ in range(self.hparams.d_steps_per_g_step):
            d_opt.zero_grad()
            
            # Generate fake embeddings
            fake_embeddings = self.generator(
                real_embeddings,
                attribute_deltas,
                lambda_anon
            )
            
            # Discriminator predictions
            real_logits, _ = self.discriminator(real_embeddings)
            fake_logits, _ = self.discriminator(fake_embeddings.detach())
            
            # Discriminator loss
            d_loss = self.d_loss_fn(real_logits, fake_logits)
            
            # Backward and optimize
            self.manual_backward(d_loss)
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            
            d_opt.step()
        
        ######################
        # (2) Update Generator
        ######################
        g_opt.zero_grad()
        
        # Generate fake embeddings (no detach - need gradients)
        fake_embeddings = self.generator(
            real_embeddings,
            attribute_deltas,
            lambda_anon
        )
        
        # Discriminator predictions on fakes
        fake_logits, predicted_attr_deltas = self.discriminator(
            fake_embeddings,
            original_embeddings=real_embeddings
        )
        
        # Generator loss (all components, INCLUDING ranker-based loss)
        g_loss_dict = self.g_loss_fn(
            fake_logits=fake_logits,
            predicted_attr_deltas=predicted_attr_deltas,
            target_attr_deltas=attribute_deltas,
            transformed_embeddings=fake_embeddings,
            original_embeddings=real_embeddings,
            lambda_anon=lambda_anon,
            attribute_rankers=self.attribute_rankers,  # NEW: Pass rankers
            attributes=self.hparams.attributes,  # NEW: Pass attribute names
            gmm_prior=self.gmm_prior,  # NEW: Pass GMM prior
            interpolated_embeddings=None  # Can add for smoothness loss
        )
        
        g_loss = g_loss_dict['total']
        
        # Backward and optimize
        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        g_opt.step()
        
        # Logging
        self.log('d_loss', d_loss, prog_bar=True, batch_size=batch_size)
        self.log('g_loss_total', g_loss, prog_bar=True, batch_size=batch_size)
        self.log('g_loss_adv', g_loss_dict['adversarial'], prog_bar=False, batch_size=batch_size)
        self.log('g_loss_attr', g_loss_dict['attribute'], prog_bar=False, batch_size=batch_size)
        self.log('g_loss_ranker', g_loss_dict['ranker'], prog_bar=True, batch_size=batch_size)  # NEW
        self.log('g_loss_dist', g_loss_dict['distance'], prog_bar=False, batch_size=batch_size)
        self.log('g_gmm_prior', g_loss_dict['gmm_prior'], prog_bar=False, batch_size=batch_size)  # NEW
        self.log('phase', float(phase_name[-1]), prog_bar=True, batch_size=batch_size)
        
        # Monitor discriminator accuracy (should stay around 70-80%)
        with torch.no_grad():
            real_pred = (torch.sigmoid(real_logits) > 0.5).float().mean()
            fake_pred = (torch.sigmoid(fake_logits) < 0.5).float().mean()
            d_acc = (real_pred + fake_pred) / 2
        
        self.log('d_accuracy', d_acc, prog_bar=True, batch_size=batch_size)
    
    def validation_step(self, batch, batch_idx):
        """Validate generator quality"""
        real_embeddings = batch['embedding']
        batch_size = real_embeddings.shape[0]
        
        # Fixed test transformations
        attribute_deltas = torch.zeros(
            batch_size, self.hparams.num_attributes
        ).to(self.device)
        attribute_deltas[:, 0] = 0.5  # Increase age
        
        lambda_anon = torch.FloatTensor(batch_size, 1).fill_(0.5).to(self.device)
        
        # Generate
        fake_embeddings = self.generator(
            real_embeddings,
            attribute_deltas,
            lambda_anon
        )
        
        # Check quality
        fake_logits, _ = self.discriminator(fake_embeddings)
        
        # Distance achieved
        cos_sim = torch.nn.functional.cosine_similarity(
            fake_embeddings, real_embeddings, dim=1
        ).mean()
        
        # NEW: Verify ranker scores changed correctly
        with torch.no_grad():
            # Get age scores (first attribute, which we increased)
            age_scores_orig = self.attribute_rankers.score(real_embeddings, 'age')
            age_scores_transformed = self.attribute_rankers.score(fake_embeddings, 'age')
            avg_age_increase = (age_scores_transformed - age_scores_orig).mean()
        
        self.log('val_fake_quality', torch.sigmoid(fake_logits).mean(), batch_size=batch_size)
        self.log('val_cos_similarity', cos_sim, batch_size=batch_size)
        self.log('val_age_score_change', avg_age_increase, batch_size=batch_size)  # NEW

    def configure_optimizers(self):
        """Separate optimizers for generator and discriminator"""
        g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr_g,
            betas=(0.5, 0.999)
        )
        
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr_d,
            betas=(0.5, 0.999)
        )
        
        return [g_opt, d_opt], []
