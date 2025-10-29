"""
Generator network for transforming speaker embeddings with attribute control
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for generator"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return x + self.dropout(self.block(x))


class AttributeControlGenerator(nn.Module):
    """
    Generator that transforms speaker embeddings with attribute control.
    
    Architecture:
        Input: [embedding, delta_age, delta_gender, ..., lambda_anon]
        Output: transformed embedding
    """
    def __init__(
        self,
        embedding_dim=192,
        num_attributes=4,  # age, gender, pitch, voice_quality
        hidden_dim=512,
        num_residual_blocks=6,
        dropout=0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attributes = num_attributes
        
        # Total input: embedding + attribute deltas + anonymization strength
        input_dim = embedding_dim + num_attributes + 1
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks for transformation
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])
        
        # Output projection back to embedding space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Bounded output
        )
        
        # Learnable residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, embeddings, attribute_deltas, lambda_anon):
        """
        Transform embeddings with attribute control.
        
        Args:
            embeddings: (batch, embedding_dim) - original speaker embeddings
            attribute_deltas: (batch, num_attributes) - desired attribute changes
            lambda_anon: (batch, 1) - anonymization strength [0, 1]
            
        Returns:
            transformed_embeddings: (batch, embedding_dim)
        """
        batch_size = embeddings.shape
        
        # Ensure lambda_anon is 2D
        if lambda_anon.dim() == 1:
            lambda_anon = lambda_anon.unsqueeze(1)
        
        # Concatenate all inputs
        x = torch.cat([embeddings, attribute_deltas, lambda_anon], dim=1) ##TODO: tile instead of concat?
        
        # Transform through network
        x = self.input_proj(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        # Project to embedding space
        delta = self.output_proj(x) #TODO: Include composite attribute change directions?
        
        # Residual connection: e' = e + alpha * delta
        # The residual weight learns how much to change
        transformed = embeddings + self.residual_weight * delta
        
        # L2 normalize to keep on embedding manifold
        transformed = nn.functional.normalize(transformed, p=2, dim=1)
        
        return transformed
    
    def interpolate(self, embedding, attribute_deltas, num_steps=10):
        """
        Generate smooth interpolation from original to target attributes.
        Useful for evaluation and visualization.
        """
        interpolated = []
        
        for i in range(num_steps + 1):
            alpha = i / num_steps
            scaled_deltas = alpha * attribute_deltas
            lambda_val = torch.tensor([[alpha]], device=embedding.device)
            
            with torch.no_grad():
                transformed = self.forward(
                    embedding.unsqueeze(0),
                    scaled_deltas.unsqueeze(0),
                    lambda_val
                )
            interpolated.append(transformed.squeeze(0))
        
        return torch.stack(interpolated)


class FiLMResBlock(nn.Module):
    """
    Feature-wise Linear Modulation residual block.
    Modulates features based on attribute conditioning.
    """
    def __init__(self, dim, cond_dim, dropout=0.1):
        super().__init__()
        
        # Main transformation
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),  # Smooth activation
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        # FiLM: learn scale and shift from conditions
        self.scale_net = nn.Linear(cond_dim, dim)
        self.shift_net = nn.Linear(cond_dim, dim)
        
        # Initialize FiLM to identity (scale=1, shift=0)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.ones_(self.scale_net.weight)
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)
    
    def forward(self, x, cond):
        """
        Args:
            x: (batch, dim) - features to transform
            cond: (batch, cond_dim) - conditioning vector
        """
        # Compute FiLM parameters from conditioning
        scale = self.scale_net(cond)  # (batch, dim)
        shift = self.shift_net(cond)  # (batch, dim)
        
        # Transform features
        h = self.transform(x)
        
        # Apply FiLM modulation
        h = scale * h + shift
        
        # Residual connection
        return x + h


class DirectionRefiner(nn.Module):
    """
    Refines PCA-based attribute directions with learned non-linear adjustments.
    """
    def __init__(
        self,
        embedding_dim=192,
        num_attributes=4,
        hidden_dim=256,
        num_layers=2
    ):
        super().__init__()
        
        # Input: [embeddings, attribute_deltas] → Output: refinement vector
        layers = []
        in_dim = embedding_dim + num_attributes
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Output layer (small refinement)
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize to output small values initially
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, embeddings, attribute_deltas):
        """
        Args:
            embeddings: (batch, embedding_dim)
            attribute_deltas: (batch, num_attributes)
        
        Returns:
            refinement: (batch, embedding_dim) - small adjustment vector
        """
        x = torch.cat([embeddings, attribute_deltas], dim=1)
        refinement = self.net(x)
        return refinement


class HybridAttributeGenerator(nn.Module):
    """
    HYBRID: Combines PCA-based semantic directions with FiLM-conditioned residual blocks.
    
    Architecture:
    1. Compute composite direction from PCA basis + learned refinement
    2. Apply direction to embedding (initial transformation)
    3. Refine through FiLM-conditioned ResBlocks
    4. Final L2 normalization
    
    This gives you:
    - Interpretable, data-driven transformations (PCA)
    - Flexible, non-linear refinement (FiLM + ResBlocks)
    - Smooth attribute control (direction-based)
    - Expressive power (deep network)
    """
    
    def __init__(
        self,
        embedding_dim=192,
        num_attributes=4,
        attribute_names=['age', 'gender', 'pitch', 'voice_quality'],
        hidden_dim=256,
        num_resblocks=4,
        dropout=0.1,
        use_pca_directions=True,
        refinement_strength=0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attributes = num_attributes
        self.attribute_names = attribute_names
        self.use_pca_directions = use_pca_directions
        self.refinement_strength = refinement_strength
        
        # ========================================
        # 1. PCA-based Direction Subspace
        # ========================================
        if use_pca_directions:
            # Register buffers for precomputed PCA directions
            for attr in attribute_names:
                self.register_buffer(f'v_{attr}', torch.randn(embedding_dim))
            
            # Learnable scale factors for each direction
            self.direction_scales = nn.ParameterDict({
                attr: nn.Parameter(torch.tensor(1.0))
                for attr in attribute_names
            })
        
        # ========================================
        # 2. Direction Refinement Network
        # ========================================
        self.direction_refiner = DirectionRefiner(
            embedding_dim=embedding_dim,
            num_attributes=num_attributes,
            hidden_dim=hidden_dim,
            num_layers=2
        )
        
        # ========================================
        # 3. Magnitude Network
        # ========================================
        # Predicts how far to move along direction based on lambda_anon
        # Input: [lambda_anon, attribute_deltas] → Output: scalar magnitude
        cond_dim = 1 + num_attributes  # lambda + deltas
        self.magnitude_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive magnitude
        )
        
        # ========================================
        # 4. FiLM-Conditioned Residual Blocks
        # ========================================
        self.res_blocks = nn.ModuleList([
            FiLMResBlock(
                dim=embedding_dim,
                cond_dim=cond_dim,  # condition on [lambda_anon, deltas]
                dropout=dropout
            )
            for _ in range(num_resblocks)
        ])
        
        # ========================================
        # 5. Output Projection (Optional)
        # ========================================
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.eye_(self.output_proj.weight)  # Initialize to identity
        nn.init.zeros_(self.output_proj.bias)
    
    def load_precomputed_directions(self, directions_dict):
        """
        Load PCA-computed directions from offline analysis.
        
        Args:
            directions_dict: Dict with keys 'v_age', 'v_gender', etc.
        """
        if not self.use_pca_directions:
            print("Warning: PCA directions disabled, skipping load")
            return
        
        for attr in self.attribute_names:
            key = f'v_{attr}'
            if key in directions_dict:
                direction = directions_dict[key]
                # Normalize and copy
                direction_normalized = F.normalize(direction, dim=0)
                getattr(self, key).copy_(direction_normalized)
                print(f"✓ Loaded direction for {attr}")
            else:
                print(f"⚠ Warning: Direction for {attr} not found in dict")
    
    def compute_composite_direction(self, embeddings, attribute_deltas):
        """
        Compute composite direction:
        1. Linear combination of PCA basis directions (weighted by deltas)
        2. Add learned refinement (small non-linear adjustment)
        
        Args:
            embeddings: (batch, embedding_dim)
            attribute_deltas: (batch, num_attributes)
        
        Returns:
            direction: (batch, embedding_dim) - normalized direction vector
        """
        batch_size = embeddings.shape[0]
        direction = torch.zeros(batch_size, self.embedding_dim, device=embeddings.device)
        
        # 1. Linear combination of PCA directions
        if self.use_pca_directions:
            for i, attr in enumerate(self.attribute_names):
                # Get basis direction and scale
                v = getattr(self, f'v_{attr}')  # (embedding_dim,)
                scale = self.direction_scales[attr]
                
                # Add weighted direction
                # attribute_deltas[:, i:i+1] shape: (batch, 1)
                # v shape: (embedding_dim,)
                # Broadcasting: (batch, 1) * (embedding_dim,) = (batch, embedding_dim)
                direction = direction + attribute_deltas[:, i:i+1] * scale * v
        
        # 2. Add learned refinement (non-linear adjustment)
        refinement = self.direction_refiner(embeddings, attribute_deltas)
        direction = direction + self.refinement_strength * refinement
        
        # 3. Normalize to unit direction
        direction = F.normalize(direction, p=2, dim=1)
        
        return direction
    
    def forward(self, embeddings, attribute_deltas, lambda_anon):
        """
        Transform embeddings with hybrid approach.
        
        Args:
            embeddings: (batch, embedding_dim) - input embeddings
            attribute_deltas: (batch, num_attributes) - desired attribute changes
            lambda_anon: (batch, 1) - anonymization strength [0, 1]
        
        Returns:
            transformed: (batch, embedding_dim) - output embeddings
        """
        # Conditioning vector for FiLM (lambda + deltas)
        cond = torch.cat([lambda_anon, attribute_deltas], dim=1)  # (batch, 1+4)
        
        # ========================================
        # Step 1: Compute Transformation Direction
        # ========================================
        direction = self.compute_composite_direction(embeddings, attribute_deltas)
        
        # ========================================
        # Step 2: Compute Magnitude
        # ========================================
        # Magnitude depends on lambda_anon and attribute deltas
        magnitude = self.magnitude_net(cond)  # (batch, 1)
        
        # ========================================
        # Step 3: Initial Transformation
        # ========================================
        # Move along direction: e' = e + magnitude * direction
        transformed = embeddings + magnitude * direction
        
        # ========================================
        # Step 4: Refine through FiLM ResBlocks
        # ========================================
        # Each block refines the transformation based on conditioning
        for block in self.res_blocks:
            transformed = block(transformed, cond)
        
        # ========================================
        # Step 5: Output Projection
        # ========================================
        transformed = self.output_proj(transformed)
        
        # ========================================
        # Step 6: L2 Normalize (stay on embedding manifold)
        # ========================================
        transformed = F.normalize(transformed, p=2, dim=1)
        
        return transformed
    
    def get_direction_norms(self):
        """Get norms of learned direction scales (for monitoring)."""
        if not self.use_pca_directions:
            return {}
        
        norms = {}
        for attr in self.attribute_names:
            scale = self.direction_scales[attr]
            v = getattr(self, f'v_{attr}')
            norms[attr] = (scale * v.norm()).item()
        return norms
