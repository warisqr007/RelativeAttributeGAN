"""
Generator network for transforming speaker embeddings with attribute control
"""
import torch
import torch.nn as nn


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
        x = torch.cat([embeddings, attribute_deltas, lambda_anon], dim=1)
        
        # Transform through network
        x = self.input_proj(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        # Project to embedding space
        delta = self.output_proj(x)
        
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
