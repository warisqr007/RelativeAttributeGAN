"""
Multi-task discriminator for GAN training
"""
import torch
import torch.nn as nn


class MultiTaskDiscriminator(nn.Module):
    """
    Discriminator with multiple heads:
    1. Real/Fake classification
    2. Attribute verification (did attributes change as requested?)
    """
    def __init__(
        self,
        embedding_dim=192,
        num_attributes=4,
        hidden_dim=512,
        dropout=0.3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attributes = num_attributes
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Head 1: Real/Fake classification
        self.real_fake_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)  # No sigmoid - using BCEWithLogitsLoss
        )
        
        # Head 2: Attribute change verification
        # Takes concatenation of original and transformed embeddings
        self.attr_verifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, num_attributes)  # Predict actual attribute changes
        )
    
    def forward(self, embeddings, original_embeddings=None):
        """
        Args:
            embeddings: (batch, embedding_dim) - embeddings to classify
            original_embeddings: (batch, embedding_dim) - optional, for attribute verification
            
        Returns:
            real_fake_logits: (batch, 1) - real/fake classification logits
            attr_changes: (batch, num_attributes) - predicted attribute changes (if original provided)
        """
        # Shared features
        features = self.shared(embeddings)
        
        # Real/Fake classification
        real_fake_logits = self.real_fake_head(features)
        
        # Attribute verification (only if original embeddings provided)
        if original_embeddings is not None:
            concat = torch.cat([original_embeddings, embeddings], dim=1)
            attr_changes = self.attr_verifier(concat)
            return real_fake_logits, attr_changes
        
        return real_fake_logits, None


class SpectralNorm(nn.Module):
    """Spectral normalization wrapper for stabilizing discriminator training"""
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        
        height = w.data.shape
        width = w.view(height, -1).data.shape
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", nn.Parameter(w.data))
    
    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data)
            )
            u.data = self._l2normalize(
                torch.mv(w.view(height, -1).data, v.data)
            )
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
