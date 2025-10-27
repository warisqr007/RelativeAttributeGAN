"""
Pairwise ranking networks for speaker attributes (age, gender, pitch, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeRanker(nn.Module):
    """
    RankNet-style pairwise ranking for a single attribute.
    Takes two embeddings and predicts relative ordering.
    """
    def __init__(
        self,
        embedding_dim=192,
        hidden_dims=[256, 128],
        dropout=0.3,
        temperature=1.0
    ):
        super().__init__()
        self.temperature = temperature
        
        # Build MLP for scoring embeddings
        layers = []
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final scoring layer (outputs scalar score)
        layers.append(nn.Linear(in_dim, 1))
        
        self.scorer = nn.Sequential(*layers)
    
    def forward(self, emb_i, emb_j=None):
        """
        Args:
            emb_i: Tensor (batch, embedding_dim) - first embedding
            emb_j: Tensor (batch, embedding_dim) - second embedding (optional)
            
        Returns:
            If emb_j is None: scores for emb_i
            If emb_j provided: probability that emb_i > emb_j
        """
        score_i = self.scorer(emb_i).squeeze(-1)  # (batch,)
        
        if emb_j is None:
            return score_i
        
        score_j = self.scorer(emb_j).squeeze(-1)  # (batch,)
        
        # RankNet probability: P(i > j)
        score_diff = self.temperature * (score_i - score_j)
        prob_i_greater = torch.sigmoid(score_diff)
        
        return prob_i_greater, score_i, score_j


class MultiAttributeRanker(nn.Module):
    """
    Manages multiple attribute rankers (age, gender, pitch, etc.)
    """
    def __init__(
        self,
        embedding_dim=192,
        attributes=['age', 'gender', 'pitch', 'voice_quality'],
        hidden_dims=[256, 128],
        dropout=0.3
    ):
        super().__init__()
        self.attributes = attributes
        
        # Create a separate ranker for each attribute
        self.rankers = nn.ModuleDict({
            attr: AttributeRanker(
                embedding_dim=embedding_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            for attr in attributes
        })
    
    def forward(self, emb_i, emb_j, attribute):
        """Rank two embeddings for a specific attribute"""
        return self.rankers[attribute](emb_i, emb_j)
    
    def score(self, embeddings, attribute):
        """Get absolute scores for embeddings on an attribute"""
        return self.rankers[attribute](embeddings)
    
    def get_all_scores(self, embeddings):
        """Get scores for all attributes"""
        scores = {}
        for attr in self.attributes:
            scores[attr] = self.score(embeddings, attr)
        return scores
