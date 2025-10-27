"""
Phase 1: Train attribute ranking networks
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.attribute_ranker import MultiAttributeRanker
from models.losses import RankNetLoss


class AttributeRankerTrainer(pl.LightningModule):
    """
    Lightning module for training attribute rankers.
    """
    def __init__(
        self,
        embedding_dim=192,
        attributes=['age', 'gender', 'pitch', 'voice_quality'],
        hidden_dims=[256, 128],
        learning_rate=1e-3,
        weight_decay=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.ranker = MultiAttributeRanker(
            embedding_dim=embedding_dim,
            attributes=attributes,
            hidden_dims=hidden_dims
        )
        
        # Loss
        self.criterion = RankNetLoss()
        
        # For tracking accuracy
        self.train_accuracies = {attr: [] for attr in attributes}
        self.val_accuracies = {attr: [] for attr in attributes}
    
    def forward(self, emb_i, emb_j, attribute):
        return self.ranker(emb_i, emb_j, attribute)
    
    def training_step(self, batch, batch_idx):
        """
        Batch contains pairwise comparisons for a specific attribute.
        """
        emb_i = batch['emb_i']
        emb_j = batch['emb_j']
        labels = batch['label']
        attribute = batch['attribute']  # All samples in batch are for same attribute
        
        # Get ranking predictions
        prob, score_i, score_j = self.ranker(emb_i, emb_j, attribute)
        
        # Compute loss
        loss = self.criterion(score_i, score_j, labels)
        
        # Compute accuracy
        predicted = (prob > 0.5).float() * 2 - 1  # Convert to {-1, 1}
        correct = (predicted == labels).float().mean()
        
        # Log metrics
        self.log(f'train_loss_{attribute}', loss, prog_bar=True)
        self.log(f'train_acc_{attribute}', correct, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        emb_i = batch['emb_i']
        emb_j = batch['emb_j']
        labels = batch['label']
        attribute = batch['attribute']
        
        # Get predictions
        prob, score_i, score_j = self.ranker(emb_i, emb_j, attribute)
        loss = self.criterion(score_i, score_j, labels)
        
        # Accuracy
        predicted = (prob > 0.5).float() * 2 - 1
        correct = (predicted == labels).float().mean()
        
        # Log
        self.log(f'val_loss_{attribute}', loss, prog_bar=True)
        self.log(f'val_acc_{attribute}', correct, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
