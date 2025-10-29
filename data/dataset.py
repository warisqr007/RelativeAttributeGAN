"""
Dataset classes for training
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path


class SpeakerEmbeddingDataset(Dataset):
    """
    Dataset of pre-extracted speaker embeddings with metadata.
    Assumes embeddings have been extracted and saved.
    """
    def __init__(
        self,
        embeddings_path,
        metadata_path,
        attributes=['age', 'gender', 'pitch', 'voice_quality']
    ):
        """
        Args:
            embeddings_path: Path to .pt file containing embeddings tensor
            metadata_path: Path to CSV with columns: speaker_id, age, gender, etc.
            attributes: List of attribute names to use
        """
        self.embeddings = torch.load(embeddings_path)
        self.metadata = pd.read_csv(metadata_path)
        self.attributes = attributes
        
        assert len(self.embeddings) == len(self.metadata), \
            "Embeddings and metadata length mismatch"
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        meta = self.metadata.iloc[idx]
        
        # Extract attribute values
        attr_values = torch.tensor([
            meta[attr] if attr in meta else 0.0
            for attr in self.attributes
        ], dtype=torch.float32)
        
        return {
            'embedding': embedding,
            'attributes': attr_values,
            'speaker_id': meta['speaker_id']
        }


class PairwiseRankingDataset(Dataset):
    """
    Dataset for training attribute rankers with pairwise comparisons.
    """
    def __init__(
        self,
        # embeddings,
        metadata,
        attribute,
        num_pairs_per_sample=5
    ):
        """
        Args:
            embeddings: Tensor (num_speakers, embedding_dim)
            metadata: DataFrame with attribute values
            attribute: String, which attribute to rank on
            num_pairs_per_sample: How many comparison pairs to generate per speaker
        """
        # self.embeddings = embeddings
        self.embeddings_fpaths = metadata['embedding_path'].values
        self.metadata = metadata
        self.attribute = attribute
        self.num_pairs = num_pairs_per_sample
        
        # Generate all comparison pairs
        self.pairs = self._generate_pairs()
    
    def _generate_pairs(self):
        """Generate pairwise comparisons based on attribute values"""
        pairs = []
        # n = len(self.embeddings)
        n = len(self.metadata)
        
        # Get attribute values
        attr_values = self.metadata[self.attribute].values
        
        # Generate pairs where we know the ranking
        for i in range(n):
            # Find speakers with different attribute values
            candidates_greater = np.where(attr_values > attr_values[i])[0]
            candidates_less = np.where(attr_values < attr_values[i])[0]

            # print(len(attr_values))
            # print(np.array(attr_values).shape)
            # print(len(candidates_greater), len(candidates_less))
            # print(len(candidates_greater[0]), len(candidates_less[0]))
            # since gender is binary, we can only form pairs between the two classes
            if self.attribute == 'gender':
                for _ in range(self.num_pairs):
                    if len(candidates_greater) > 0:
                        j = np.random.choice(candidates_greater)
                        label = -1  # j > i
                        pairs.append((i, j, label))
                    if len(candidates_less) > 0:
                        j = np.random.choice(candidates_less)
                        label = 1  # i > j
                        pairs.append((i, j, label))
            else:
                # Sample random pairs
                for _ in range(self.num_pairs):
                    if len(candidates_greater) > 0 and len(candidates_less) > 0:
                        # Randomly choose whether i should be greater or less
                        if np.random.rand() > 0.5 and len(candidates_greater) > 0:
                            j = np.random.choice(candidates_greater)
                            label = -1  # j > i
                        elif len(candidates_less) > 0:
                            j = np.random.choice(candidates_less)
                            label = 1  # i > j
                        else:
                            continue
                        
                        pairs.append((i, j, label))

        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]

        embed_fp_i = self.embeddings_fpaths[i]
        embed_fp_j = self.embeddings_fpaths[j]

        emb_i = np.load(embed_fp_i)
        emb_j = np.load(embed_fp_j)

        
        return {
            'emb_i': torch.from_numpy(emb_i),
            'emb_j': torch.from_numpy(emb_j),
            'label': torch.tensor(label, dtype=torch.float32),
            'attribute': self.attribute
        }


class GANTrainingDataset(Dataset):
    """
    Dataset for GAN training - samples random attribute transformations.
    """
    def __init__(
        self,
        # embeddings,
        metadata,
        attributes=['age', 'gender', 'pitch', 'voice_quality'],
        delta_range=(-1.0, 1.0),
        lambda_anon_range=(0.0, 1.0)
    ):
        # self.embeddings = embeddings
        self.embeddings_fpaths = metadata['embedding_path'].values
        self.metadata = metadata
        self.attributes = attributes
        self.delta_range = delta_range
        self.lambda_range = lambda_anon_range
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        embedding = np.load(self.embeddings_fpaths[idx])
        embedding = torch.from_numpy(embedding)
        
        # Sample random attribute deltas
        deltas = torch.FloatTensor(len(self.attributes)).uniform_(
            self.delta_range[0],
            self.delta_range[1]
        )
        
        # Sample random anonymization level
        lambda_anon = torch.FloatTensor(1).uniform_(
            self.lambda_range[0],
            self.lambda_range[1]
        )
        
        return {
            'embedding': embedding,
            'deltas': deltas,
            'lambda_anon': lambda_anon
        }
