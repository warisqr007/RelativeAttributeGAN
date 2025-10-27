"""
Evaluation metrics: EER, ranking accuracy, etc.
"""
import torch
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(scores, labels):
    """
    Compute Equal Error Rate for speaker verification.
    
    Args:
        scores: Array of similarity scores
        labels: Array of binary labels (1 = same speaker, 0 = different)
        
    Returns:
        eer: Equal Error Rate (lower is better)
        threshold: EER threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find where FPR = FNR
    eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer = interp1d(fpr, fnr)(eer_threshold)
    
    return float(eer), float(eer_threshold)


def compute_ranking_accuracy(ranker, embeddings, attribute_values):
    """
    Compute ranking accuracy: how often ranker orders pairs correctly.
    
    Args:
        ranker: Trained AttributeRanker model
        embeddings: (N, embedding_dim)
        attribute_values: (N,) ground truth attribute values
        
    Returns:
        accuracy: Fraction of correctly ordered pairs
    """
    n = len(embeddings)
    correct = 0
    total = 0
    
    # Test on random pairs
    for _ in range(min(1000, n * n)):
        i, j = np.random.choice(n, size=2, replace=False)
        
        # Skip if equal
        if attribute_values[i] == attribute_values[j]:
            continue
        
        # Get ranking prediction
        with torch.no_grad():
            prob, _, _ = ranker(
                embeddings[i:i+1],
                embeddings[j:j+1]
            )
        
        # Check if correct
        predicted_i_greater = prob.item() > 0.5
        actual_i_greater = attribute_values[i] > attribute_values[j]
        
        if predicted_i_greater == actual_i_greater:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_anonymization(
    generator,
    original_embeddings,
    speaker_ids,
    speaker_verifier,
    lambda_values=[0.0, 0.3, 0.5, 0.7, 1.0]
):
    """
    Evaluate anonymization effectiveness at different lambda levels.
    
    Args:
        generator: Trained generator
        original_embeddings: (N, embedding_dim)
        speaker_ids: (N,) speaker IDs for each embedding
        speaker_verifier: Pre-trained speaker verification model
        lambda_values: List of anonymization levels to test
        
    Returns:
        results: Dict with EER for each lambda value
    """
    results = {}
    
    for lambda_val in lambda_values:
        # Generate anonymized embeddings
        batch_size = len(original_embeddings)
        lambda_tensor = torch.FloatTensor(batch_size, 1).fill_(lambda_val)
        zero_deltas = torch.zeros(batch_size, generator.num_attributes)
        
        with torch.no_grad():
            anon_embeddings = generator(
                original_embeddings,
                zero_deltas,
                lambda_tensor
            )
        
        # Compute all pairwise similarities
        similarities = torch.mm(anon_embeddings, anon_embeddings.t())
        
        # Create labels (1 if same speaker, 0 if different)
        labels = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
        
        # Remove diagonal
        mask = ~torch.eye(len(labels), dtype=torch.bool)
        similarities = similarities[mask].cpu().numpy()
        labels = labels[mask].cpu().numpy()
        
        # Compute EER
        eer, threshold = compute_eer(similarities, labels)
        
        results[f'lambda_{lambda_val}'] = {
            'eer': eer,
            'threshold': threshold
        }
    
    return results


class GANEvaluator:
    """Complete evaluation suite for the GAN"""
    def __init__(
        self,
        generator,
        discriminator,
        attribute_rankers,
        speaker_encoder
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.rankers = attribute_rankers
        self.encoder = speaker_encoder
    
    def evaluate_all(self, test_embeddings, test_metadata):
        """Run all evaluation metrics"""
        results = {}
        
        # 1. Anonymization effectiveness (EER)
        results['anonymization'] = evaluate_anonymization(
            self.generator,
            test_embeddings,
            test_metadata['speaker_id'],
            self.encoder
        )
        
        # 2. Ranking accuracy preservation
        for attr in self.rankers.attributes:
            results[f'ranking_acc_{attr}'] = compute_ranking_accuracy(
                self.rankers.rankers[attr],
                test_embeddings,
                test_metadata[attr].values
            )
        
        # 3. Generation quality (discriminator confusion)
        with torch.no_grad():
            fake_logits, _ = self.discriminator(test_embeddings)
            results['generation_quality'] = torch.sigmoid(fake_logits).mean().item()
        
        return results
