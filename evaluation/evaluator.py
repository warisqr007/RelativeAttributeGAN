"""
CORRECTED: Complete evaluation suite
"""
import torch
import numpy as np
import pandas as pd
from .metrics import (
    evaluate_anonymization,
    evaluate_linkability,
    compute_ranking_accuracy,
    evaluate_attribute_preservation
)


class GANEvaluator:
    """
    CORRECTED: Complete evaluation suite for the Relative Attribute GAN.
    
    Properly evaluates:
    1. Anonymization effectiveness (EER, linkability)
    2. Attribute control accuracy
    3. Generation quality
    4. Utility preservation (if vocoder available)
    """
    def __init__(
        self,
        generator,
        discriminator,
        attribute_rankers,
        speaker_encoder=None  # Now optional but recommended
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.rankers = attribute_rankers
        self.encoder = speaker_encoder
    
    def evaluate_all(
        self,
        test_embeddings,
        test_metadata,
        lambda_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ):
        """
        CORRECTED: Run all evaluation metrics.
        
        Args:
            test_embeddings: (N, embedding_dim) test embeddings
            test_metadata: DataFrame with speaker_id and attribute columns
            lambda_values: Anonymization levels to test
            
        Returns:
            results: Dictionary with all evaluation metrics
        """
        print("\n" + "="*70)
        print("Running Comprehensive Evaluation")
        print("="*70)
        
        results = {}
        device = next(self.generator.parameters()).device
        test_embeddings = test_embeddings.to(device)
        speaker_ids = [int(sid[2:]) for sid in test_metadata['speaker_id'].values]
        speaker_ids = torch.tensor(speaker_ids)
        
        # 1. CORRECTED: Anonymization effectiveness (EER)
        print("\n1. Evaluating Anonymization Effectiveness...")
        results['anonymization'] = evaluate_anonymization(
            self.generator,
            test_embeddings,
            speaker_ids,
            lambda_values=lambda_values,
            num_trials=3
        )
        
        # 2. Linkability at different lambda values
        print("\n2. Evaluating Linkability...")
        results['linkability'] = {}
        
        for lambda_val in lambda_values:
            print(f"  Evaluating λ_anon = {lambda_val:.1f}...")

            N = len(test_embeddings)
            lambda_tensor = torch.FloatTensor(N, 1).fill_(lambda_val).to(device)
            zero_deltas = torch.zeros(N, self.generator.num_attributes).to(device)
            
            with torch.no_grad():
                anonymized = self.generator(
                    test_embeddings,
                    zero_deltas,
                    lambda_tensor
                )
            
            link_score, link_eer = evaluate_linkability(
                test_embeddings.cpu(),
                anonymized.cpu(),
                speaker_ids
            )
            
            results['linkability'][f'lambda_{lambda_val}'] = {
                'linkability_score': link_score,
                'eer': link_eer
            }
        
        # 3. Ranking accuracy preservation
        print("\n3. Evaluating Ranking Accuracy...")
        for attr in self.rankers.attributes:
            print(f"  Evaluating attribute: {attr}...")
            if attr in test_metadata.columns:
                results[f'ranking_acc_{attr}'] = compute_ranking_accuracy(
                    self.rankers.rankers[attr],
                    test_embeddings,
                    test_metadata[attr].values
                )
        
        # 4. Attribute preservation during anonymization
        print("\n4. Evaluating Attribute Control...")
        # Test with various attribute changes
        N = min(len(test_embeddings), 100)  # Subsample for speed
        test_deltas = torch.randn(N, self.generator.num_attributes) * 0.5
        test_lambda = torch.FloatTensor(N, 1).uniform_(0.3, 0.8)
        
        attr_accuracy = evaluate_attribute_preservation(
            self.generator,
            self.rankers,
            test_embeddings[:N],
            test_deltas.to(device),
            test_lambda.to(device)
        )
        results['attribute_control_accuracy'] = attr_accuracy
        
        # 5. Generation quality (discriminator confusion)
        print("\n5. Evaluating Generation Quality...")
        with torch.no_grad():
            # Test at different lambda values
            quality_scores = {}
            for lambda_val in [0.3, 0.5, 0.7]:
                print(f"  Evaluating λ_anon = {lambda_val:.1f}...")
                lambda_tensor = torch.FloatTensor(N, 1).fill_(lambda_val).to(device)
                zero_deltas = torch.zeros(N, self.generator.num_attributes).to(device)
                
                fake_embeddings = self.generator(
                    test_embeddings[:N],
                    zero_deltas,
                    lambda_tensor
                )
                
                fake_logits, _ = self.discriminator(fake_embeddings)
                quality = torch.sigmoid(fake_logits).mean().item()
                quality_scores[f'lambda_{lambda_val}'] = quality
            
            results['generation_quality'] = quality_scores
        
        # 6. Distance control accuracy
        print("\n6. Evaluating Distance Control...")
        results['distance_control'] = self._evaluate_distance_control(
            test_embeddings[:N],
            lambda_values
        )
        
        return results
    
    def _evaluate_distance_control(
        self,
        test_embeddings,
        lambda_values
    ):
        """
        Check if lambda_anon actually controls distance as expected.
        Expected: cosine_similarity = 1 - lambda_anon
        """
        device = next(self.generator.parameters()).device
        N = len(test_embeddings)
        
        results = {}
        
        for lambda_val in lambda_values:
            lambda_tensor = torch.FloatTensor(N, 1).fill_(lambda_val).to(device)
            zero_deltas = torch.zeros(N, self.generator.num_attributes).to(device)
            
            with torch.no_grad():
                anonymized = self.generator(
                    test_embeddings,
                    zero_deltas,
                    lambda_tensor
                )
            
            # Compute actual similarity
            actual_sim = torch.cosine_similarity(
                test_embeddings,
                anonymized,
                dim=1
            ).mean().item()
            
            # Expected similarity
            expected_sim = 1.0 - lambda_val
            
            # Error
            error = abs(actual_sim - expected_sim)
            
            results[f'lambda_{lambda_val}'] = {
                'expected_similarity': expected_sim,
                'actual_similarity': actual_sim,
                'error': error
            }
        
        return results
    
    def print_results(self, results):
        """Pretty print evaluation results."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Anonymization effectiveness
        print("\n📊 Anonymization Effectiveness (EER):")
        print("-" * 70)
        for key, value in results['anonymization'].items():
            lambda_val = key.split('_')
            eer = value['eer']
            sim = value['mean_similarity']
            print(f"  λ={lambda_val}: EER={eer:.4f} ± {value['eer_std']:.4f}, "
                  f"Similarity={sim:.3f} ± {value['similarity_std']:.3f}")
        
        # Linkability
        print("\n🔗 Linkability (Lower is Better):")
        print("-" * 70)
        for key, value in results['linkability'].items():
            lambda_val = key.split('_')
            print(f"  λ={lambda_val}: Link Success={value['linkability_score']:.2%}, "
                  f"EER={value['eer']:.4f}")
        
        # Ranking accuracy
        print("\n📈 Attribute Ranking Accuracy:")
        print("-" * 70)
        for key, value in results.items():
            if key.startswith('ranking_acc'):
                attr = key.replace('ranking_acc_', '')
                print(f"  {attr}: {value:.2%}")
        
        # Attribute control
        print(f"\n🎯 Attribute Control Accuracy: {results['attribute_control_accuracy']:.2%}")
        
        # Generation quality
        print("\n✨ Generation Quality (Discriminator Confusion):")
        print("-" * 70)
        for key, value in results['generation_quality'].items():
            lambda_val = key.split('_')
            print(f"  λ={lambda_val}: {value:.2%} (target: ~50%)")
        
        # Distance control
        print("\n📏 Distance Control Accuracy:")
        print("-" * 70)
        for key, value in results['distance_control'].items():
            lambda_val = key.split('_')
            print(f"  λ={lambda_val}: Expected={value['expected_similarity']:.3f}, "
                  f"Actual={value['actual_similarity']:.3f}, "
                  f"Error={value['error']:.3f}")
