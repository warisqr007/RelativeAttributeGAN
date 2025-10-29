import torch
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from itertools import combinations


def compute_eer(scores, labels):
    """
    Compute Equal Error Rate for speaker verification.
    
    Args:
        scores: Array of similarity scores (higher = more similar)
        labels: Array of binary labels (1 = same speaker, 0 = different)
        
    Returns:
        eer: Equal Error Rate (lower is better for anonymization)
        threshold: EER threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find where FPR = FNR
    eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer = interp1d(fpr, fnr)(eer_threshold)
    
    return float(eer), float(eer_threshold)


def evaluate_anonymization(
    generator,
    original_embeddings,
    speaker_ids,
    lambda_values=[0.0, 0.3, 0.5, 0.7, 1.0],
    num_trials=3
):
    """
    CORRECTED: Properly evaluate anonymization effectiveness.
    
    Key differences from buggy version:
    1. Actually generates anonymized embeddings
    2. Computes cosine similarity between ORIGINAL and ANONYMIZED
    3. Creates proper same/different speaker labels
    4. Computes EER from these similarities
    
    Args:
        generator: Trained generator model
        original_embeddings: (N, embedding_dim) original speaker embeddings
        speaker_ids: (N,) speaker IDs for each embedding
        lambda_values: List of anonymization levels to test
        num_trials: Number of random trials per lambda (for stability)
        
    Returns:
        results: Dict with EER and other metrics for each lambda value
    """
    generator.eval()
    device = next(generator.parameters()).device
    
    results = {}
    N = len(original_embeddings)
    num_attributes = generator.num_attributes
    
    for lambda_val in lambda_values:
        print(f"  Evaluating λ_anon = {lambda_val:.1f}...")
        
        trial_eers = []
        trial_similarities = []
        
        for trial in range(num_trials):
            # Generate anonymized embeddings with random attribute changes
            lambda_tensor = torch.FloatTensor(N, 1).fill_(lambda_val).to(device)
            
            # Random attribute deltas for this trial
            attribute_deltas = torch.randn(N, num_attributes).to(device) * 0.5
            
            with torch.no_grad():
                anonymized_embeddings = generator(
                    original_embeddings.to(device),
                    attribute_deltas,
                    lambda_tensor
                )
            
            # Move to CPU for metric computation
            orig_cpu = original_embeddings.cpu()
            anon_cpu = anonymized_embeddings.cpu()
            
            # Compute all pairwise cosine similarities between original and anonymized
            # This simulates an attacker trying to link anonymized speech to original
            similarities = []
            labels = []
            
            # Strategy 1: Original vs its own anonymized version (should be similar)
            same_speaker_sims = torch.cosine_similarity(
                orig_cpu, anon_cpu, dim=1
            ).numpy()
            
            # Strategy 2: Cross-speaker comparisons (should be dissimilar)
            for i in range(min(N, 500)):  # Limit to 500 for speed
                for j in range(i+1, min(N, 500)):
                    # Original i vs Anonymized j
                    sim = torch.cosine_similarity(
                        orig_cpu[i:i+1],
                        anon_cpu[j:j+1],
                        dim=1
                    ).item()
                    
                    similarities.append(sim)
                    labels.append(1 if speaker_ids[i] == speaker_ids[j] else 0)
            
            similarities = np.array(similarities)
            labels = np.array(labels)
            
            # Compute EER
            if len(np.unique(labels)) > 1:  # Need both classes
                eer, threshold = compute_eer(similarities, labels)
                trial_eers.append(eer)
                trial_similarities.append(same_speaker_sims.mean())
        
        # Aggregate results across trials
        results[f'lambda_{lambda_val}'] = {
            'eer': np.mean(trial_eers),
            'eer_std': np.std(trial_eers),
            'mean_similarity': np.mean(trial_similarities),
            'similarity_std': np.std(trial_similarities)
        }
    
    return results


def evaluate_linkability(
    original_embeddings,
    anonymized_embeddings,
    speaker_ids
):
    """
    Linkability metric: Can an attacker link anonymized embeddings to originals?
    
    This is the core privacy metric for voice anonymization.
    
    Args:
        original_embeddings: (N, dim) original embeddings
        anonymized_embeddings: (N, dim) anonymized embeddings
        speaker_ids: (N,) speaker IDs
        
    Returns:
        linkability_score: Fraction of correct links (0 = perfect privacy, 1 = no privacy)
        eer: Equal error rate for linking
    """
    N = len(original_embeddings)
    
    # Compute similarity matrix: original[i] vs anonymized[j]
    sim_matrix = torch.mm(
        torch.nn.functional.normalize(original_embeddings, dim=1),
        torch.nn.functional.normalize(anonymized_embeddings, dim=1).t()
    ).cpu().numpy()
    
    # For each anonymized embedding, find most similar original
    correct_links = 0
    
    for j in range(N):
        predicted_original = np.argmax(sim_matrix[:, j])
        if speaker_ids[predicted_original] == speaker_ids[j]:
            correct_links += 1
    
    linkability_score = correct_links / N
    
    # Compute EER for linking
    similarities = []
    labels = []
    
    for i in range(N):
        for j in range(N):
            similarities.append(sim_matrix[i, j])
            labels.append(1 if speaker_ids[i] == speaker_ids[j] else 0)
    
    eer, _ = compute_eer(np.array(similarities), np.array(labels))
    
    return linkability_score, eer


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
    for _ in range(min(1000, n * n // 4)):
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


def evaluate_attribute_preservation(
    generator,
    rankers,
    original_embeddings,
    target_deltas,
    lambda_anon
):
    """
    Evaluate if attributes change as requested while anonymizing.
    
    Args:
        generator: Generator model
        rankers: MultiAttributeRanker
        original_embeddings: (N, dim) original embeddings
        target_deltas: (N, num_attributes) requested attribute changes
        lambda_anon: (N, 1) anonymization levels
        
    Returns:
        accuracy: Fraction of correct attribute changes
    """
    generator.eval()
    device = next(generator.parameters()).device
    
    with torch.no_grad():
        anonymized = generator(
            original_embeddings.to(device),
            target_deltas.to(device),
            lambda_anon.to(device)
        )
    
    correct = 0
    total = 0
    
    # Check each attribute
    for attr_idx, attr_name in enumerate(rankers.attributes):
        # Get ranker scores
        orig_scores = rankers.score(original_embeddings.to(device), attr_name).cpu()
        anon_scores = rankers.score(anonymized.to(device), attr_name).cpu()
        
        # Check if change direction matches target
        actual_change = anon_scores - orig_scores
        target_change = target_deltas[:, attr_idx].cpu()
        
        # Count correct directions (sign agreement)
        correct += ((actual_change * target_change) > 0).sum().item()
        total += len(actual_change)
    
    return correct / total if total > 0 else 0.0


def evaluate_utility_asr(
    original_audio,
    anonymized_audio,
    asr_model,
    transcripts=None
):
    """
    Evaluate utility: does speech content remain understandable?
    Measures Word Error Rate (WER) on anonymized speech.
    
    Args:
        original_audio: List of original audio waveforms
        anonymized_audio: List of anonymized audio waveforms
        asr_model: Automatic speech recognition model
        transcripts: Ground truth transcripts (optional)
        
    Returns:
        wer_original: WER on original speech
        wer_anonymized: WER on anonymized speech
        wer_increase: Increase in WER due to anonymization
    """
    # This would require actual speech synthesis from embeddings
    # Placeholder for now - in full system, you'd:
    # 1. Convert embeddings back to speech using vocoder
    # 2. Run ASR on both original and anonymized
    # 3. Compute WER
    
    raise NotImplementedError(
        "Utility evaluation requires speech synthesis from embeddings. "
        "Integrate with a vocoder (e.g., HiFi-GAN) to enable this."
    )
