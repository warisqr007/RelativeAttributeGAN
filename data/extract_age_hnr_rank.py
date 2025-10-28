#!/usr/bin/env python3
"""
UPDATED: Extract speaker attributes using audeering's pretrained age model.

This version integrates the state-of-the-art age predictor with MAE ~7-10 years.

Usage:
    python extract_attributes_with_audeering.py \
        --audio_pattern "data/**/wav/*.wav" \
        --f0_dir "data/world_f0" \
        --metadata_csv "data/metadata.csv" \
        --use_audeering_age \
        --jobs 8
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

# Import the audeering age predictor
from age_predictor import AudeeringAgePredictor


# ============================================================================
# HNR Computation (same as before)
# ============================================================================

def compute_hnr_praat_style(
    audio: np.ndarray,
    sr: int,
    periods_per_window: float = 4.5,
    silence_threshold: float = 0.1,
    min_pitch: float = 75.0
) -> float:
    """
    Compute HNR using PRAAT-style autocorrelation.
    """
    # Window length based on periods per window
    window_length = int(periods_per_window * sr / min_pitch)
    hop_length = window_length // 4  # 75% overlap
    
    # Normalize
    audio = audio.astype(np.float64)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    silence_amp = silence_threshold
    hnr_values = []
    
    # Frame-by-frame processing
    for i in range(0, len(audio) - window_length, hop_length):
        frame = audio[i:i + window_length]
        
        # Skip silent frames
        if np.max(np.abs(frame)) < silence_amp:
            continue
        
        # Compute normalized autocorrelation
        autocorr = np.correlate(frame, frame, mode='same')
        center = len(autocorr) // 2
        autocorr = autocorr[center:]
        
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            continue
        
        # Find first peak after lag corresponding to min_pitch
        min_lag = int(sr / 1000)  # Avoid very small lags
        max_lag = int(sr / min_pitch)
        
        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
        
        if min_lag < max_lag:
            peak = np.max(autocorr[min_lag:max_lag])
            
            if 0 < peak < 1:
                hnr = 10 * np.log10(peak / (1 - peak))
                hnr_values.append(hnr)
    
    if len(hnr_values) > 0:
        return float(np.mean(hnr_values))
    else:
        return np.nan


# ============================================================================
# F0 Statistics
# ============================================================================

def compute_mean_f0(f0_array: np.ndarray) -> float:
    """
    Compute mean F0, ignoring zero/unvoiced values.
    """
    # If multi-dimensional, extract first row (F0)
    if f0_array.ndim > 1:
        f0 = f0_array[0, :]
    else:
        f0 = f0_array
    
    # Filter out zeros and NaNs (unvoiced frames)
    voiced_f0 = f0[(f0 > 0) & ~np.isnan(f0)]
    
    if len(voiced_f0) > 0:
        return float(np.mean(voiced_f0))
    else:
        return np.nan


# ============================================================================
# Worker Function for Multiprocessing
# ============================================================================

def process_one_file(
    wav_path: str,
    f0_dir: str,
    age_predictor: Optional[AudeeringAgePredictor],
    compute_hnr_flag: bool = True,
    compute_f0_flag: bool = True,
    compute_age_flag: bool = False
) -> Tuple[str, dict]:
    """
    Process a single audio file to extract all attributes.
    """
    result = {
        'file_path': wav_path,
        'hnr': np.nan,
        'mean_f0': np.nan,
        'predicted_age': np.nan,
        'predicted_gender': None,
        'gender_confidence': np.nan
    }
    
    try:
        # Load audio once for HNR and age prediction
        audio = None
        sr = None
        
        if compute_hnr_flag or compute_age_flag:
            # audio, sr = sf.read(wav_path, always_2d=False)
            audio, sr = librosa.load(wav_path, sr=None, mono=False)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        
        # 1. Extract HNR from audio
        if compute_hnr_flag and audio is not None:
            hnr = compute_hnr_praat_style(audio, sr)
            result['hnr'] = hnr
        
        # 2. Read F0 from .npy file and compute mean
        if compute_f0_flag:
            wav_path_obj = Path(wav_path)
            f0_filename = wav_path_obj.stem + '.f0.npy'
            f0_path = Path(f0_dir) / f0_filename
            
            if f0_path.exists():
                f0_data = np.load(f0_path)  # Shape: (3, T) with [F0, Ap, Aap]
                mean_f0 = compute_mean_f0(f0_data)
                result['mean_f0'] = mean_f0
        
        # 3. Predict age using audeering model
        if compute_age_flag and age_predictor is not None and audio is not None:
            age, gender, confidence = age_predictor.predict_from_audio(audio, sr)
            result['predicted_age'] = age
            result['predicted_gender'] = gender
            result['gender_confidence'] = confidence
        
        return (wav_path, result)
    
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return (wav_path, result)


# ============================================================================
# Main Processing Function
# ============================================================================
def shard_by_rank(items, rank, world_size):
    return items[rank::world_size]


def extract_attributes_rank(
    audio_pattern: str,
    f0_dir: str,
    metadata_csv: str,
    use_audeering_age: bool = False,
    audeering_model: str = 'audeering/wav2vec2-large-robust-24-ft-age-gender',
    jobs: int = 8,
    compute_hnr: bool = True,
    compute_f0: bool = True,
    rank=0,
    world_size=1, 
    device="cuda:0"
):
    """
    Extract speaker attributes and update metadata CSV.
    """
    print("\n" + "="*70)
    print("Speaker Attribute Extraction with Audeering Age Model")
    print("="*70)
    
    # Find all audio files
    audio_files = sorted(glob.glob(audio_pattern, recursive=True))
    print(f"\nFound {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print(f"No files matched pattern: {audio_pattern}")
        return
    
    # Load existing metadata if it exists
    # if os.path.exists(metadata_csv):
    #     print(f"Loading existing metadata from {metadata_csv}")
    #     metadata = pd.read_csv(metadata_csv)
    # else:
    print(f"Creating new metadata file")
    metadata = pd.DataFrame()
    
    # Initialize audeering age predictor
    # age_predictor = None
    # if use_audeering_age:
    #     print(f"\nInitializing audeering age model: {audeering_model}")
    #     age_predictor = AudeeringAgePredictor(model_name=audeering_model)
    
    # Process files
    print(f"\nProcessing files with {jobs} workers...")
    
    # Special handling: age predictor uses GPU, so we process sequentially if using it
    if use_audeering_age:
        print("Note: Processing sequentially due to GPU model (age predictor)")
        results = []
        # for audio_file in tqdm(audio_files):
        #     result = process_one_file(
        #         audio_file,
        #         f0_dir=f0_dir,
        #         age_predictor=age_predictor,
        #         compute_hnr_flag=compute_hnr,
        #         compute_f0_flag=compute_f0,
        #         compute_age_flag=use_audeering_age
        #     )
        #     results.append(result)


        shard = shard_by_rank(audio_files, rank, world_size)
        # model = get_model(device=device)
        print(f"\nInitializing audeering age model: {audeering_model} on {device}")
        model = AudeeringAgePredictor(model_name=audeering_model, device=device)

        pbar = tqdm(shard, desc=f"[rank {rank}] encoding", unit="file")
        for fp in pbar:
            try:
                # save_codes_for_file(model, fp, device)
                result = process_one_file(
                    fp,
                    f0_dir=f0_dir,
                    age_predictor=model,
                    compute_hnr_flag=compute_hnr,
                    compute_f0_flag=compute_f0,
                    compute_age_flag=use_audeering_age
                )
                results.append(result)
            except Exception as e:
                pbar.write(f"[rank {rank}] ERROR {fp}: {e}")
    else:
        # Multiprocessing for HNR and F0 only
        worker = partial(
            process_one_file,
            f0_dir=f0_dir,
            age_predictor=None,
            compute_hnr_flag=compute_hnr,
            compute_f0_flag=compute_f0,
            compute_age_flag=False
        )
        
        if jobs == 1:
            results = [worker(f) for f in tqdm(audio_files)]
        else:
            with Pool(processes=jobs) as pool:
                results = list(tqdm(
                    pool.imap_unordered(worker, audio_files),
                    total=len(audio_files)
                ))
    
    # Create results dataframe
    results_df = pd.DataFrame([r[1] for r in results])
    
    # Add speaker_id from filename
    results_df['speaker_id'] = results_df['file_path'].apply(
        lambda x: Path(x).parent.parent.name
    )
    
    # Merge with existing metadata
    if 'file_path' in metadata.columns:
        # Merge on file_path
        metadata = metadata.merge(
            results_df,
            on='file_path',
            how='outer',
            suffixes=('_old', '')
        )
        
        # Update values (prefer new over old)
        for col in ['hnr', 'mean_f0', 'predicted_age', 'predicted_gender', 'gender_confidence']:
            if f'{col}_old' in metadata.columns:
                metadata[col] = metadata[col].fillna(metadata[f'{col}_old'])
                metadata.drop(columns=[f'{col}_old'], inplace=True)
    else:
        # No existing metadata, use results directly
        metadata = results_df
    
    # Ensure required columns exist
    required_cols = ['speaker_id', 'file_path']
    for col in required_cols:
        if col not in metadata.columns:
            if col == 'speaker_id':
                metadata['speaker_id'] = metadata['file_path'].apply(
                    lambda x: Path(x).stem if pd.notna(x) else None
                )
    
    if world_size > 1:
        print(f"[rank {rank}] Finished processing {len(shard)} files.")
        # Save updated metadata
        ranked_metadata_csv = metadata_csv.replace('.csv', f'_rank{rank}.csv')
        metadata.to_csv(ranked_metadata_csv, index=False)
        print(f"[rank {rank}] ✓ Saved metadata to {ranked_metadata_csv}")
    else:
        # Save updated metadata
        metadata.to_csv(metadata_csv, index=False)
        print(f"\n✓ Saved metadata to {metadata_csv}")
    
    # Print statistics
    print("\n" + "-"*70)
    print("Statistics:")
    print("-"*70)
    
    if compute_hnr and 'hnr' in metadata.columns:
        hnr_valid = metadata['hnr'].dropna()
        if world_size > 1:
            print(f"[rank {rank}] HNR: {len(hnr_valid)}/{len(metadata)} valid")
        else:
            print(f"HNR: {len(hnr_valid)}/{len(metadata)} valid")
        if len(hnr_valid) > 0:
            print(f"  Mean: {hnr_valid.mean():.2f} dB")
            print(f"  Std:  {hnr_valid.std():.2f} dB")
            print(f"  Range: [{hnr_valid.min():.2f}, {hnr_valid.max():.2f}] dB")
    
    if compute_f0 and 'mean_f0' in metadata.columns:
        f0_valid = metadata['mean_f0'].dropna()
        if world_size > 1:
            print(f"[rank {rank}] Mean F0: {len(f0_valid)}/{len(metadata)} valid")
        else:
            print(f"\nMean F0: {len(f0_valid)}/{len(metadata)} valid")
        if len(f0_valid) > 0:
            print(f"  Mean: {f0_valid.mean():.2f} Hz")
            print(f"  Std:  {f0_valid.std():.2f} Hz")
            print(f"  Range: [{f0_valid.min():.2f}, {f0_valid.max():.2f}] Hz")
    
    if use_audeering_age and 'predicted_age' in metadata.columns:
        age_valid = metadata['predicted_age'].dropna()
        if world_size > 1:
            print(f"[rank {rank}] Predicted Age: {len(age_valid)}/{len(metadata)} valid")
        else:
            print(f"\nPredicted Age: {len(age_valid)}/{len(metadata)} valid")
        if len(age_valid) > 0:
            print(f"  Mean: {age_valid.mean():.2f} years")
            print(f"  Std:  {age_valid.std():.2f} years")
            print(f"  Range: [{age_valid.min():.2f}, {age_valid.max():.2f}] years")
        
        # Gender distribution
        if 'predicted_gender' in metadata.columns:
            gender_counts = metadata['predicted_gender'].value_counts()
            print(f"\nGender Distribution:")
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count} ({count/len(metadata)*100:.1f}%)")
    
    print("\n✓ Attribute extraction complete!")
    print(f"\nYour metadata.csv now has the following columns:")
    print(metadata.columns.tolist())


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker attributes with audeering's pretrained age model"
    )
    
    parser.add_argument(
        '--audio_pattern',
        type=str,
        required=True,
        help='Glob pattern for audio files (e.g., "data/**/wav/*.wav")'
    )
    
    parser.add_argument(
        '--f0_dir',
        type=str,
        required=True,
        help='Directory containing F0 .npy files from batch_analyze.py'
    )
    
    parser.add_argument(
        '--metadata_csv',
        type=str,
        required=True,
        help='Path to metadata CSV file (will be created if not exists)'
    )
    
    parser.add_argument(
        '--use_audeering_age',
        action='store_true',
        help='Use audeering pretrained model for age prediction (recommended!)'
    )
    
    parser.add_argument(
        '--audeering_model',
        type=str,
        default='audeering/wav2vec2-large-robust-24-ft-age-gender',
        choices=[
            'audeering/wav2vec2-large-robust-24-ft-age-gender',
            'audeering/wav2vec2-large-robust-6-ft-age-gender'
        ],
        help='Which audeering model to use (24 layers = better accuracy, 6 layers = faster)'
    )
    
    parser.add_argument(
        '--jobs',
        type=int,
        default=max(1, cpu_count() // 2),
        help='Number of parallel workers (only for HNR/F0, age prediction is sequential)'
    )
    
    parser.add_argument(
        '--compute_hnr',
        action='store_true',
        default=True,
        help='Compute HNR (Harmonic-to-Noise Ratio)'
    )
    
    parser.add_argument(
        '--compute_f0',
        action='store_true',
        default=False,
        help='Compute mean F0 from .npy files'
    )
    
    args = parser.parse_args()
    
    # Install check
    if args.use_audeering_age:
        try:
            from transformers import Wav2Vec2Processor
            print("✓ transformers library found")
        except ImportError:
            print("ERROR: transformers library not found!")
            print("Install with: pip install transformers")
            sys.exit(1)
    

    # Multi-GPU via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        # Use local rank if launched with torchrun; else default to cuda:0
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda:0"
    else:
        device = "cpu"

    extract_attributes_rank(
        audio_pattern=args.audio_pattern,
        f0_dir=args.f0_dir,
        metadata_csv=args.metadata_csv,
        use_audeering_age=args.use_audeering_age,
        audeering_model=args.audeering_model,
        jobs=args.jobs,
        compute_hnr=args.compute_hnr,
        compute_f0=args.compute_f0,
        rank=local_rank,
        world_size=world_size, 
        device=device
    )


if __name__ == '__main__':
    main()
