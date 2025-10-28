import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm



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
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare final metadata CSV with extracted attributes"
    )
    
    parser.add_argument(
        '--f0_dirname',
        type=str,
        required=True,
        help='Directory containing F0 .npy files from batch_analyze.py'
    )

    parser.add_argument(
        '--audio_dirname',
        type=str,
        required=True,
        help='Directory containing audio files'
    )

    parser.add_argument(
        '--spkr_embed_dirname',
        type=str,
        required=True,
        help='Directory containing speaker embedding files'
    )
    
    parser.add_argument(
        '--age_metadata_csv',
        type=str,
        required=True,
        help='Path to metadata CSV file'
    )

    parser.add_argument(
        '--gender_metadata_csv',
        type=str,
        required=True,
        help='Path to metadata CSV file with speaker genders'
    )

    parser.add_argument(
        '--out_metadata_csv',
        type=str,
        required=True,
        help='Path to final metadata CSV file (will be created)'
    )
    
    args = parser.parse_args()
    
    # Load
    age_metadata    = pd.read_csv(args.age_metadata_csv)
    gender_metadata = pd.read_csv(args.gender_metadata_csv, sep="\t")  # or delimiter="\t"

    # --- Normalize join keys ---
    age_metadata['speaker_id'] = age_metadata['speaker_id'].astype(str).str.strip()
    gender_metadata['VoxCeleb1 ID'] = gender_metadata['VoxCeleb1 ID'].astype(str).str.strip()

    # --- Ensure speaker-level uniqueness (safety) ---
    # If gender_metadata might have multiple rows per speaker, keep first (or choose a rule)
    gender_metadata = (
        gender_metadata
        .drop_duplicates(subset=['VoxCeleb1 ID'])
        .rename(columns={'VoxCeleb1 ID': 'speaker_id'})
    )

    # --- Merge: utterance-level (left) × speaker-level (many-to-one) ---
    # validate='many_to_one' will raise if gender_metadata still has duplicates per speaker_id
    utterance_level = age_metadata.merge(
        gender_metadata,
        on='speaker_id',
        how='left',
        validate='many_to_one'
    )

    # --- (Optional) quick diagnostics ---
    missing_mask = ~utterance_level['speaker_id'].isin(gender_metadata['speaker_id'])
    n_missing = int(missing_mask.sum())
    if n_missing:
        print(f"[warn] {n_missing} utterances have speaker_id not found in gender_metadata.")


    # --- Extract mean F0 for each utterance ---
    wav_fpaths = utterance_level['file_path'].values

    # Prepare F0 file paths -> replace /{audio_dirname}/ with /{f0_dirname}/ and .wav with .f0.npy
    # ../wav/../../file.wav  ->  ../f0/../../file.f0.npy
    f0_fpaths = [
        fpath.replace(f"/{args.audio_dirname}/", f"/{args.f0_dirname}/").replace('.wav', '.f0.npy')
        for fpath in wav_fpaths
    ]

    embedding_fpaths = [
        fpath.replace(f"/{args.audio_dirname}/", f"/{args.spkr_embed_dirname}/").replace('.wav', '.npy')
        for fpath in wav_fpaths
    ]

    mean_f0s = []

    for f0_path in tqdm(f0_fpaths, desc="Computing mean F0s"):
        if os.path.isfile(f0_path):
            f0_array = np.load(f0_path)
            mean_f0 = compute_mean_f0(f0_array)
        else:
            mean_f0 = np.nan  # or some sentinel value
        mean_f0s.append(mean_f0)

    # --- Create final DataFrame ---
    final_dataframe = pd.DataFrame({
        'speaker_id': utterance_level['speaker_id'],
        'embedding_path': embedding_fpaths,
        'age': utterance_level['predicted_age'],
        'pitch': mean_f0s,
        'gender': utterance_level['Gender'].map({'m': 0.0, 'f': 1.0}),  # map to numeric
        'voice_quality': utterance_level['hnr'],
        'set': utterance_level['Set'],
    })

    # split into train/test based on 'Set' column
    final_dataframe = final_dataframe[final_dataframe['set'].isin(['dev', 'test'])].reset_index(drop=True)
    train_dataframe = final_dataframe[final_dataframe['set'] == 'dev'].reset_index(drop=True)
    test_dataframe  = final_dataframe[final_dataframe['set'] == 'test'].reset_index(drop=True)

    # --- Save final metadata CSV ---
    train_dataframe.to_csv(args.out_metadata_csv.replace('.csv', '_train.csv'), index=False)
    test_dataframe.to_csv(args.out_metadata_csv.replace('.csv', '_test.csv'), index=False)
    print(f"Saved final metadata CSVs: ")
    print(f"  Train: {args.out_metadata_csv.replace('.csv', '_train.csv')}")
    print(f"  Len Train: {len(train_dataframe)}")
    print(f"  Test:  {args.out_metadata_csv.replace('.csv', '_test.csv')}")
    print(f"  Len Test: {len(test_dataframe)}")


if __name__ == '__main__':
    main()