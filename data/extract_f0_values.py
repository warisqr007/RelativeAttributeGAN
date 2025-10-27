#!/usr/bin/env python3
"""
Batch F0 / A_p / A_ap extraction with multiprocessing.

Usage:
  python batch_analyze.py \
    --pattern "data/**/wav/*.wav" \
    --backend world \
    --target_sr 16000 \
    --frame_ms 10 \
    --jobs 8
"""

import os
import sys
import glob
import json
from pathlib import Path
from functools import partial
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# sys.path.append('..')
from f0_extractor import analyze_wav_to_tensors, AnalyzeConfig
import warnings

# Suppress UserWarning and FutureWarning globally
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def _analyze_one(
    wav_path: str,
    backend: str = "world",
    target_sr: int = 16000,
    frame_ms: float = 10.0,
    f0_floor: float = 50.0,
    f0_ceil: float = 1000.0,
    smooth_f0_med: int = 5,
    enforce_voicing: bool = True,
    aperiodic_bands_json: Optional[str] = None,
    aperiodic_weights_json: Optional[str] = None,
) -> Tuple[str, bool, Optional[str]]:
    """
    Returns (wav_path, success, error_message_if_any)
    """
    try:
        wav_path = str(wav_path)
        try:
            wav, sr = sf.read(wav_path, always_2d=False)  # (N,) or (N, C)
        except Exception:
            # try librosa as fallback
            wav, sr = librosa.load(wav_path, sr=None, mono=False)  # (N,) or (C,N)
            if wav.ndim > 1:
                wav = wav.T  # (N,C)
        # parse optional banding/weights
        bands = json.loads(aperiodic_bands_json) if aperiodic_bands_json else None
        weights = json.loads(aperiodic_weights_json) if aperiodic_weights_json else None

        cfg = AnalyzeConfig(
            backend=backend,
            target_sr=target_sr,
            frame_ms=frame_ms,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            smooth_f0_med=smooth_f0_med,
            enforce_voicing=enforce_voicing,
            aperiodic_bands=bands,
            aperiodic_weights=weights,
        )

        # Run analyzer (single item batch)
        f0, Ap, Aap = analyze_wav_to_tensors([wav], [sr], cfg)  # tensors (1,1,T)
        f0_np  = f0.squeeze().cpu().numpy().astype(np.float32)     # (T,)
        Ap_np  = Ap.squeeze().cpu().numpy().astype(np.float32)     # (T,)
        Aap_np = Aap.squeeze().cpu().numpy().astype(np.float32)    # (T,)

        out_arr = np.stack([f0_np, Ap_np, Aap_np], axis=0)         # (3, T)

        # Save path: <wav_dir>/../world_f0/<stem>.f0.npy
        p = Path(wav_path)
        out_dir = p.parent.parent.parent.parent / "world_f0" / p.parent.parent.name / p.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{p.stem}.f0.npy"
        np.save(out_path, out_arr)
        return (wav_path, True, None)
    except Exception as e:
        return (wav_path, False, str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multiprocess F0/A_p/A_ap extraction")
    parser.add_argument("--pattern", type=str, default="data/**/wav/*.wav",
                        help="Glob pattern for WAV files")
    parser.add_argument("--backend", type=str, choices=["world", "crepe_hnr"], default="world",
                        help="Pitch/AP backend")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--frame_ms", type=float, default=20.0, help="Frame hop in ms")
    parser.add_argument("--f0_floor", type=float, default=50.0, help="WORLD/torchcrepe fmin")
    parser.add_argument("--f0_ceil", type=float, default=1000.0, help="WORLD/torchcrepe fmax")
    parser.add_argument("--smooth_f0_med", type=int, default=5, help="Median filter width (frames). 0/1 to disable")
    parser.add_argument("--enforce_voicing", action="store_true", default=True, help="Zero F0 in unvoiced frames")
    parser.add_argument("--no-enforce_voicing", dest="enforce_voicing", action="store_false")
    parser.add_argument("--aperiodic_bands", type=str, default=None,
                        help="JSON list of [lo,hi] pairs in Hz (overrides defaults). Example: '[[0,1000],[1000,2000],[2000,4000],[4000,8000]]'")
    parser.add_argument("--aperiodic_weights", type=str, default=None,
                        help="JSON list of weights same length as bands")
    parser.add_argument("--jobs", type=int, default=max(1, cpu_count() // 2),
                        help="Parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    wav_files = glob.glob(args.pattern, recursive=True)
    if not wav_files:
        print(f"No files matched: {args.pattern}")
        return

    worker = partial(
        _analyze_one,
        backend=args.backend,
        target_sr=args.target_sr,
        frame_ms=args.frame_ms,
        f0_floor=args.f0_floor,
        f0_ceil=args.f0_ceil,
        smooth_f0_med=args.smooth_f0_med,
        enforce_voicing=args.enforce_voicing,
        aperiodic_bands_json=args.aperiodic_bands,
        aperiodic_weights_json=args.aperiodic_weights,
    )

    n_jobs = max(1, int(args.jobs))
    if n_jobs == 1:
        it = map(worker, wav_files)
        results = list(tqdm(it, total=len(wav_files), disable=args.quiet))
    else:
        with Pool(processes=n_jobs) as pool:
            it = pool.imap_unordered(worker, wav_files, chunksize=4)
            results = list(tqdm(it, total=len(wav_files), disable=args.quiet))

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_bad = len(results) - n_ok
    if not args.quiet:
        print(f"Done. OK: {n_ok}  Failed: {n_bad}")

    # optional: write a small report of failures
    failures = [(p, msg) for p, ok, msg in results if not ok]
    if failures:
        report_path = Path("f0_ap_report_failures.json")
        with open(report_path, "w") as f:
            json.dump([{"path": p, "error": msg} for p, msg in failures], f, indent=2)
        if not args.quiet:
            print(f"Wrote failure report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
