import os
import sys
sys.path.append('..')
import argparse
import glob
from pathlib import Path

import torch
import librosa
from tqdm import tqdm

from typing import Any

import pytorch_lightning as pl
import torch.nn as nn
from speechbrain.inference import EncoderClassifier
import numpy as np

import warnings

# Suppress UserWarning and FutureWarning globally
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SR = 16000
OUTDIR_NAME = "spkr_embeds"  # output directory for speaker embeddings


class SpeakerModule(pl.LightningModule):
    """Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        embed_type="xvector+ecapa",
        device="cuda",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.speaker_encoders = nn.ModuleList()
        if "xvector" in embed_type:
            _xvec_model = EncoderClassifier.from_hparams(
                source='speechbrain/spkrec-xvect-voxceleb',
                savedir='pretrained_models/speaker/spkrec-xvect-voxceleb',
                run_opts={'device': device}
            )
            self.speaker_encoders.append(_xvec_model)
        if "ecapa" in embed_type:
            _tdnn_model = EncoderClassifier.from_hparams(
                source='speechbrain/spkrec-ecapa-voxceleb',
                savedir='pretrained_models/speaker/spkrec-ecapa-voxceleb',
                run_opts={'device': device}
            )
            self.speaker_encoders.append(_tdnn_model)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        embeddings = []
        for encoder in self.speaker_encoders:
            embeddings.append(encoder.encode_batch(x))
        
        # for i, emb in enumerate(embeddings):
        #     print(f"Embedding {i}: {emb.shape}")
        embeddings = torch.cat(embeddings, dim=-1).squeeze(1)
        return embeddings


def get_model(device='cuda'):
    model = SpeakerModule(device=device)
    model.eval()
    return model


@torch.inference_mode()
def get_speaker_embeds(model, audio_fpath, device='cuda'):
    audio, _ = librosa.load(audio_fpath, sr=SR, mono=True)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device, non_blocking=True)

    embeds = model(audio_tensor)
    return embeds.squeeze().cpu().numpy()


def save_embeds_for_file(model, wav_path: str, device: str):
    wav_path = Path(wav_path)

    out_dir = wav_path.parent.parent.parent.parent / OUTDIR_NAME / wav_path.parent.parent.name / wav_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{wav_path.stem}.npy"

    if out_path.exists():
        return  # skip if already done

    embed_vec = get_speaker_embeds(model, str(wav_path), device=device)
    np.save(out_path, embed_vec)


def shard_by_rank(items, rank, world_size):
    return items[rank::world_size]


def run_rank(pattern: str, rank: int, world_size: int, device: str):
    wav_files = sorted(glob.glob(pattern, recursive=True))
    if not wav_files:
        print(f"[rank {rank}] No files matched pattern: {pattern}")
        return

    shard = shard_by_rank(wav_files, rank, world_size)
    model = get_model(device=device)

    pbar = tqdm(shard, desc=f"[rank {rank}] encoding", unit="file")
    for fp in pbar:
        try:
            save_embeds_for_file(model, fp, device)
        except Exception as e:
            pbar.write(f"[rank {rank}] ERROR {fp}: {e}")


def run_single_gpu_with_multiprocessing(pattern: str, jobs: int, device: str):
    """
    Optional: multiprocessing on a single GPU. Each process loads the model
    on the SAME GPU. This may give little/no speedup and can increase VRAM use.
    Prefer batching or multi-GPU instead.
    """
    from multiprocessing import get_context
    wav_files = sorted(glob.glob(pattern, recursive=True))
    if not wav_files:
        print("No files matched pattern:", pattern)
        return

    shards = [wav_files[i::jobs] for i in range(jobs)]

    def _worker(shard):
        # Lazy import in subprocess
        mdl = get_model(device=device)
        for fp in shard:
            try:
                save_embeds_for_file(mdl, fp, device)
            except Exception as e:
                print(f"[mp] ERROR {fp}: {e}")

    ctx = get_context("spawn")
    procs = []
    for i in range(jobs):
        p = ctx.Process(target=_worker, args=(shards[i],))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def main():
    parser = argparse.ArgumentParser(description="Multiprocess/multi-GPU token extraction")
    parser.add_argument("--pattern", type=str, default="data/**/wav/*.wav",
                        help="Glob pattern for WAV files (recursive OK)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="If WORLD_SIZE=1, you can use N processes on a single GPU (not recommended).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device string, e.g., cuda:0 or cpu. Defaults: cuda:<LOCAL_RANK> if available, else cpu.")
    args = parser.parse_args()

    # Multi-GPU via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Device resolution
    if args.device is not None:
        device = args.device
    else:
        if torch.cuda.is_available():
            # Use local rank if launched with torchrun; else default to cuda:0
            device = f"cuda:{local_rank}" if world_size > 1 else "cuda:0"
        else:
            device = "cpu"

    if world_size > 1:
        # torchrun --nproc_per_node=NGPUS this_script.py --pattern ...
        run_rank(args.pattern, local_rank, world_size, device)
    else:
        # Single-GPU / CPU path
        if args.jobs > 1 and "cuda" in device:
            # Optional (see note below)
            run_single_gpu_with_multiprocessing(args.pattern, args.jobs, device)
        else:
            # Just run in a single process
            run_rank(args.pattern, rank=0, world_size=1, device=device)


if __name__ == "__main__":
    main()


'''
Example usage (multi-GPU, 4 GPUs):
torchrun --nproc_per_node=4 dump_tokens.py --pattern="/mnt/data2/waris/Datasets/Multilingual_dataset/data/L2EN/**/wav/*.wav"
'''