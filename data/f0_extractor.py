import math
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional

import numpy as np
import torch

# Optional backends
try:
    import pyworld as pw
    _HAS_WORLD = True
except Exception:
    _HAS_WORLD = False

try:
    import torchcrepe  # optional neural F0
    _HAS_TORCHCREPE = True
except Exception:
    _HAS_TORCHCREPE = False


# --------------------------
# Utilities
# --------------------------
def to_mono_resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """wav: (N,) or (C,N). Returns mono @ target_sr as float64 in [-1,1]."""
    x = wav
    if x.ndim == 2:
        x = x.mean(axis=0)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    return x

def median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    from scipy.signal import medfilt
    if k <= 1:
        return x
    return medfilt(x, kernel_size=k)

def hz_to_voicing_mask(f0_hz: np.ndarray, thr_hz: float = 20.0) -> np.ndarray:
    return (f0_hz > thr_hz).astype(np.float64)

def safe_div(a: np.ndarray, b: np.ndarray, eps=1e-12) -> np.ndarray:
    return a / (b + eps)


# --------------------------
# WORLD-based analysis (recommended)
# --------------------------
@dataclass
class WorldParams:
    fs: int = 16000
    f0_floor: float = 50.0
    f0_ceil: float = 1000.0
    frame_ms: float = 10.0  # hop in ms

def world_analyze(wav: np.ndarray, params: WorldParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      f0_hz: (T,) in Hz
      ap_full: (T, F) aperiodicity spectrum in [0,1]
    """
    assert _HAS_WORLD, "pyworld is not installed."
    f0_hz, t = pw.harvest(
        wav, fs=params.fs,
        f0_floor=params.f0_floor, f0_ceil=params.f0_ceil,
        frame_period=params.frame_ms
    )
    ap_full = pw.d4c(wav, f0_hz, t, fs=params.fs) #, frame_period=params.frame_ms)
    if ap_full.ndim == 3:   # (1,T,F) -> (T,F)
        ap_full = ap_full[0]
    return f0_hz.astype(np.float64), np.clip(ap_full, 0.0, 1.0)


def default_ap_bands(sr: int) -> List[Tuple[float, float]]:
    """
    Choose 5 bands up to Nyquist; for 16k => up to 8k, for 24k => up to 12k.
    """
    nyq = sr / 2.0
    # Start with canonical cuts, then clamp to Nyquist.
    cuts = [0, 1000, 2000, 4000, 6000, 8000, 12000]
    cuts = [c for c in cuts if c <= nyq]
    if cuts[-1] != nyq:
        cuts.append(nyq)
    bands = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]
    return bands

def ap_spectrum_to_scalars(
    ap_full: np.ndarray,
    sr: int,
    bands: Optional[List[Tuple[float,float]]] = None,
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Convert AP spectrum per frame to a scalar A_ap in [0,1].
    If bands is None, picks 4-6 bands up to Nyquist depending on sr.
    """
    T, F = ap_full.shape
    freqs = np.linspace(0.0, sr/2.0, F)
    if bands is None:
        bands = default_ap_bands(sr)
    if weights is None:
        # emphasize HF slightly; length = nbands
        nb = len(bands)
        base = np.linspace(0.9, 1.3, nb)
        weights = base
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    vals = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        v = ap_full[:, m].mean(axis=1) if m.any() else np.zeros(T, dtype=np.float64)
        vals.append(v)
    vals = np.stack(vals, axis=1)  # (T, nbands)
    A_ap = np.clip((vals * w[None, :]).sum(axis=1), 0.0, 1.0)  # (T,)
    return A_ap


# --------------------------
# TorchCrepe + HNR fallback (no pyworld)
# --------------------------
def torchcrepe_f0_anysr(wav: torch.Tensor, sample_rate: int, hop_length: int,
                        fmin: float = 50.0, fmax: float = 1000.0) -> torch.Tensor:
    """
    wav: (1,N) float32 on CPU/CUDA, arbitrary sample_rate.
    """
    assert _HAS_TORCHCREPE, "torchcrepe not installed."
    device = wav.device
    f0 = torchcrepe.predict(
        wav, sample_rate=sample_rate, hop_length=hop_length,
        fmin=fmin, fmax=fmax, model='full', batch_size=1024,
        device=device, return_periodicity=False
    )  # (1, T)
    return f0

def hnr_scalar_from_autocorr(wav: np.ndarray, sr: int, hop: int, win: int,
                             f0_hz: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rough HNR-based A_ap in [0,1]. Uses sr to set lag bounds.
    """
    N = len(wav)
    T = max(1, N // hop)
    A_ap = np.zeros(T, dtype=np.float64)
    for i in range(T):
        s = i * hop
        e = min(N, s + win)
        x = wav[s:e]
        if len(x) < 8:
            A_ap[i] = 1.0
            continue
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')
        r = r[len(x)-1:]
        r = r / (r[0] + 1e-12)
        if f0_hz is not None and f0_hz[i] > 20:
            lag = int(round(sr / f0_hz[i]))
            lo = max(1, int(lag * 0.6))
            hi = min(len(r)-1, int(lag * 1.4))
        else:
            lo = int(sr / 1000.0)  # 1 kHz upper bound
            hi = int(sr / 50.0)    # 50 Hz lower bound
        peak = r[lo:hi].max() if hi > lo else 0.0
        A_p = 1.0 / (1.0 + math.exp(-8.0 * (peak - 0.5)))  # sigmoid around 0.5
        A_ap[i] = float(np.clip(1.0 - A_p, 0.0, 1.0))
    return A_ap


# --------------------------
# Public API
# --------------------------
@dataclass
class AnalyzeConfig:
    backend: Literal["world", "crepe_hnr"] = "world"
    target_sr: int = 16000            # <<< set this to 16000, 24000, etc.
    frame_ms: float = 10.0
    f0_floor: float = 50.0
    f0_ceil: float = 1000.0
    smooth_f0_med: int = 5            # frames; 0/1 to disable
    enforce_voicing: bool = True      # zero f0 in unvoiced
    aperiodic_bands: Optional[List[Tuple[float,float]]] = None  # override if desired
    aperiodic_weights: Optional[List[float]] = None

def analyze_wav_to_tensors(
    wavs: List[np.ndarray],
    srs: List[int],
    cfg: AnalyzeConfig = AnalyzeConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Input:
      wavs: list of waveforms (np.float32/float64), each (N,) or (C,N)
      srs:  list of sample rates
    Output tensors (B,1,T) in float32 at cfg.frame_ms frames for cfg.target_sr:
      f0_hz, A_p, A_ap
    """
    assert len(wavs) == len(srs)
    f0_list, Ap_list, Aap_list = [], [], []

    for wav, sr in zip(wavs, srs):
        x = to_mono_resample(wav, sr, cfg.target_sr)  # float64 @ target_sr
        hop = int(round(cfg.target_sr * (cfg.frame_ms / 1000.0)))

        if cfg.backend == "world":
            assert _HAS_WORLD, "pyworld not available."
            f0_hz, ap_full = world_analyze(
                x, WorldParams(fs=cfg.target_sr, f0_floor=cfg.f0_floor, f0_ceil=cfg.f0_ceil, frame_ms=cfg.frame_ms)
            )
            if cfg.smooth_f0_med and cfg.smooth_f0_med > 1:
                f0_hz = median_filter_1d(f0_hz, cfg.smooth_f0_med)
            vmask = hz_to_voicing_mask(f0_hz)
            if cfg.enforce_voicing:
                f0_hz = f0_hz * vmask
            A_ap = ap_spectrum_to_scalars(
                ap_full, sr=cfg.target_sr, bands=cfg.aperiodic_bands, weights=cfg.aperiodic_weights
            )
            A_ap = np.clip(A_ap, 0.0, 1.0)
            A_p = 1.0 - A_ap

        elif cfg.backend == "crepe_hnr":
            assert _HAS_TORCHCREPE, "torchcrepe not available."
            xt = torch.from_numpy(x).float().unsqueeze(0)  # (1,N)
            f0_t = torchcrepe_f0_anysr(
                xt, sample_rate=cfg.target_sr, hop_length=hop,
                fmin=cfg.f0_floor, fmax=cfg.f0_ceil
            ).cpu().numpy()[0]  # (T,)
            if cfg.smooth_f0_med and cfg.smooth_f0_med > 1:
                f0_t = median_filter_1d(f0_t, cfg.smooth_f0_med)
            vmask = hz_to_voicing_mask(f0_t)
            if cfg.enforce_voicing:
                f0_t = f0_t * vmask
            A_ap = hnr_scalar_from_autocorr(
                x, sr=cfg.target_sr, hop=hop, win=4*hop, f0_hz=f0_t
            )
            A_ap = np.clip(A_ap, 0.0, 1.0)
            A_p = 1.0 - A_ap
            f0_hz = f0_t
        else:
            raise ValueError("Unknown backend")

        f0_list.append(torch.from_numpy(f0_hz).float()[None, None, :])     # (1,1,T)
        Ap_list.append(torch.from_numpy(A_p).float()[None, None, :])       # (1,1,T)
        Aap_list.append(torch.from_numpy(A_ap).float()[None, None, :])     # (1,1,T)

    # Pad to longest T in batch
    def _pad_cat(tensors: List[torch.Tensor]) -> torch.Tensor:
        # tensors: list of (1,1,T_i)
        maxT = max(t.size(-1) for t in tensors)
        out = []
        for t in tensors:
            pad = maxT - t.size(-1)
            if pad > 0:
                t = torch.nn.functional.pad(t, (0, pad))
            out.append(t)
        return torch.cat(out, dim=0)  # (B,1,maxT)

    f0  = _pad_cat(f0_list)
    Ap  = _pad_cat(Ap_list)
    Aap = _pad_cat(Aap_list)
    return f0, Ap, Aap



if __name__ == "__main__":
    # Simple test
    import soundfile as sf

    # Load example wav
    wav, sr = sf.read("/mnt/data1/waris/PSI-TAMU/DarkStreamExt/sample_wav/p225/p225_001_mic1.wav")  # replace with your file
    print(f"Loaded wav: {wav.shape}, sr={sr}")

    cfg = AnalyzeConfig(backend="world", target_sr=16000, frame_ms=20.0)
    f0, Ap, Aap = analyze_wav_to_tensors([wav], [sr], cfg)
    print(f"f0: {f0.shape}, Ap: {Ap.shape}, Aap: {Aap.shape}")
    print(f"f0 (first 10): {f0[0,0,:10]}")
    print(f"Ap (first 10): {Ap[0,0,:10]}")
    print(f"Aap(first 10): {Aap[0,0,:10]}")