# scripts/compute_directions.py
import argparse
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List

@torch.no_grad()
def pca_components(X: torch.Tensor, k: int = None):
    # Mean-center
    Xc = X - X.mean(0, keepdim=True)
    # SVD: Xc = U S Vh, PC rows in Vh
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    if k is not None and k > 0:
        Vh = Vh[:k]
        S = S[:k]
    return Vh, S, Xc  # Vh: [K,D], S: [K], Xc: [N,D]

@torch.no_grad()
def compute_composite_direction(
    Xc: torch.Tensor,          # [N,D] mean-centered embeddings
    scores: torch.Tensor,      # [N,] attribute scores
    Vh: torch.Tensor,          # [K,D] PCs (rows)
    S: torch.Tensor,           # [K] singular values
    mode: str = "corr_var"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a composite direction from PCs, weighting by |corr| * variance.
    Returns:
      v:   [D] unit vector
      corr:[K] corr per PC
      var: [K] variance per PC (eigenvalues)
    """
    Z = Xc @ Vh.t()                     # [N,K] PC scores
    Zc = Z - Z.mean(0, keepdim=True)
    sc = scores - scores.mean()
    denom = (Zc.pow(2).sum(0).sqrt() * sc.pow(2).sum().sqrt() + 1e-8)
    corr = (Zc.mul(sc[:, None]).sum(0)) / denom          # [K]
    var = (S**2) / (Xc.shape[0]-1)                       # [K] eigenvalues

    if mode == "corr_var":
        w = corr.abs() * var
    else:
        raise ValueError(f"Unknown mode {mode}")

    signed_w = torch.sign(corr) * w                      # keep sign
    v = (signed_w[:, None] * Vh).sum(0)                  # [D]
    v = F.normalize(v, dim=0)
    return v, corr, var

@torch.no_grad()
def orthogonalize_set(v_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Orthogonalize multiple directions jointly.
    Uses QR on stacked column matrix, preserves order, and aligns sign with originals.
    """
    # Stack as D x M (columns are directions)
    V = torch.stack(v_list, dim=1)        # [D, M]
    # QR gives V = Q R, with Q (D,M) column-orthonormal
    Q, R = torch.linalg.qr(V, mode="reduced")  # Q: [D,M]
    ortho = []
    for j in range(Q.shape[1]):
        qj = Q[:, j]
        # sign alignment to original direction (maximize dot >= 0)
        sign = 1.0 if torch.dot(qj, v_list[j]) >= 0 else -1.0
        ortho.append(F.normalize(sign * qj, dim=0))
    return ortho


def main(args):
    # Load embeddings & scores
    embs = torch.load(args.embs)                    # [N, D]
    age_scores  = torch.load(args.age_scores).view(-1)    # [N]
    gen_scores  = torch.load(args.gen_scores).view(-1)    # [N]
    pitch_scores = torch.load(args.pitch_scores).view(-1) # [N]
    vq_scores    = torch.load(args.vq_scores).view(-1)    # [N]

    assert embs.shape[0] == age_scores.numel() == gen_scores.numel() == pitch_scores.numel() == vq_scores.numel(), \
        "All score tensors must align with embeddings (same N)."

    # PCA on mean-centered embeddings
    Vh, S, Xc = pca_components(embs, k=args.topk if args.topk > 0 else None)

    # Raw composite directions for each attribute (before orthogonalization)
    v_age,   corr_a, var = compute_composite_direction(Xc, age_scores,   Vh, S)
    v_gen,   corr_g, _   = compute_composite_direction(Xc, gen_scores,   Vh, S)
    v_pitch, corr_p, _   = compute_composite_direction(Xc, pitch_scores, Vh, S)
    v_vq,    corr_vq, _  = compute_composite_direction(Xc, vq_scores,    Vh, S)

    # Joint orthogonalization (preserves order: age, gender, pitch, voice_quality)
    v_age_o, v_gen_o, v_pitch_o, v_vq_o = orthogonalize_set([v_age, v_gen, v_pitch, v_vq])

    # Save
    out = {
        "topk": args.topk,
        "v_age": v_age_o.cpu(),
        "v_gender": v_gen_o.cpu(),
        "v_pitch": v_pitch_o.cpu(),
        "v_voice_quality": v_vq_o.cpu(),
        # also keep the pre-ortho vectors if you want to inspect drift
        "v_age_raw": v_age.cpu(),
        "v_gender_raw": v_gen.cpu(),
        "v_pitch_raw": v_pitch.cpu(),
        "v_voice_quality_raw": v_vq.cpu(),
        # corr vectors (per-PC) for diagnostics
        "corr_age": corr_a.cpu(),
        "corr_gender": corr_g.cpu(),
        "corr_pitch": corr_p.cpu(),
        "corr_voice_quality": corr_vq.cpu(),
        # per-PC variance used
        "pc_var": var.cpu(),         # same for all attributes (from PCA)
        # PC basis for reproducibility (optional)
        "Vh": Vh.cpu(),
        "S": S.cpu(),
    }
    torch.save(out, args.out)

    # Logs
    print(
        "Saved directions to", args.out,
        f"||v_age||={v_age_o.norm():.3f}",
        f"||v_gender||={v_gen_o.norm():.3f}",
        f"||v_pitch||={v_pitch_o.norm():.3f}",
        f"||v_voice_quality||={v_vq_o.norm():.3f}",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embs", required=True, help="path to train_embs.pt [N,D]")
    parser.add_argument("--age_scores", required=True, help="path to age_scores.pt [N]")
    parser.add_argument("--gen_scores", required=True, help="path to gender_scores.pt [N]")
    parser.add_argument("--pitch_scores", required=True, help="path to pitch_scores.pt [N]")
    parser.add_argument("--vq_scores", required=True, help="path to voice_quality_scores.pt [N]")
    parser.add_argument("--topk", type=int, default=256, help="use top-K PCs (0=all)")
    parser.add_argument("--out", default="composite_dirs_4attr.pt")
    args = parser.parse_args()
    main(args)
