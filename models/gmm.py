# models/speaker_embedding_gmm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


@dataclass
class GMMState:
    n_components: int
    embedding_dim: int
    covariance_type: str
    pi: torch.Tensor        # [K]
    mu: torch.Tensor        # [K, D]
    var: torch.Tensor       # [K, D]  (diagonal variances per component)


class SpeakerEmbeddingGMM:
    """
    GMM prior for speaker embedding distribution.
    Fit on real embeddings (preferably training split), export diag parameters (pi, mu, var)
    for use in a Torch prior (e.g., GMMNLLPrior). Also supports scoring and sampling.
    """
    def __init__(
        self,
        n_components: int = 32,
        embedding_dim: int = 192,
        covariance_type: str = "diag",  # 'diag' (recommended), 'full', 'tied', 'spherical'
        max_iter: int = 200,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
        )
        self.embedding_dim = embedding_dim
        self.covariance_type = covariance_type
        self.fitted: bool = False

        # Cached torch parameters (cpu by default)
        self._pi_t: Optional[torch.Tensor] = None   # [K]
        self._mu_t: Optional[torch.Tensor] = None   # [K, D]
        self._var_t: Optional[torch.Tensor] = None  # [K, D]

    # ---------------------------- Fitting ----------------------------

    def fit(self, real_embeddings: torch.Tensor) -> "SpeakerEmbeddingGMM":
        """
        Fit GMM to real speaker embeddings.
        Args:
            real_embeddings: (N, D) torch tensor
        """
        assert real_embeddings.dim() == 2, "real_embeddings must be (N, D)"
        N, D = real_embeddings.shape
        if D != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: got {D}, expected {self.embedding_dim}")

        X = real_embeddings.detach().cpu().numpy().astype(np.float64)
        print(f"Fitting GMM with K={self.gmm.n_components}, cov='{self.covariance_type}' on {N}x{D}...")
        self.gmm.fit(X)
        self.fitted = True

        self._cache_torch_params_from_sklearn()
        avg_ll = self.gmm.score(X)  # average log-likelihood per sample
        print(f"✓ GMM fitted. Average log-likelihood: {avg_ll:.4f}")
        return self

    # ---------------------------- Save / Load ----------------------------

    def save(self, path: str) -> None:
        """
        Save the DIAGONALIZED mixture parameters as a torch file:
            { 'n_components', 'embedding_dim', 'covariance_type', 'pi', 'mu', 'var' }
        """
        assert self.fitted, "Fit the GMM before saving."
        state = {
            "n_components": self.gmm.n_components,
            "embedding_dim": self.embedding_dim,
            "covariance_type": self.covariance_type,
            "pi": self._pi_t.cpu(),
            "mu": self._mu_t.cpu(),
            "var": self._var_t.cpu(),
        }
        torch.save(state, path)
        print(f"Saved GMM (diag form) to {path}")

    def load(self, path: str) -> "SpeakerEmbeddingGMM":
        """
        Load parameters saved by .save(). This sets the class into a 'fitted' state.
        Note: we do NOT reconstruct the sklearn.GaussianMixture internals; we operate
        from the cached torch tensors for scoring/sampling.
        """
        state = torch.load(path, map_location="cpu")
        self.embedding_dim = int(state["embedding_dim"])
        self.covariance_type = str(state["covariance_type"])
        # Keep original K to be consistent (but we will use diag tensors)
        n_components = int(state["n_components"])

        self._pi_t = state["pi"].float().clone()
        self._mu_t = state["mu"].float().clone()
        self._var_t = state["var"].float().clone()

        # Minimal sanity checks
        assert self._pi_t.shape == (n_components,), "pi shape mismatch"
        assert self._mu_t.shape == (n_components, self.embedding_dim), "mu shape mismatch"
        assert self._var_t.shape == (n_components, self.embedding_dim), "var shape mismatch"

        self.fitted = True
        # We won't rebuild sklearn's mixture internals (not needed). Set to None to avoid confusion.
        self.gmm = None
        print(f"Loaded GMM (diag form) from {path} with K={n_components}, D={self.embedding_dim}")
        return self

    # Legacy internal names (if you want to keep private API)
    def _save_gmm_params(self, path: str) -> None:
        self.save(path)

    def _load_gmm_params(self, path: str) -> None:
        self.load(path)

    # ---------------------------- Export to Torch Prior ----------------------------

    def to_prior_state(self) -> GMMState:
        """
        Return a dataclass with (pi, mu, var) suitable for GMMNLLPrior.load_gmm.
        """
        assert self.fitted and (self._pi_t is not None)
        return GMMState(
            n_components=self._pi_t.numel(),
            embedding_dim=self.embedding_dim,
            covariance_type="diag",
            pi=self._pi_t.clone(),
            mu=self._mu_t.clone(),
            var=self._var_t.clone(),
        )

    # ---------------------------- Scoring ----------------------------

    @torch.no_grad()
    def log_probability(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Log p(x) under the *diagonalized* mixture (torch, batched).
        Args:
            embeddings: (B, D) tensor
        Returns:
            log_prob: (B,) tensor
        """
        assert self.fitted, "GMM must be fitted or loaded first."
        x = embeddings  # (B, D)
        assert x.dim() == 2 and x.size(1) == self.embedding_dim

        # Move cached params to the device of x
        pi = self._pi_t.to(x.device)           # [K]
        mu = self._mu_t.to(x.device)           # [K, D]
        var = self._var_t.clamp_min(1e-8).to(x.device)  # [K, D]
        log2pi = math.log(2.0 * math.pi)

        # Compute per-component log-probs: [B, K]
        # -0.5 * [ sum_d ((x - mu)^2 / var + log(2π) + log var) ]
        B, D = x.shape
        x_exp = x[:, None, :]                   # [B, 1, D]
        diff2 = (x_exp - mu)**2                 # [B, K, D]
        log_det = var.log()                     # [K, D]
        mahal = (diff2 / var).sum(dim=-1)       # [B, K]
        log_gauss = -0.5 * (mahal + log_det.sum(dim=-1)[None, :] + D * log2pi)  # [B, K]

        # log-sum-exp over components with priors pi
        log_mix = (pi + 1e-12).log()[None, :] + log_gauss  # [B, K]
        log_prob = torch.logsumexp(log_mix, dim=-1)        # [B]
        return log_prob

    @torch.no_grad()
    def nll(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood per sample (B,).
        """
        return -self.log_probability(embeddings)

    # ---------------------------- Sampling ----------------------------

    @torch.no_grad()
    def sample(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from the *diagonalized* mixture.
        Returns: (n_samples, D) tensor
        """
        assert self.fitted and (self._pi_t is not None)
        pi = self._pi_t
        mu = self._mu_t
        var = self._var_t.clamp_min(1e-8)

        # Choose components
        comp_idx = torch.multinomial(pi / pi.sum(), num_samples=n_samples, replacement=True)  # [n]
        means = mu[comp_idx]                  # [n, D]
        std = var[comp_idx].sqrt()            # [n, D]
        z = torch.randn_like(means)
        samples = means + std * z
        if device is not None:
            samples = samples.to(device)
        return samples

    # ---------------------------- Responsibilities ----------------------------

    @torch.no_grad()
    def get_component_assignment(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          resp: (B, K) responsibilities (posterior p(k|x))
          hard: (B,) hard assignments (argmax over components)
        """
        assert self.fitted
        x = embeddings
        pi = self._pi_t.to(x.device)
        mu = self._mu_t.to(x.device)
        var = self._var_t.clamp_min(1e-8).to(x.device)
        log2pi = math.log(2.0 * math.pi)

        x_exp = x[:, None, :]                  # [B, 1, D]
        diff2 = (x_exp - mu)**2
        mahal = (diff2 / var).sum(dim=-1)      # [B, K]
        log_gauss = -0.5 * (mahal + var.log().sum(-1)[None, :] + x.size(1) * log2pi)  # [B, K]
        log_post = (pi + 1e-12).log()[None, :] + log_gauss
        # normalize
        resp = torch.softmax(log_post, dim=-1)  # [B, K]
        hard = resp.argmax(dim=-1)
        return resp, hard

    # ---------------------------- Internals ----------------------------

    def _cache_torch_params_from_sklearn(self) -> None:
        """
        Convert sklearn parameters into DIAGONAL form tensors:
          pi:  [K]
          mu:  [K, D]
          var: [K, D]  (diagonal-only; if full/tied/spherical, convert accordingly)
        """
        assert self.fitted and self.gmm is not None
        K = self.gmm.n_components
        D = self.embedding_dim

        pi = torch.from_numpy(self.gmm.weights_).float()          # [K]
        mu = torch.from_numpy(self.gmm.means_).float()            # [K, D]

        cov_type = self.gmm.covariance_type
        cov = self.gmm.covariances_  # shape depends on cov_type

        if cov_type == "diag":
            var = torch.from_numpy(cov).float()                   # [K, D]
        elif cov_type == "spherical":
            # cov: [K] variances; expand to [K, D]
            var = torch.from_numpy(cov).float().unsqueeze(1).expand(K, D).contiguous()
        elif cov_type == "tied":
            # cov: [D, D] shared covariance; take diagonal across D and expand to [K, D]
            diag = torch.from_numpy(np.diag(cov)).float()         # [D]
            var = diag.unsqueeze(0).expand(K, D).contiguous()     # [K, D]
        elif cov_type == "full":
            # cov: [K, D, D]; take diagonal
            diag_list = [np.diag(cov[k]) for k in range(K)]
            var = torch.from_numpy(np.stack(diag_list, axis=0)).float()  # [K, D]
        else:
            raise ValueError(f"Unsupported covariance_type: {cov_type}")

        # Safety clamps
        var = var.clamp_min(1e-8)

        self._pi_t = pi
        self._mu_t = mu
        self._var_t = var
        print(f"Cached GMM params to torch tensors (diag form): pi[{pi.shape}], mu[{mu.shape}], var[{var.shape}]")
    
    def _visualize_gmm_components(self, embeddings):
        """Visualize GMM clusters using t-SNE"""
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Get component assignments
        components = self.get_component_assignment(embeddings)[0]
        
        # t-SNE projection
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=components,
            cmap='tab20',
            alpha=0.6,
            s=10
        )
        plt.colorbar(scatter, label='GMM Component')
        plt.title(f'Speaker Embeddings Clustered by GMM ({self.gmm_prior.gmm.n_components} components)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig('gmm_clusters_visualization.png')
        print("✓ Saved GMM visualization to gmm_clusters_visualization.png")



class GMMNLLPrior(nn.Module):
    def __init__(self, emb_dim=704, num_components=32, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.num_components = num_components
        self.emb_dim = emb_dim
        self.register_buffer("pi", torch.ones(num_components) / num_components)
        self.register_buffer("mu", torch.zeros(num_components, emb_dim))
        self.register_buffer("logvar", torch.zeros(num_components, emb_dim))

    @torch.no_grad()
    def load_gmm(self, pi, mu, var):
        self.pi.copy_(pi)
        self.mu.copy_(mu)
        self.logvar.copy_(var.clamp_min(1e-6).log())

    def nll(self, e):
        if not self.enabled:
            return torch.zeros(e.size(0), device=e.device)
        # log-sum-exp over components
        x = e[:, None, :]  # [B, K, D]
        logp = -0.5 * ((x - self.mu)**2 / self.logvar.exp() + self.logvar + math.log(2*math.pi)).sum(dim=-1)  # [B, K]
        logp = torch.log(self.pi + 1e-9) + logp
        return -torch.logsumexp(logp, dim=1)