import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .components import ModalityInputBlock, AttentionBlock, MLP 

class EncoderXYU(nn.Module):
    """
    Encoder mapping (x, y, sample_idx) -> q(u | x, y, sample_idx).

    This is the first-level encoder in a hierarchical VAE, producing latent variable u.
    It processes gene and protein modalities, combines them, incorporates a sample
    embedding, and then uses a shared MLP head to produce parameters for the
    Normal distribution over u.

    Parameters
    ----------
    n_genes
        Number of gene features.
    n_proteins
        Number of protein features.
    n_latent
        Dimensionality of the output latent variable u.
    n_hidden
        Size of hidden layers in ModalityInputBlocks, sample embedding, and the shared MLP head.
    n_sample
        Number of unique sample categories for conditioning.
    n_layers
        Number of residual blocks in the shared MLP head (using `totalmrvi.components.MLP`).
        Must be at least 1.
    """
    def __init__(self, n_genes: int, n_proteins: int, n_latent: int, n_hidden: int, n_sample: int, n_layers: int = 1):
        super().__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_sample = n_sample

        self.x_proj = ModalityInputBlock(n_genes, n_hidden, n_sample)
        self.y_proj = ModalityInputBlock(n_proteins, n_hidden, n_sample)
        self.shared_proj = ModalityInputBlock(2 * n_hidden, n_hidden, n_sample)
        self.sample_embedding = nn.Embedding(n_sample, n_hidden)

        self.mlp = MLP(in_dim=n_hidden, hidden_dim=n_hidden, out_dim=n_hidden, n_layers=n_layers)
        self.loc_readout = nn.Linear(n_hidden, n_latent)
        self.scale_readout = nn.Linear(n_hidden, n_latent)

        self.softplus = nn.Softplus()
        self.scale_eps = 1e-5

    def forward(self, x: torch.Tensor, y: torch.Tensor, sample_idx: torch.Tensor) -> Normal:
        x_log = torch.log1p(x)
        y_log = torch.log1p(y)

        h_x = self.x_proj(x_log, sample_idx)
        h_y = self.y_proj(y_log, sample_idx)

        h = torch.cat([h_x, h_y], dim=-1)
        h = self.shared_proj(h, sample_idx)
        h = h + self.sample_embedding(sample_idx.squeeze(-1) if sample_idx.ndim == 2 else sample_idx) # Ensure 1D for embedding
        
        h = self.mlp(h)
        loc = self.loc_readout(h)
        scale = self.softplus(self.scale_readout(h)) + self.scale_eps
        return Normal(loc, scale)

class EncoderUZ(nn.Module):
    """
    Encoder mapping (u, sample_idx) -> q(z | u, sample_idx).

    This is the second-level encoder, producing z = z_base + eps.
    `eps` is derived from `u` attending to a sample-specific context.

    Parameters
    ----------
    n_latent
        Dimensionality of the output latent variable z (and its components).
    n_sample
        Number of unique sample categories for conditioning.
    n_latent_u
        Dimensionality of the input latent variable u. If None, defaults to `n_latent`.
    n_latent_sample
        Embedding dimension for the sample index.
    n_heads
        Number of attention heads in the internal AttentionBlock.
    dropout_rate
        Dropout rate for the internal AttentionBlock.
    stop_gradients
        If True, detach u and sample_context before processing for eps.
    stop_gradients_mlp
        If True, freeze MLPs within the internal AttentionBlock.
    use_map
        If True, eps is a tensor. If False, eps is (mean, scale_chunk) tuple.
    n_hidden
        Hidden dimension for MLPs within the internal AttentionBlock.
    n_layers
        Number of layers for MLPs within the internal AttentionBlock.
    """
    def __init__(
        self,
        n_latent: int,
        n_sample: int,
        n_latent_u: int | None = None,
        n_latent_sample: int = 16,
        n_channels: int = 4,
        n_heads: int = 2,
        dropout_rate: float = 0.0,
        stop_gradients: bool = False,
        stop_gradients_mlp: bool = False,
        use_map: bool = True,
        n_hidden: int = 32,
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.use_map = use_map
        self.n_latent_u = n_latent_u if n_latent_u is not None else n_latent
        self.stop_gradients = stop_gradients

        self.u_layernorm = nn.LayerNorm(self.n_latent_u)
        self.sample_layernorm = nn.LayerNorm(n_latent_sample)
        self.sample_embed = nn.Embedding(n_sample, n_latent_sample)

        n_outs = 2 if not use_map else 1
        self.attention = AttentionBlock(
            query_dim=self.n_latent_u,
            kv_dim=n_latent_sample,
            out_dim=n_outs * n_latent,
            outerprod_dim=n_latent_sample,
            n_channels=n_channels,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            n_hidden=n_hidden,
            n_layers=n_layers,
            stop_gradients_mlp=stop_gradients_mlp,
        )
        if self.n_latent_u != self.n_latent:
            self.z_base_proj = nn.Linear(self.n_latent_u, n_latent)
        else:
            self.z_base_proj = nn.Identity()

    def forward(
        self,
        u: torch.Tensor,
        sample_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:

        has_mc_samples = (u.ndim == 3)

        u_stop = u if not self.stop_gradients else u.detach()
        u_norm = self.u_layernorm(u_stop)

        context = self.sample_embed(sample_index.squeeze(-1) if sample_index.ndim == 2 else sample_index)
        context = self.sample_layernorm(context)
        if has_mc_samples:
            context = context.unsqueeze(0).expand(u.shape[0], -1, -1)

        eps = self.attention(u_norm, context)
        z_base = self.z_base_proj(u)

        return z_base, eps