import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .components import ModalityInputBlock, AttentionBlock 

class EncoderXYU(nn.Module):
    """
    Encoder mapping (x, y, sample_idx) -> q(u | x, y, sample_idx).

    This is the first-level encoder in a hierarchical VAE, producing latent variable u.

    Parameters
    ----------
    n_genes
        Number of gene features.
    n_proteins
        Number of protein features.
    n_latent
        Dimensionality of the output latent variable u.
    n_hidden
        Size of hidden layers in ModalityInputBlocks and for sample embedding.
    n_sample
        Number of unique sample categories for conditioning.
    n_layers
        Number of layers in the final MLPs generating u's parameters.
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

        self.loc_layer = nn.Sequential(*[
            nn.Linear(n_hidden, n_hidden), nn.ReLU()
        ] * (n_layers - 1) + [nn.Linear(n_hidden, n_latent)])
        self.scale_layer = nn.Sequential(*[
            nn.Linear(n_hidden, n_hidden), nn.ReLU()
        ] * (n_layers - 1) + [nn.Linear(n_hidden, n_latent)])
        
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
        
        loc = self.loc_layer(h)
        scale = self.softplus(self.scale_layer(h)) + self.scale_eps
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
        self.n_latent_u_eff = n_latent_u if n_latent_u is not None else n_latent
        self.stop_gradients = stop_gradients

        self.layer_norm_u = nn.LayerNorm(self.n_latent_u_eff, elementwise_affine=False)
        self.sample_layernorm = nn.LayerNorm(n_latent_sample)
        self.sample_embed = nn.Embedding(n_sample, n_latent_sample)

        attention_out_dim = (2 if not use_map else 1) * n_latent
        self.attention = AttentionBlock(
            query_dim=self.n_latent_u_eff,
            kv_dim=n_latent_sample,
            out_dim=attention_out_dim,
            outerprod_dim=n_latent_sample,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            n_hidden_mlp=n_hidden,
            n_layers_mlp=n_layers,
            stop_gradients_mlp=stop_gradients_mlp,
            use_map=use_map,
        )
        if self.n_latent_u_eff != self.n_latent:
            self.z_base_proj = nn.Linear(self.n_latent_u_eff, n_latent)
        else:
            self.z_base_proj = nn.Identity()

    def forward(
        self,
        u: torch.Tensor,
        sample_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        u_proc = u
        u_for_norm = u
        context_norm = None

        _squeezed_sample_idx = sample_index.squeeze(-1) if sample_index.ndim == 2 else sample_index

        if self.stop_gradients:
            u_proc = u.detach()
            u_for_norm = u_proc
            with torch.no_grad():
                 context = self.sample_embed(_squeezed_sample_idx)
                 context_norm = self.sample_layernorm(context)
            # assert not context_norm.requires_grad, "INTERNAL: context_norm should NOT require grad" # For testing
        else:
            context = self.sample_embed(_squeezed_sample_idx)
            context_norm = self.sample_layernorm(context)

        u_norm = self.layer_norm_u(u_for_norm)
        # if self.stop_gradients: # For testing
        #      assert not u_norm.requires_grad, "INTERNAL: u_norm should NOT require grad"

        eps = self.attention(u_norm, context_norm)
        z_base = self.z_base_proj(u_proc)
        # if self.stop_gradients: # For testing
        #     assert not z_base.requires_grad, "INTERNAL: z_base should NOT require grad"
        return z_base, eps

class BackgroundProteinEncoder(nn.Module):
    """
    Infers a Normal distribution over log protein background levels (log(beta)).

    Conditioned on latent variable z and batch_index.

    Parameters
    ----------
    n_latent
        Dimensionality of latent variable z.
    n_batch
        Number of experimental batches. If 0, no batch effect is modeled via one-hot.
    n_proteins
        Number of protein features.
    n_hidden
        Number of hidden units in the MLP.
    """
    def __init__(self, n_latent: int, n_batch: int, n_proteins: int, n_hidden: int = 128):
        super().__init__()
        self.n_batch = n_batch
        self.n_proteins = n_proteins

        input_dim = n_latent
        if self.n_batch > 0:
            input_dim += self.n_batch
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(n_hidden, n_proteins)
        self.var_layer = nn.Linear(n_hidden, n_proteins) # Outputs log_scale

    def forward(self, z: torch.Tensor, batch_index: torch.Tensor) -> Normal:
        S_dim, B_dim = -1, -1
        z_flat = z
        batch_index_expanded = None
        
        if z.ndim == 3:
            S_dim = z.shape[0]
            B_dim = z.shape[1]
            z_flat = z.reshape(S_dim * B_dim, -1)
            if self.n_batch > 0:
                batch_index_expanded = batch_index.unsqueeze(0).expand(S_dim, B_dim).reshape(S_dim * B_dim)
        elif z.ndim == 2:
            B_dim = z.shape[0]
            if self.n_batch > 0:
                batch_index_expanded = batch_index
        else:
            raise ValueError(f"z has unexpected ndim: {z.ndim}")

        if self.n_batch > 0:
            _batch_idx_squeezed = batch_index_expanded.squeeze(-1) if batch_index_expanded.ndim == 2 and batch_index_expanded.shape[-1] == 1 else batch_index_expanded
            one_hot_batch = F.one_hot(_batch_idx_squeezed, num_classes=self.n_batch).float()
            x_combined = torch.cat([z_flat, one_hot_batch], dim=-1)
        else:
            x_combined = z_flat

        h = self.encoder(x_combined)
        mean = self.mean_layer(h)
        log_scale = self.var_layer(h)
        scale = F.softplus(log_scale) + 1e-4

        if S_dim != -1 :
            mean = mean.reshape(S_dim, B_dim, self.n_proteins)
            scale = scale.reshape(S_dim, B_dim, self.n_proteins)
        return Normal(mean, scale)