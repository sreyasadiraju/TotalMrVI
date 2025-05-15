import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import MLP, AttentionBlock

from scvi.distributions import NegativeBinomial, NegativeBinomialMixture

class DecoderZX(nn.Module):
    """
    Decodes gene expression from latent z, conditioned on batch and library size.

    Uses an attention mechanism for z to attend to a batch embedding.

    Parameters
    ----------
    n_latent
        Dimensionality of latent variable z.
    n_output_genes
        Number of gene features to output.
    n_batch
        Number of batch categories. If 0, batch attention is skipped.
    n_latent_sample
        Dimension of the batch embedding.
    n_channels
        Number of channels for the internal AttentionBlock.
    n_heads
        Number of attention heads for the internal AttentionBlock.
    dropout_rate
        Dropout rate for AttentionBlock.
    stop_gradients
        Whether to stop gradient flow of query embedding before AttentionBlock.
    stop_gradients_mlp
        Whether to stop gradient flow of MLPs within the internal AttentionBlock.
    n_hidden
        Hidden dimension for MLPs within the internal AttentionBlock.
    n_layers
        Number of layers for MLPs within the internal AttentionBlock.
    low_dim_batch
        If True, batch embedding is low-dimensional.
    dispersion
        Dispersion type: "gene" or "gene-cell".
    """
    def __init__(
        self,
        n_latent: int,
        n_output_genes: int,
        n_batch: int, 
        n_latent_sample: int = 16, 
        n_channels: int = 4,
        n_heads: int = 2,
        dropout_rate: float = 0.0,
        stop_gradients: bool = False,
        stop_gradients_mlp: bool = False,
        n_hidden: int = 128,
        n_layers: int = 1,
        low_dim_batch: bool = False, 
        dispersion: str = "gene", 
    ):
        super().__init__()
        assert dispersion in {"gene", "gene-cell"}
        self.dispersion = dispersion
        self.n_output_genes = n_output_genes
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.n_latent_sample = n_latent_sample
        self.stop_gradients = stop_gradients
        self.low_dim_batch = low_dim_batch
        self.using_attention = (n_batch >= 2)

        if self.using_attention:
            self.batch_embedding = nn.Embedding(n_batch, n_latent_sample)
            self.batch_norm_layer = nn.LayerNorm(n_latent_sample)

            res_dim = n_latent if low_dim_batch else n_output_genes
        
            self.z_norm_layer = nn.LayerNorm(n_latent)
            self.attention = AttentionBlock(
                query_dim=n_latent,
                kv_dim=n_latent_sample,
                out_dim=res_dim,
                outerprod_dim=n_latent_sample,
                n_channels=n_channels,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                n_hidden=n_hidden,
                n_layers=n_layers,
                stop_gradients_mlp=stop_gradients_mlp,
            )

            if self.low_dim_batch:
                self.attention_proj = nn.Linear(n_latent, n_output_genes)
            else:
                self.attention_proj = nn.Linear(n_output_genes, n_output_genes)
        else:
            self.output_proj = nn.Linear(n_latent, n_output_genes)
        self.activation = nn.Softmax(dim=-1)

        self.inverse_dispersion = nn.Parameter(torch.randn(n_output_genes,)) if dispersion == "gene" else nn.Linear(n_latent, n_output_genes)

    def forward(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
        library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        library = library.squeeze(-1) if library.ndim == 2 else library

        z_stop = z if not self.stop_gradients else z.detach()

        if self.using_attention:
            has_mc_samples = (z.ndim == 3)

            batch_embed = self.batch_embedding(batch_index.squeeze(-1) if batch_index.ndim == 2 else batch_index)
            batch_embed_norm = self.batch_norm_layer(batch_embed)

            if has_mc_samples:
                batch_embed_norm = batch_embed_norm.unsqueeze(0).expand(z.shape[0], -1, -1)
            
            z_norm = self.z_norm_layer(z_stop)
            residual = self.attention(z_norm, batch_embed_norm)

            if self.low_dim_batch:
                mu = self.attention_proj(z + residual)
            else:
                mu = self.attention_proj(z) + residual
        else:
            mu = self.output_proj(z_norm)
        mu = self.activation(mu)
        theta = self.inverse_dispersion if self.dispersion == "gene" else self.inverse_dispersion(z_norm)
        theta = torch.exp(theta) + 1e-8
        return {
            "rho": mu,
            "theta": theta,
            "library": library
        }


class DecoderZY(nn.Module):
    """
    Decodes protein expression from latent z, conditioned on batch and log_beta.

    Uses an attention mechanism for z to attend to a batch embedding.
    Outputs parameters for a Negative Binomial Mixture distribution.

    Parameters
    ----------
    n_latent
        Dimensionality of latent variable z.
    n_output_proteins
        Number of protein features to output.
    n_batch
        Number of batch categories. If 0, batch attention is skipped.
    n_latent_sample
        Dimension of the batch embedding.
    n_channels
        Number of channels for the internal AttentionBlocks.
    n_heads
        Number of attention heads for the internal AttentionBlocks.
    dropout_rate
        Dropout rate for AttentionBlocks.
    stop_gradients
        Whether to stop gradient flow of query embedding before AttentionBlocks.
    stop_gradients_mlp
        Whether to stop gradient flow of MLPs within the internal AttentionBlocks.
    n_hidden
        Hidden dimension for MLPs within the internal AttentionBlocks.
    n_layers
        Number of layers for MLPs within the internal AttentionBlocks.
    low_dim_batch
        If True, batch embedding is low-dimensional.
    dispersion
        Dispersion type: "protein" or "protein-cell".
    """
    def __init__(
        self,
        n_latent: int,
        n_output_proteins: int,
        n_batch: int, 
        n_latent_sample: int = 16, 
        n_channels: int = 4,
        n_heads: int = 2,
        dropout_rate: float = 0.0,
        stop_gradients: bool = False,
        stop_gradients_mlp: bool = False,
        n_hidden: int = 128,
        n_layers: int = 1,
        low_dim_batch: bool = False, 
        dispersion: str = "protein", 
    ):
        super().__init__()
        assert dispersion in {"protein", "protein-cell"}
        self.dispersion = dispersion
        self.n_output_proteins = n_output_proteins
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.n_latent_sample = n_latent_sample
        self.stop_gradients = stop_gradients
        self.low_dim_batch = low_dim_batch
        self.using_attention = (n_batch >= 2)

        if self.using_attention:
            self.batch_embedding = nn.Embedding(n_batch, n_latent_sample)
            self.batch_norm_layer = nn.LayerNorm(n_latent_sample)

            res_dim = n_latent if low_dim_batch else n_output_proteins
        
            self.z_norm_layer = nn.LayerNorm(n_latent)

            self.attention_background = AttentionBlock(
                query_dim=n_latent,
                kv_dim=n_latent_sample,
                out_dim=res_dim,
                outerprod_dim=n_latent_sample,
                n_channels=n_channels,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                n_hidden=n_hidden,
                n_layers=n_layers,
                stop_gradients_mlp=stop_gradients_mlp,
            )
            self.attention_foreground = AttentionBlock(
                query_dim=n_latent,
                kv_dim=n_latent_sample,
                out_dim=res_dim,
                outerprod_dim=n_latent_sample,
                n_channels=n_channels,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                n_hidden=n_hidden,
                n_layers=n_layers,
                stop_gradients_mlp=stop_gradients_mlp,
            )
            self.attention_mixing = AttentionBlock(
                query_dim=n_latent,
                kv_dim=n_latent_sample,
                out_dim=res_dim,
                outerprod_dim=n_latent_sample,
                n_channels=n_channels,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                n_hidden=n_hidden,
                n_layers=n_layers,
                stop_gradients_mlp=stop_gradients_mlp,
            )

            if self.low_dim_batch:
                self.attention_proj_background = nn.Linear(n_latent, n_hidden)
                self.attention_proj_foreground = nn.Linear(n_latent, n_output_proteins)
                self.attention_proj_mixing = nn.Linear(n_latent, n_output_proteins)
            else:
                self.attention_proj_background = nn.Linear(n_output_proteins, n_hidden)
                self.attention_proj_foreground = nn.Linear(n_output_proteins, n_output_proteins)
                self.attention_proj_mixing = nn.Linear(n_output_proteins, n_output_proteins)
        else:
            self.mlp_background = MLP(n_latent, n_hidden, n_hidden, n_layers)
            self.mlp_foreground = MLP(n_latent, n_output_proteins, n_output_proteins, n_layers)
            self.mlp_mixing = MLP(n_latent, n_output_proteins, n_output_proteins, n_layers)
        self.background_mean_readout = nn.Linear(n_hidden, n_output_proteins)
        self.background_scale_readout = nn.Linear(n_hidden, n_output_proteins)

        self.inverse_dispersion = nn.Parameter(torch.randn(n_output_proteins,)) if dispersion == "gene" else nn.Linear(n_latent, n_output_proteins)

    def forward(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        z_stop = z if not self.stop_gradients else z.detach()

        if self.using_attention:
            has_mc_samples = (z.ndim == 3)

            batch_embed = self.batch_embedding(batch_index.squeeze(-1) if batch_index.ndim == 2 else batch_index)
            batch_embed_norm = self.batch_norm_layer(batch_embed)
            if has_mc_samples:
                batch_embed_norm = batch_embed_norm.unsqueeze(0).expand(z.shape[0], -1, -1)
            
            z_norm = self.z_norm_layer(z_stop)
            residual_background = self.attention_background(z_norm, batch_embed_norm)
            residual_foreground = self.attention_foreground(z_norm, batch_embed_norm)
            residual_mixing = self.attention_mixing(z_norm, batch_embed_norm)

            if self.low_dim_batch:
                h_background = self.attention_proj_background(z + residual_background)
                alpha = self.attention_proj_foreground(z + residual_foreground)
                pi_logits = self.attention_proj_mixing(z + residual_mixing)
            else:
                h_background = self.attention_proj_background(z) + residual_background
                alpha = self.attention_proj_foreground(z) + residual_foreground
                pi_logits = self.attention_proj_mixing(z) + residual_mixing
        else:
            h_background = self.mlp_background(z_norm)
            alpha = self.mlp_foreground(z_norm)
            pi_logits = self.mlp_mixing(z_norm)
        logbeta_loc = self.background_mean_readout(h_background)
        logbeta_scale = self.background_scale_readout(h_background)
        logbeta_scale = F.softplus(logbeta_scale) + 1e-8
        alpha = F.relu(alpha) + 1 + 1e-8
        # pi = F.sigmoid(pi) + 1e-8
        phi = self.inverse_dispersion if self.dispersion == "protein" else self.inverse_dispersion(z_norm)
        phi = torch.exp(phi) + 1e-8
        return {
            "logbeta_loc": logbeta_loc,
            "logbeta_scale": logbeta_scale,
            "alpha": alpha,
            "pi_logits": pi_logits,
            "phi": phi
        }