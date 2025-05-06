import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import MLP, AttentionBlock

class DecoderZXAttention(nn.Module):
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
    n_latent_batch_embed
        Dimension of the batch embedding.
    n_hidden
        Hidden dimension for the main residual MLP.
    n_layers_attn_mlp
        Number of layers for MLPs within the internal AttentionBlock.
    n_heads_attn
        Number of attention heads for the internal AttentionBlock.
    dropout_rate_attn
        Dropout rate for AttentionBlock and residual MLP.
    dispersion
        Dispersion type: "gene" or "gene-cell".
    """
    def __init__(
        self,
        n_latent: int,
        n_output_genes: int,
        n_batch: int, 
        n_latent_batch_embed: int = 16, 
        n_hidden: int = 128,
        n_layers_attn_mlp: int = 1, 
        n_heads_attn: int = 2,      
        dropout_rate_attn: float = 0.0, 
        dispersion: str = "gene", 
    ):
        super().__init__()
        assert dispersion in {"gene", "gene-cell"}
        self.dispersion = dispersion
        self.n_output_genes = n_output_genes
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.n_latent_batch_embed = n_latent_batch_embed

        if self.n_batch > 0:
            self.batch_embed_proj = nn.Linear(n_batch, n_latent_batch_embed)
            self.batch_embed_norm = nn.LayerNorm(n_latent_batch_embed)
        
        self.z_norm_layer = nn.LayerNorm(n_latent)
        self.attention = AttentionBlock(
            query_dim=n_latent,
            kv_dim=n_latent_batch_embed,
            out_dim=n_latent, 
            outerprod_dim=n_latent_batch_embed, 
            n_heads=n_heads_attn,
            dropout_rate=dropout_rate_attn,
            n_hidden_mlp=n_hidden, 
            n_layers_mlp=n_layers_attn_mlp,
            use_map=True 
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(),
            nn.Dropout(dropout_rate_attn), 
            nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(),
            nn.Dropout(dropout_rate_attn), 
        )
        self.scale_decoder = nn.Linear(n_hidden, n_output_genes)
        if dispersion == "gene-cell":
            self.r_decoder = nn.Linear(n_hidden, n_output_genes) # Outputs log_r
        else: 
            self.px_r_log = nn.Parameter(torch.randn(n_output_genes)) 

    def forward(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
        library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S_dim, B_dim_in_z = -1, -1
        z_flat = z
        batch_index_expanded_flat = None
        library_flat = library

        if z.ndim == 3:
            S_dim = z.shape[0]
            B_dim_in_z = z.shape[1]
            z_flat = z.reshape(S_dim * B_dim_in_z, -1)
            if self.n_batch > 0:
                _batch_idx_input_squeezed = batch_index.squeeze(-1) if batch_index.ndim == 2 and batch_index.shape[-1] == 1 else batch_index
                assert _batch_idx_input_squeezed.ndim == 1
                if _batch_idx_input_squeezed.shape[0] == B_dim_in_z:
                    batch_index_expanded_flat = _batch_idx_input_squeezed.unsqueeze(0).expand(S_dim, B_dim_in_z).reshape(S_dim * B_dim_in_z)
                else: raise ValueError(f"DecoderZXAttention: z B_dim {B_dim_in_z} != batch_index B_dim {_batch_idx_input_squeezed.shape[0]}")
            
            if library.ndim == 3 and library.shape[0]==S_dim and library.shape[1]==B_dim_in_z : 
                 library_flat = library.reshape(S_dim * B_dim_in_z, -1)
            elif library.ndim == 2 and library.shape[0]==B_dim_in_z : 
                 library_flat = library.unsqueeze(0).expand(S_dim, B_dim_in_z, -1).reshape(S_dim*B_dim_in_z, -1)
            else: raise ValueError(f"DecoderZXAttention: Library shape {library.shape} incompatible with z {z.shape}")
        
        elif z.ndim == 2:
            B_dim_in_z = z.shape[0]
            if self.n_batch > 0:
                _batch_idx_input_squeezed = batch_index.squeeze(-1) if batch_index.ndim == 2 and batch_index.shape[-1] == 1 else batch_index
                assert _batch_idx_input_squeezed.ndim == 1
                if _batch_idx_input_squeezed.shape[0] != B_dim_in_z: raise ValueError(f"DecoderZXAttention: z B_dim {B_dim_in_z} != batch_index B_dim {_batch_idx_input_squeezed.shape[0]}")
                batch_index_expanded_flat = _batch_idx_input_squeezed
            
            if library.ndim == 3 and library.shape[0] == 1 and library.shape[1] == B_dim_in_z: 
                library_flat = library.squeeze(0)
            elif library.ndim != 2 or library.shape[0] != B_dim_in_z or library.shape[1] !=1:
                 raise ValueError(f"DecoderZXAttention: Library shape {library.shape} incompatible with z {z.shape} (expected B,1)")
        else:
            raise ValueError(f"DecoderZXAttention: z has unexpected ndim: {z.ndim}")

        z_norm = self.z_norm_layer(z_flat)
        attn_output = torch.zeros_like(z_norm)
        if self.n_batch > 0:
            batch_one_hot = F.one_hot(batch_index_expanded_flat, self.n_batch).float()
            batch_embed = self.batch_embed_proj(batch_one_hot)
            batch_embed_norm = self.batch_embed_norm(batch_embed)
            attn_output = self.attention(z_norm, batch_embed_norm)
        
        residual = z_flat + attn_output
        h = self.residual_mlp(residual)
        px_scale_logits = self.scale_decoder(h) 
        px_scale = F.softmax(px_scale_logits, dim=-1)
        px_rate = torch.exp(library_flat) * px_scale

        if self.dispersion == "gene-cell":
            px_r_log = self.r_decoder(h) 
            px_r = torch.exp(px_r_log) + 1e-8
        else: 
            px_r = torch.exp(self.px_r_log).expand(h.size(0), -1) + 1e-8

        if S_dim != -1:
            px_scale = px_scale.reshape(S_dim, B_dim_in_z, -1)
            px_rate = px_rate.reshape(S_dim, B_dim_in_z, -1)
            px_r = px_r.reshape(S_dim, B_dim_in_z, -1)
        return px_scale, px_rate, px_r

class ProteinDecoderZYAttention(nn.Module):
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
    n_latent_batch_embed
        Dimension of the batch embedding.
    n_hidden
        Hidden dimension for the main MLP.
    n_layers_attn_mlp
        Number of layers for MLPs within the internal AttentionBlock.
    n_heads_attn
        Number of attention heads for the internal AttentionBlock.
    dropout_rate_attn
        Dropout rate for AttentionBlock and MLP.
    dispersion
        Dispersion type: "protein" or "protein-cell".
    """
    def __init__(
        self,
        n_latent: int,
        n_output_proteins: int,
        n_batch: int,
        n_latent_batch_embed: int = 16, 
        n_hidden: int = 128,
        n_layers_attn_mlp: int = 1,
        n_heads_attn: int = 2,
        dropout_rate_attn: float = 0.1, 
        dispersion: str = "protein", 
    ):
        super().__init__()
        assert dispersion in {"protein", "protein-cell"}
        self.dispersion = dispersion
        self.n_output_proteins = n_output_proteins
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_latent_batch_embed = n_latent_batch_embed

        if self.n_batch > 0:
            self.batch_embed_proj = nn.Linear(n_batch, n_latent_batch_embed)
            self.batch_embed_norm = nn.LayerNorm(n_latent_batch_embed)
        
        self.z_norm_layer = nn.LayerNorm(n_latent)
        self.attention = AttentionBlock(
            query_dim=n_latent,
            kv_dim=n_latent_batch_embed,
            out_dim=n_latent, 
            outerprod_dim=n_latent_batch_embed,
            n_heads=n_heads_attn,
            dropout_rate=dropout_rate_attn,
            n_hidden_mlp=n_hidden,
            n_layers_mlp=n_layers_attn_mlp,
            use_map=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(),
            nn.Dropout(dropout_rate_attn),
            nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(),
            nn.Dropout(dropout_rate_attn),
        )
        self.foreground_scale_decoder = nn.Linear(n_hidden, n_output_proteins) 
        self.background_mixing_decoder = nn.Linear(n_hidden, n_output_proteins)
        if dispersion == "protein-cell":
            self.r_decoder_log = nn.Linear(n_hidden, n_output_proteins) 
        else: 
            self.py_r_log = nn.Parameter(torch.randn(n_output_proteins))

    def forward(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
        logbeta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        S_dim, B_dim_in_z = -1, -1
        z_flat = z
        batch_index_expanded_flat = None
        logbeta_flat = logbeta

        if z.ndim == 3: 
            S_dim = z.shape[0]
            B_dim_in_z = z.shape[1]
            z_flat = z.reshape(S_dim * B_dim_in_z, -1)
            if self.n_batch > 0:
                _batch_idx_input_squeezed = batch_index.squeeze(-1) if batch_index.ndim == 2 and batch_index.shape[-1] == 1 else batch_index
                assert _batch_idx_input_squeezed.ndim == 1
                if _batch_idx_input_squeezed.shape[0] == B_dim_in_z:
                    batch_index_expanded_flat = _batch_idx_input_squeezed.unsqueeze(0).expand(S_dim, B_dim_in_z).reshape(S_dim * B_dim_in_z)
                else: raise ValueError(f"ProteinDecoder: z B_dim {B_dim_in_z} != batch_index B_dim {_batch_idx_input_squeezed.shape[0]}")
            
            if logbeta.ndim == 3 and logbeta.shape[0]==S_dim and logbeta.shape[1]==B_dim_in_z : 
                 logbeta_flat = logbeta.reshape(S_dim * B_dim_in_z, -1)
            elif logbeta.ndim == 2 and logbeta.shape[0]==B_dim_in_z : 
                 logbeta_flat = logbeta.unsqueeze(0).expand(S_dim, B_dim_in_z, -1).reshape(S_dim*B_dim_in_z, -1)
            else: raise ValueError(f"ProteinDecoder: Logbeta shape {logbeta.shape} incompatible with z {z.shape}")
        elif z.ndim == 2: 
            B_dim_in_z = z.shape[0]
            if self.n_batch > 0:
                _batch_idx_input_squeezed = batch_index.squeeze(-1) if batch_index.ndim == 2 and batch_index.shape[-1] == 1 else batch_index
                assert _batch_idx_input_squeezed.ndim == 1
                if _batch_idx_input_squeezed.shape[0] != B_dim_in_z: raise ValueError(f"ProteinDecoder: z B_dim {B_dim_in_z} != batch_index B_dim {_batch_idx_input_squeezed.shape[0]}")
                batch_index_expanded_flat = _batch_idx_input_squeezed
            
            if logbeta.ndim == 3 and logbeta.shape[0] == 1 and logbeta.shape[1] == B_dim_in_z: 
                logbeta_flat = logbeta.squeeze(0)
            elif logbeta.ndim != 2 or logbeta.shape[0] != B_dim_in_z:
                 raise ValueError(f"ProteinDecoder: Logbeta shape {logbeta.shape} incompatible with z {z.shape} (expected B,P)")
        else:
            raise ValueError(f"ProteinDecoder: z has unexpected ndim: {z.ndim}")

        rate_back = torch.exp(logbeta_flat)
        z_norm = self.z_norm_layer(z_flat)
        attn_output = torch.zeros_like(z_norm)
        if self.n_batch > 0:
            batch_one_hot = F.one_hot(batch_index_expanded_flat, self.n_batch).float()
            batch_embed = self.batch_embed_proj(batch_one_hot)
            batch_embed_norm = self.batch_embed_norm(batch_embed)
            attn_output = self.attention(z_norm, batch_embed_norm)
            
        residual = z_flat + attn_output
        h = self.mlp(residual)
        log_delta_decoded = self.foreground_scale_decoder(h) 
        fore_scale_factor = 1 + F.softplus(log_delta_decoded) + 1e-8 
        rate_fore = rate_back * fore_scale_factor 
        mixing_logits = self.background_mixing_decoder(h) 
        py_scale_unused = F.normalize((1-torch.sigmoid(mixing_logits))*rate_fore, p=1, dim=-1)

        if self.dispersion == "protein-cell":
            py_r_log = self.r_decoder_log(h) 
            py_r = torch.exp(py_r_log) + 1e-8
        else: 
            py_r = torch.exp(self.py_r_log).expand(h.size(0), -1) + 1e-8

        if S_dim != -1:
            rate_back = rate_back.reshape(S_dim, B_dim_in_z, -1)
            rate_fore = rate_fore.reshape(S_dim, B_dim_in_z, -1)
            mixing_logits = mixing_logits.reshape(S_dim, B_dim_in_z, -1)
            py_scale_unused = py_scale_unused.reshape(S_dim, B_dim_in_z, -1)
            py_r = py_r.reshape(S_dim, B_dim_in_z, -1)
        return rate_back, rate_fore, mixing_logits, py_scale_unused, py_r