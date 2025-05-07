import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture
from torch.distributions import Normal, kl_divergence as kl

from .encoders import EncoderXYU, EncoderUZ, BackgroundProteinEncoder
from .decoders import DecoderZXAttention, ProteinDecoderZYAttention

DEFAULT_QU_KWARGS = {}
DEFAULT_QZ_KWARGS = {
    "use_map": True,
    "stop_gradients": False,
    "stop_gradients_mlp": True,
    "dropout_rate": 0.03,
}

class TOTALMRVAE(BaseModuleClass):
    """
    Variational Autoencoder for CITE-seq data, extending MrVI with TOTALVI-like protein modeling.

    Models RNA and protein expression, incorporating sample and batch effects.
    Uses a hierarchical latent space (u -> z).

    Parameters
    ----------
    n_input_genes
        Number of input gene features.
    n_input_proteins
        Number of input protein features.
    n_sample
        Number of unique sample categories for conditioning.
    n_batch
        Number of unique batch categories for conditioning. If 0, batch effects
        related to one-hot encoding or batch-specific priors are not modeled.
    n_labels
        Number of cell type labels (currently passed but minimally used by this module's core logic).
    n_latent
        Dimensionality of the main latent variable z.
    n_latent_u
        Dimensionality of the intermediate latent variable u. Defaults to `n_latent`.
    n_hidden
        General hidden layer size. Used as default for `EncoderXYU(n_hidden=...)`,
        `BackgroundProteinEncoder(n_hidden=...)`, `DecoderZXAttention(n_hidden=...)`,
        and `ProteinDecoderZYAttention(n_hidden=...)` if not overridden by more specific
        parameters or `qu_kwargs`/`qz_kwargs`.
    encoder_n_layers
        Default number of layers for the MLPs in `EncoderXYU` (for its `loc_layer` and `scale_layer`).
        Can be overridden if "n_layers" is in `qu_kwargs`.
    decoder_n_hidden_attn_mlp
        Default hidden dimension for MLPs within `AttentionBlock`s used in `EncoderUZ`,
        `DecoderZXAttention`, and `ProteinDecoderZYAttention`. Can be overridden if "n_hidden"
        (for `EncoderUZ`) or relevant keys are in `qz_kwargs` or if specific decoder attention
        params are passed to their respective modules (not currently exposed at `TOTALMRVAE` level for decoders).
    decoder_n_layers_attn_mlp
        Default number of layers for the post-MLP within `AttentionBlock`s used in `EncoderUZ`,
        `DecoderZXAttention`, and `ProteinDecoderZYAttention`. Can be overridden if "n_layers"
        (for `EncoderUZ`) or relevant keys are in `qz_kwargs`.
    decoder_n_heads_attn
        Default number of attention heads for `AttentionBlock`s in `EncoderUZ` and decoders.
    decoder_dropout_rate_attn
        Default dropout rate for `AttentionBlock`s and subsequent MLPs in `EncoderUZ` and decoders.
    n_latent_batch_embed
        Dimension for batch embeddings used in decoders' attention mechanisms (`DecoderZXAttention`,
        `ProteinDecoderZYAttention`).
    dispersion_pro
        Dispersion parameter type for proteins: "protein" (shared per protein) or
        "protein-cell" (specific to each cell-protein pair).
    dispersion_rna
        Dispersion parameter type for RNA: "gene" (shared per gene) or
        "gene-cell" (specific to each cell-gene pair).
    protein_background_prior_mean
        NumPy array for the mean of the Normal prior on log_beta (protein background).
        Shape should be (n_input_proteins,) or (n_input_proteins, n_batch).
    protein_background_prior_scale
        NumPy array for the scale of the Normal prior on log_beta.
        Shape should be (n_input_proteins,) or (n_input_proteins, n_batch).
    qu_kwargs
        Keyword arguments for :class:`~totalmrvi.encoders.EncoderXYU`. Can include "n_hidden" and
        "n_layers" to override defaults derived from `n_hidden` and `encoder_n_layers`.
    qz_kwargs
        Keyword arguments for :class:`~totalmrvi.encoders.EncoderUZ`. Can include "n_hidden",
        "n_layers", "n_heads", "dropout_rate", "stop_gradients_mlp", etc., to configure
        its internal `AttentionBlock` and other properties. These will override defaults
        derived from `decoder_n_hidden_attn_mlp`, `decoder_n_layers_attn_mlp`, etc.
    """
    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_sample: int,
        n_batch: int,
        n_labels: int,
        n_latent: int = 30,
        n_latent_u: int | None = None,
        n_hidden: int = 128, # General hidden size for components unless overridden
        encoder_n_layers: int = 2, # Default for EncoderXYU's internal MLPs
        decoder_n_hidden_attn_mlp: int = 32, # Default for AttentionBlock n_hidden_mlp in decoders AND qz
        decoder_n_layers_attn_mlp: int = 1,  # Default for AttentionBlock n_layers_mlp in decoders AND qz
        decoder_n_heads_attn: int = 2,       # Default for AttentionBlock n_heads in decoders AND qz
        decoder_dropout_rate_attn: float = 0.0, # Default for AttentionBlock dropout in decoders AND qz
        n_latent_batch_embed: int = 16,      # For decoders' batch attention kv_dim
        dispersion_pro: str = "protein",
        dispersion_rna: str = "gene-cell",
        protein_background_prior_mean: np.ndarray | None = None,
        protein_background_prior_scale: np.ndarray | None = None,
        qu_kwargs: dict | None = None,
        qz_kwargs: dict | None = None,
    ):
        super().__init__()

        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.n_sample = n_sample
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent = n_latent
        self.n_latent_u_eff = n_latent if n_latent_u is None else n_latent_u
        self.dispersion_pro = dispersion_pro
        self.dispersion_rna = dispersion_rna

        # --- EncoderXYU (qu) setup ---
        _qu_kwargs_processed = DEFAULT_QU_KWARGS.copy()
        if qu_kwargs is not None:
            _qu_kwargs_processed.update(qu_kwargs)
        _n_hidden_for_qu = _qu_kwargs_processed.pop("n_hidden", n_hidden)
        _n_layers_for_qu = _qu_kwargs_processed.pop("n_layers", encoder_n_layers)
        self.qu = EncoderXYU(
            n_genes=n_input_genes, n_proteins=n_input_proteins, n_latent=self.n_latent_u_eff,
            n_hidden=_n_hidden_for_qu, n_sample=n_sample, n_layers=_n_layers_for_qu,
            **_qu_kwargs_processed,
        )

        # --- EncoderUZ (qz) setup ---
        _qz_kwargs_processed = DEFAULT_QZ_KWARGS.copy()
        if qz_kwargs is not None:
            _qz_kwargs_processed.update(qz_kwargs)
        
        # Parameters for EncoderUZ, which will then pass them to its internal AttentionBlock:
        # EncoderUZ's `n_hidden` param becomes AttentionBlock's `n_hidden_mlp`.
        # EncoderUZ's `n_layers` param becomes AttentionBlock's `n_layers_mlp`.
        _n_hidden_for_qz_attention_mlp = _qz_kwargs_processed.pop("n_hidden", decoder_n_hidden_attn_mlp)
        _n_layers_for_qz_attention_mlp = _qz_kwargs_processed.pop("n_layers", decoder_n_layers_attn_mlp)
        # Other AttentionBlock params come from EncoderUZ defaults or _qz_kwargs_processed
        _n_heads_for_qz_attn = _qz_kwargs_processed.pop("n_heads", decoder_n_heads_attn)
        _dropout_rate_for_qz_attn = _qz_kwargs_processed.pop("dropout_rate", decoder_dropout_rate_attn)
        # stop_gradients_mlp is also an EncoderUZ param passed to AttentionBlock
        _stop_grads_mlp_for_qz_attn = _qz_kwargs_processed.pop("stop_gradients_mlp", DEFAULT_QZ_KWARGS.get("stop_gradients_mlp", True))


        self.qz = EncoderUZ(
            n_latent=n_latent,
            n_sample=n_sample,
            n_latent_u=self.n_latent_u_eff if self.n_latent_u_eff != self.n_latent else None,
            # Pass the correct args to EncoderUZ based on its signature
            n_hidden=_n_hidden_for_qz_attention_mlp, # This will be n_hidden_mlp in its AttentionBlock
            n_layers=_n_layers_for_qz_attention_mlp,   # This will be n_layers_mlp in its AttentionBlock
            n_heads=_n_heads_for_qz_attn,
            dropout_rate=_dropout_rate_for_qz_attn,
            stop_gradients_mlp=_stop_grads_mlp_for_qz_attn,
            # Other direct params for EncoderUZ come from _qz_kwargs_processed or its own defaults
            **_qz_kwargs_processed, 
        )

        # --- BackgroundProteinEncoder setup ---
        self.background_encoder = BackgroundProteinEncoder(
            n_latent=n_latent, n_batch=n_batch, n_proteins=n_input_proteins, n_hidden=n_hidden,
        )

        # --- DecoderZXAttention (px) setup ---
        self.px = DecoderZXAttention(
            n_latent=n_latent, n_output_genes=n_input_genes, n_batch=n_batch,
            n_latent_batch_embed=n_latent_batch_embed, n_hidden=n_hidden,
            n_layers_attn_mlp=decoder_n_layers_attn_mlp, # For its internal AttentionBlock
            n_heads_attn=decoder_n_heads_attn,           # For its internal AttentionBlock
            dropout_rate_attn=decoder_dropout_rate_attn, # For its internal AttentionBlock & MLP
            dispersion=self.dispersion_rna,
        )

        # --- ProteinDecoderZYAttention (py) setup ---
        self.py = ProteinDecoderZYAttention(
            n_latent=n_latent, n_output_proteins=n_input_proteins, n_batch=n_batch,
            n_latent_batch_embed=n_latent_batch_embed, n_hidden=n_hidden,
            n_layers_attn_mlp=decoder_n_layers_attn_mlp, # For its internal AttentionBlock
            n_heads_attn=decoder_n_heads_attn,           # For its internal AttentionBlock
            dropout_rate_attn=decoder_dropout_rate_attn, # For its internal AttentionBlock & MLP
            dispersion=dispersion_pro,
        )

        # --- Priors setup ---
        self.register_buffer("u_prior_means", torch.zeros(n_sample, self.n_latent_u_eff))
        self.register_buffer("u_prior_logscales", torch.zeros(n_sample, self.n_latent_u_eff))
        target_shape = (self.n_input_proteins,) if n_batch == 0 else (self.n_input_proteins, n_batch)
        if protein_background_prior_mean is None:
            self.prior_logbeta_loc = nn.Parameter(torch.zeros(*target_shape))
            self.prior_logbeta_logscale = nn.Parameter(torch.zeros(*target_shape) - 2.3)
        else:
            _prior_mean_np = protein_background_prior_mean.astype(np.float32)
            _prior_scale_np = protein_background_prior_scale.astype(np.float32)
            _prior_mean = torch.from_numpy(_prior_mean_np)
            _prior_scale = torch.from_numpy(_prior_scale_np)
            if n_batch == 0:
                if _prior_mean.ndim == 2 and _prior_mean.shape[1] == 1: _prior_mean = _prior_mean.squeeze(-1)
                if _prior_scale.ndim == 2 and _prior_scale.shape[1] == 1: _prior_scale = _prior_scale.squeeze(-1)
            else:
                if _prior_mean.ndim == 1: _prior_mean = _prior_mean.unsqueeze(-1)
                if _prior_scale.ndim == 1: _prior_scale = _prior_scale.unsqueeze(-1)
                if _prior_mean.shape[1] == 1 and target_shape[1] > 1: _prior_mean = _prior_mean.expand(-1, target_shape[1])
                if _prior_scale.shape[1] == 1 and target_shape[1] > 1: _prior_scale = _prior_scale.expand(-1, target_shape[1])
            if _prior_mean.shape != target_shape:
                raise ValueError(f"Prior mean shape {_prior_mean.shape} != target {target_shape}")
            if _prior_scale.shape != target_shape:
                raise ValueError(f"Prior scale shape {_prior_scale.shape} != target {target_shape}")
            self.prior_logbeta_loc = nn.Parameter(_prior_mean)
            self.prior_logbeta_logscale = nn.Parameter(torch.log(_prior_scale + 1e-8))

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].squeeze(-1) if tensors[REGISTRY_KEYS.SAMPLE_KEY].ndim == 2 and tensors[REGISTRY_KEYS.SAMPLE_KEY].shape[-1] == 1 else tensors[REGISTRY_KEYS.SAMPLE_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1) if tensors[REGISTRY_KEYS.BATCH_KEY].ndim == 2 and tensors[REGISTRY_KEYS.BATCH_KEY].shape[-1] == 1 else tensors[REGISTRY_KEYS.BATCH_KEY]
        assert sample_index.ndim == 1
        assert batch_index.ndim == 1
        return {"x": x, "y": y, "sample_index": sample_index, "batch_index": batch_index}

    @auto_move_data
    def inference(
        self, x: torch.Tensor, y: torch.Tensor, sample_index: torch.Tensor, batch_index: torch.Tensor,
        mc_samples: int = 1, cf_sample: torch.Tensor | None = None, use_mean: bool = False,
    ) -> dict[str, torch.Tensor | Normal | None]:
        x_ = x
        y_ = y
        qu = self.qu(x_, y_, sample_index)
        _rsample_shape = (mc_samples,) if mc_samples > 0 else ()
        u = qu.mean if use_mean else qu.rsample(_rsample_shape)
        context_idx_for_qz = sample_index if cf_sample is None else cf_sample
        z_base, eps_params_or_tensor = self.qz(u, context_idx_for_qz)
        qeps = None
        eps_sample = eps_params_or_tensor
        if isinstance(eps_params_or_tensor, tuple):
            mean_eps, log_scale_eps_chunk = eps_params_or_tensor
            scale_eps = F.softplus(log_scale_eps_chunk) + 1e-5
            qeps = Normal(loc=mean_eps, scale=scale_eps)
            eps_sample = qeps.mean if use_mean else qeps.rsample()
        z = z_base + eps_sample
        library = torch.log(x_.sum(dim=1, keepdim=True) + 1e-8)
        qbeta = self.background_encoder(z, batch_index)
        logbeta = qbeta.mean if use_mean else qbeta.rsample()
        beta = torch.exp(logbeta)
        library_out = library
        if z.ndim == 3:
            S_dim, B_dim = z.shape[0], z.shape[1]
            if library.ndim == 2 and library.shape[0] == B_dim:
                library_out = library.unsqueeze(0).expand(S_dim, B_dim, 1)
            elif library.shape != (S_dim, B_dim, 1):
                 raise ValueError(f"Unexpected library shape {library.shape} when z is 3D")
        return {
            "qu": qu, "qeps": qeps, "eps": eps_sample, "u": u, "z": z, "z_base": z_base,
            "library": library_out, "qbeta": qbeta, "logbeta": logbeta, "beta": beta,
            "mc_samples": mc_samples
        }

    def _get_generative_input(
        self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_index_orig = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1) if tensors[REGISTRY_KEYS.BATCH_KEY].ndim == 2 and tensors[REGISTRY_KEYS.BATCH_KEY].shape[-1] == 1 else tensors[REGISTRY_KEYS.BATCH_KEY]
        assert batch_index_orig.ndim == 1
        return {
            "z": inference_outputs["z"], "library": inference_outputs["library"],
            "batch_index": batch_index_orig, "logbeta": inference_outputs["logbeta"],
        }

    @auto_move_data
    def generative(
        self, z: torch.Tensor, library: torch.Tensor, batch_index: torch.Tensor, logbeta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert batch_index.ndim == 1, f"Generative: batch_index should be 1D (B,), got {batch_index.shape}"
        px_scale, px_rate, px_r = self.px(z, batch_index, library)
        py_rate_back, py_rate_fore, py_mixing_logits, py_scale_unused, py_r_pro = self.py(z, batch_index, logbeta)
        return {
            "px_scale": px_scale, "px_rate": px_rate, "px_r": px_r,
            "py_rate_back": py_rate_back, "py_rate_fore": py_rate_fore,
            "py_mixing": py_mixing_logits, "py_scale_unused": py_scale_unused, "py_r": py_r_pro,
        }

    def get_prior_u(self, sample_index: torch.Tensor, n_mc_samples: int = 0) -> Normal:
        _sample_idx = sample_index.squeeze(-1) if sample_index.ndim == 2 and sample_index.shape[-1] == 1 else sample_index
        assert _sample_idx.ndim == 1
        means = self.u_prior_means[_sample_idx]
        log_scales = self.u_prior_logscales[_sample_idx]
        if n_mc_samples > 0:
            means = means.unsqueeze(0).expand(n_mc_samples, -1, -1)
            log_scales = log_scales.unsqueeze(0).expand(n_mc_samples, -1, -1)
        return Normal(loc=means, scale=torch.exp(log_scales))

    def loss(
        self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor | Normal | None],
        generative_outputs: dict[str, torch.Tensor], kl_weight: float = 1.0, protein_recon_weight: float = 1.0,
    ) -> LossOutput:
        x_orig = tensors[REGISTRY_KEYS.X_KEY]
        y_orig = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        sample_idx_orig = (tensors[REGISTRY_KEYS.SAMPLE_KEY].squeeze(-1)
                           if tensors[REGISTRY_KEYS.SAMPLE_KEY].ndim == 2 and tensors[REGISTRY_KEYS.SAMPLE_KEY].shape[-1] == 1
                           else tensors[REGISTRY_KEYS.SAMPLE_KEY])
        batch_idx_orig  = (tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
                           if tensors[REGISTRY_KEYS.BATCH_KEY].ndim == 2 and tensors[REGISTRY_KEYS.BATCH_KEY].shape[-1] == 1
                           else tensors[REGISTRY_KEYS.BATCH_KEY])
        assert sample_idx_orig.ndim == 1
        assert batch_idx_orig.ndim == 1

        z_from_inference = inference_outputs["z"]
        S_dim_active = z_from_inference.shape[0] if z_from_inference.ndim == 3 else -1

        _x_expanded = x_orig.unsqueeze(0).expand(S_dim_active, -1, -1) if S_dim_active > 0 else x_orig
        _y_expanded = y_orig.unsqueeze(0).expand(S_dim_active, -1, -1) if S_dim_active > 0 else y_orig
        
        px_rate, px_r = generative_outputs["px_rate"], generative_outputs["px_r"]
        gene_recon = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(_x_expanded).sum(-1)

        py_rate_back, py_rate_fore = generative_outputs["py_rate_back"], generative_outputs["py_rate_fore"]
        py_mixing, py_r_pro = generative_outputs["py_mixing"], generative_outputs["py_r"]
        prot_recon_full = -NegativeBinomialMixture(
            mu1=py_rate_back, mu2=py_rate_fore, theta1=py_r_pro, mixture_logits=py_mixing,
        ).log_prob(_y_expanded)
        prot_recon = prot_recon_full.sum(-1)

        qu = inference_outputs["qu"]
        qeps = inference_outputs["qeps"]
        qbeta = inference_outputs["qbeta"]

        n_mc_for_pu = S_dim_active if qu.loc.ndim == 3 and S_dim_active > 0 else 0
        if qu.loc.ndim == 3 and S_dim_active <= 0: n_mc_for_pu = qu.loc.shape[0]
        pu = self.get_prior_u(sample_idx_orig, n_mc_samples=n_mc_for_pu)
        _qu_loc, _qu_scale = qu.loc, qu.scale
        if _qu_loc.shape != pu.loc.shape:
             if _qu_loc.ndim == 2 and pu.loc.ndim == 3:
                  _qu_loc = _qu_loc.unsqueeze(0).expand(pu.loc.shape)
                  _qu_scale = _qu_scale.unsqueeze(0).expand(pu.scale.shape)
             elif _qu_loc.ndim == 3 and pu.loc.ndim == 2:
                  _qu_loc = _qu_loc.mean(0); _qu_scale = (_qu_scale**2).mean(0).sqrt()
             else: raise ValueError(f"Shape mismatch KL(qu||pu): qu={qu.loc.shape}, pu={pu.loc.shape}")
        kl_u = kl(Normal(_qu_loc, _qu_scale), pu).sum(-1)

        kl_eps = torch.zeros_like(kl_u)
        if qeps is not None:
            loc_peps = torch.zeros_like(qeps.loc); scale_peps = torch.ones_like(qeps.scale)
            if qeps.loc.shape != loc_peps.shape:
                 loc_peps = loc_peps.expand_as(qeps.loc); scale_peps = scale_peps.expand_as(qeps.scale)
            kl_eps = kl(qeps, Normal(loc_peps, scale_peps)).sum(-1)

        p_logbeta_loc_params = self.prior_logbeta_loc
        p_logbeta_scale_params = torch.exp(self.prior_logbeta_logscale)
        qbeta_loc_shape = qbeta.loc.shape
        _batch_idx_for_pbeta_selection = batch_idx_orig
        if qbeta.loc.ndim == 3 and batch_idx_orig.ndim == 1:
            _batch_idx_for_pbeta_selection = batch_idx_orig.unsqueeze(0).expand(qbeta.loc.shape[0], -1)

        if self.n_batch > 0:
            assert p_logbeta_loc_params.ndim == 2 and p_logbeta_loc_params.shape[0] == self.n_input_proteins
            max_idx = _batch_idx_for_pbeta_selection.max()
            if max_idx >= self.n_batch: raise IndexError(f"Batch index {max_idx} for prior out of bounds ({self.n_batch})")
            p_loc_sel = p_logbeta_loc_params[:, _batch_idx_for_pbeta_selection]
            p_scale_sel = p_logbeta_scale_params[:, _batch_idx_for_pbeta_selection]
            p_logbeta_loc_final = p_loc_sel.permute(1,2,0) if p_loc_sel.ndim == 3 else p_loc_sel.permute(1,0)
            p_logbeta_scale_final = p_scale_sel.permute(1,2,0) if p_scale_sel.ndim == 3 else p_scale_sel.permute(1,0)
        elif self.n_batch == 0:
            assert p_logbeta_loc_params.ndim == 1 and p_logbeta_loc_params.shape[0] == self.n_input_proteins
            p_logbeta_loc_final = p_logbeta_loc_params.expand_as(qbeta.loc)
            p_logbeta_scale_final = p_logbeta_scale_params.expand_as(qbeta.scale)
        else:
            raise ValueError(f"Invalid n_batch value: {self.n_batch}")

        if qbeta_loc_shape != p_logbeta_loc_final.shape:
             if qbeta.loc.ndim == 3 and p_logbeta_loc_final.ndim == 2:
                  p_logbeta_loc_final = p_logbeta_loc_final.unsqueeze(0).expand(qbeta_loc_shape)
                  p_logbeta_scale_final = p_logbeta_scale_final.unsqueeze(0).expand(qbeta_loc_shape) # expand scale too
             elif qbeta.loc.ndim == 2 and p_logbeta_loc_final.ndim == 3:
                  p_logbeta_loc_final = p_logbeta_loc_final.mean(0)
                  p_logbeta_scale_final = (p_logbeta_scale_final**2).mean(0).sqrt()
             else: raise ValueError(f"Final shape mismatch KL(qbeta||pbeta): qbeta={qbeta_loc_shape}, pbeta={p_logbeta_loc_final.shape}")
        
        kl_beta = kl(qbeta, Normal(p_logbeta_loc_final, p_logbeta_scale_final)).sum(-1)
        loss = (gene_recon + protein_recon_weight * prot_recon + kl_weight * (kl_u + kl_eps + kl_beta)).mean()

        def _reduce(t): return t.mean(0) if t.ndim == 2 and S_dim_active > 0 else t
        return LossOutput(loss=loss, reconstruction_loss={"gene": _reduce(gene_recon.detach()), "protein": _reduce(prot_recon.detach())},
                          kl_local={"u": _reduce(kl_u.detach()), "eps": _reduce(kl_eps.detach()), "beta": _reduce(kl_beta.detach())})