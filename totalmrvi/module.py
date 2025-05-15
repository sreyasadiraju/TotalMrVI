import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture
from torch.distributions import Categorical, Normal, MixtureSameFamily, kl_divergence

from .encoders import EncoderXYU, EncoderUZ
from .decoders import DecoderZX, DecoderZY

DEFAULT_QU_KWARGS = {
    "n_hidden": 128,
    "n_layers": 1,
}
DEFAULT_QZ_KWARGS = {
    "n_latent_sample": 16,
    "n_channels": 4,
    "n_heads": 2,
    "dropout_rate": 0,
    "stop_gradients": False,
    "stop_gradients_mlp": False,
    "use_map": True,
    "n_hidden": 32,
    "n_layers": 1
}
DEFAULT_PX_KWARGS = {
    "n_latent_sample": 16,
    "n_channels": 4,
    "n_heads": 2,
    "dropout_rate": 0,
    "stop_gradients": False,
    "stop_gradients_mlp": False,
    "n_hidden": 128,
    "n_layers": 1,
    "low_dim_batch": False,
}
DEFAULT_PY_KWARGS = {
    "n_latent_sample": 16,
    "n_channels": 4,
    "n_heads": 2,
    "dropout_rate": 0,
    "stop_gradients": False,
    "stop_gradients_mlp": False,
    "n_hidden": 128,
    "n_layers": 1,
    "low_dim_batch": False,
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
        dispersion_pro: str = "protein",
        dispersion_rna: str = "gene",
        z_u_prior: bool = True,
        z_u_prior_scale: float = 0.0,
        u_prior_scale: float = 0.0,
        u_prior_mixture: bool = True,
        u_prior_mixture_k: int = 20,
        learn_z_u_prior_scale: bool = False,
        protein_background_prior_mean: np.ndarray | None = None,
        protein_background_prior_scale: np.ndarray | None = None,
        scale_observations: bool = False,
        n_obs_per_sample: np.ndarray | None = None,
        qu_kwargs: dict | None = None,
        qz_kwargs: dict | None = None,
        px_kwargs: dict | None = None,
        py_kwargs: dict | None = None,
    ):
        super().__init__()

        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.n_sample = n_sample
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent = n_latent
        self.n_latent_u = n_latent if n_latent_u is None else n_latent_u
        self.dispersion_pro = dispersion_pro
        self.dispersion_rna = dispersion_rna

        self.z_u_prior = z_u_prior
        self.z_u_prior_scale = z_u_prior_scale
        self.u_prior_scale = u_prior_scale
        self.u_prior_mixture = u_prior_mixture
        self.u_prior_mixture_k = u_prior_mixture_k
        self.learn_z_u_prior_scale = learn_z_u_prior_scale

        self.scale_observations = scale_observations
        self.n_obs_per_sample = n_obs_per_sample

        self.qu_kwargs = DEFAULT_QU_KWARGS.copy()
        if qu_kwargs is not None:
            self.qu_kwargs.update(qu_kwargs)

        self.qz_kwargs = DEFAULT_QZ_KWARGS.copy()
        if qz_kwargs is not None:
            self.qz_kwargs.update(qz_kwargs)

        self.px_kwargs = DEFAULT_PX_KWARGS.copy()
        if px_kwargs is not None:
            self.px_kwargs.update(px_kwargs)

        self.py_kwargs = DEFAULT_PY_KWARGS.copy()
        if py_kwargs is not None:
            self.py_kwargs.update(py_kwargs)

        self.qu = EncoderXYU(n_genes=n_input_genes, n_proteins=n_input_proteins, n_latent=self.n_latent_u, n_sample=n_sample,
                             **self.qu_kwargs)
        self.qz = EncoderUZ(n_latent=n_latent, n_sample=n_sample, n_latent_u=self.n_latent_u,
                            **self.qz_kwargs)
        self.px = DecoderZX(n_latent=n_latent, n_output_genes=n_input_genes, n_batch=n_batch, dispersion=dispersion_rna,
                            **self.px_kwargs)
        self.py = DecoderZY(n_latent=n_latent, n_output_proteins=n_input_proteins, n_batch=n_batch, dispersion=dispersion_pro,
                            **self.py_kwargs)

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.zeros(n_latent))
        else:
            self.pz_scale = z_u_prior_scale
        if self.u_prior_mixture:
            u_prior_mixture_k = self.n_labels if self.n_labels > 1 else u_prior_mixture_k
            self.u_prior_logits = nn.Parameter(torch.zeros(u_prior_mixture_k,))
            self.u_prior_means = nn.Parameter(torch.normal(0, 1, (u_prior_mixture_k, self.n_latent_u)))
            self.u_prior_logscales = nn.Parameter(torch.zeros(u_prior_mixture_k, self.n_latent_u))

        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.log_background_loc = torch.nn.Parameter(
                    3.0 * torch.ones(n_input_proteins, n_batch)
                )
                self.log_background_logscale= torch.nn.Parameter(
                    torch.zeros(n_input_proteins, n_batch)
                )
            else:
                self.log_background_loc = torch.nn.Parameter(torch.ones(n_input_proteins))
                self.log_background_logscale = torch.nn.Parameter(torch.zeros(n_input_proteins))
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.log_background_loc = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.log_background_logscale = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)) + 1e-8)
            )

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        
        sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY]
        if sample_index.ndim == 2 and sample_index.shape[-1] == 1:
            sample_index = sample_index.squeeze(-1)
        sample_index = sample_index.long() 

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        if batch_index.ndim == 2 and batch_index.shape[-1] == 1:
            batch_index = batch_index.squeeze(-1)
        batch_index = batch_index.long()

        assert sample_index.ndim == 1, f"sample_index after processing should be 1D, got {sample_index.ndim}"
        assert batch_index.ndim == 1, f"batch_index after processing should be 1D, got {batch_index.ndim}"
        
        return {"x": x, "y": y, "sample_index": sample_index, "batch_index": batch_index}

    @auto_move_data
    def inference(
        self, x: torch.Tensor, y: torch.Tensor, sample_index: torch.Tensor, batch_index: torch.Tensor,
        mc_samples: int | None = None, cf_sample: torch.Tensor | None = None, use_mean: bool = False,
    ) -> dict[str, torch.Tensor | Normal | None]:
        
        qu: Normal = self.qu(x, y, sample_index)
        _sample_shape = (mc_samples,) if mc_samples is not None else ()
        u = qu.mean if use_mean else qu.rsample(_sample_shape)

        context_index = sample_index if cf_sample is None else cf_sample

        z_base, eps = self.qz(u, context_index)
        qeps_ = eps

        qeps = None
        if qeps_.shape[-1] == 2 * self.n_latent:
            mean_eps, scale_eps = qeps_[..., :self.n_latent], qeps_[..., self.n_latent:]
            scale_eps = F.softplus(scale_eps) + 1e-5
            qeps = Normal(loc=mean_eps, scale=scale_eps)
            eps = qeps.mean if use_mean else qeps.rsample()
        z = z_base + eps

        library = torch.log(x.sum(dim=1, keepdim=True) + 1e-8)

        return {
            "qu": qu, "qeps": qeps, "eps": eps, "u": u, "z": z, "z_base": z_base, "library": library, 
        }

    def _get_generative_input(
        self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        
        z = inference_outputs["z"]

        library = inference_outputs["library"]

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        batch_index = batch_index.squeeze(-1) if (batch_index.ndim == 2 and batch_index.shape[-1] == 1) else batch_index
        batch_index = batch_index.long()

        label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
        label_index = label_index.squeeze(-1) if (label_index.ndim == 2 and label_index.shape[-1] == 1) else label_index
        label_index = label_index.long()  

        return {
            "z": z, "library": library, "batch_index": batch_index, "label_index": label_index,
        }

    @auto_move_data
    def generative(
        self, z: torch.Tensor, library: torch.Tensor, batch_index: torch.Tensor, label_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        
        library_exp = torch.exp(library)
        px = self.px(z, batch_index, library_exp)
        py = self.py(z, batch_index)

        logbeta = Normal(py["logbeta_loc"], py["logbeta_scale"])
        beta = torch.exp(logbeta.rsample())

        if self.u_prior_mixture:
            offset = (
                10.0 * F.one_hot(label_index, self.n_labels) if self.n_labels >= 2 else 0.0
            )
            cats = Categorical(logits = self.u_prior_logits + offset)
            normal_dists = Normal(self.u_prior_means, torch.exp(self.u_prior_scales))
            pu = MixtureSameFamily(cats, normal_dists)
        else:
            pu = Normal(0, torch.exp(self.u_prior_scale))

        return {
            "pu": pu,
            "px_rho": px["rho"], "px_library": px["library"], "px_theta": px["theta"],
            "py_logbeta_loc": py["logbeta_loc"], "py_logbeta_scale": py["logbeta_scale"], "py_beta": beta,
            "py_alpha": py["alpha"], "py_pi_logits": py["pi_logits"], "py_phi": py["phi"],
        }

    def loss(
        self, tensors: dict[str, torch.Tensor], 
        inference_outputs: dict[str, torch.Tensor | Normal | None],
        generative_outputs: dict[str, torch.Tensor], 
        kl_weight: float = 1.0, protein_recon_weight: float = 1.0,
    ) -> LossOutput:
        
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        px_dist = NegativeBinomial(
            generative_outputs["px_library"] * generative_outputs["px_rho"], generative_outputs["px_theta"]
        )
        reconstruction_loss_gene = -px_dist.log_prob(x).sum(-1)

        py_dist = NegativeBinomialMixture(mu1=generative_outputs["py_beta"], 
                                          mu2=generative_outputs["py_alpha"]*generative_outputs["py_beta"],
                                          theta1=generative_outputs["py_phi"],
                                          mixture_logits=generative_outputs["py_pi_logits"])
        reconstruction_loss_protein = -py_dist.log_prob(y).sum(-1)
        reconstruction_loss = reconstruction_loss_gene + protein_recon_weight * reconstruction_loss_protein

        if self.u_prior_mixture:
            kl_u = inference_outputs["qu"].log_prob(inference_outputs["u"]).sum(-1) - generative_outputs["pu"].log_prob(inference_outputs["u"])
        else:
            kl_u = kl_divergence(inference_outputs["qu"], generative_outputs["pu"]).sum(-1)

        kl_z = 0.0
        eps = inference_outputs["z"] - inference_outputs["z_base"]
        if self.z_u_prior:
            peps = Normal(0, torch.exp(self.pz_scale))
            kl_z = -peps.log_prob(eps).sum(-1)

        logbeta_prior = Normal(self.log_background_loc, torch.exp(self.log_background_logscale))
        logbeta_posterior = Normal(generative_outputs["py_logbeta_loc"], generative_outputs["py_logbeta_scale"])
        kl_beta = kl_divergence(logbeta_posterior, logbeta_prior).sum(-1)

        weighted_kl_loss = kl_weight * (kl_u + kl_z + kl_beta)

        loss = reconstruction_loss + weighted_kl_loss

        if self.scale_observations:
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].flatten().astype(int)
            prefactors = self.n_obs_per_sample[sample_index]
            loss = loss / prefactors

        loss = torch.mean(loss)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=(kl_u + kl_z + kl_beta),
            extra_metrics = {"gene_reconstruction": reconstruction_loss_gene, "protein_reconstruction": reconstruction_loss_protein,
                             "kl_u": kl_u, "kl_z": kl_z, "kl_beta": kl_beta},
        )