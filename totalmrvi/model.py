import logging
import warnings
from collections.abc import Iterable
from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd
from pandas import get_dummies
import xarray as xr
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent, Chi2
from anndata import AnnData
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from scipy.stats import false_discovery_control

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data import _constants
from scvi.data._constants import _SCVI_UUID_KEY as ADATA_UUID_KEY_CONST
from scvi.dataloaders import DataSplitter
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture
from scvi.model.base import (
    BaseModelClass,
    RNASeqMixin,
    VAEMixin,
)
from scvi.model._utils import (
    _get_batch_code_from_category,
    _get_var_names_from_manager,
    get_max_epochs_heuristic,
)
from scvi.train import TrainRunner, TrainingPlan
from scvi.utils._docstrings import devices_dsp, setup_anndata_dsp

# Import from within your package
from .module import TOTALMRVAE

logger = logging.getLogger(__name__)


class TOTALMRVI(
    RNASeqMixin,
    VAEMixin,
    BaseModelClass,
):
    """
    Total Multi-Resolution Variational Inference (TOTALMRVI) model.

    Combines hierarchical latent variables for sample and batch modeling (inspired by MrVI)
    with a dedicated likelihood for CITE-seq protein data (inspired by TOTALVI).

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~totalmrvi.model.TOTALMRVI.setup_anndata`.
    n_latent
        Dimensionality of the main latent variable z. Default is 30.
    n_latent_u
        Dimensionality of the intermediate latent variable u. Defaults to `n_latent` if None.
    n_hidden
        General hidden layer size for MLPs and embeddings in `TOTALMRVAE`. Default is 128.
    encoder_n_layers
        Number of layers for MLPs in `EncoderXYU` and default for `EncoderUZ`'s internal MLPs. Default is 2.
    decoder_n_hidden_attn_mlp
        Hidden dimension for MLPs within AttentionBlocks in decoders. Default is 32.
    decoder_n_layers_attn_mlp
        Number of layers for MLPs within AttentionBlocks in decoders. Default is 1.
    decoder_n_heads_attn
        Number of attention heads for AttentionBlocks in decoders. Default is 2.
    decoder_dropout_rate_attn
        Dropout rate for AttentionBlocks and MLPs in decoders. Default is 0.0.
    n_latent_batch_embed
        Dimension for batch embeddings used in decoders' attention. Default is 16.
    dispersion_rna
        Dispersion parameter type for RNA in `TOTALMRVAE`: "gene" (shared per gene) or
        "gene-cell" (specific to each cell-gene pair). Default is "gene-cell".
    dispersion_pro
        Dispersion parameter type for proteins in `TOTALMRVAE`: "protein" (shared per protein) or
        "protein-cell" (specific to each cell-protein pair). Default is "protein".
    empirical_protein_background_prior
        If True, sets the initialization of protein background prior empirically by fitting a GMM.
        If False, randomly initializes. If None (default), sets to True if >10 proteins are used.
    qu_kwargs
        Keyword arguments for :class:`~totalmrvi.encoders.EncoderXYU`.
    qz_kwargs
        Keyword arguments for :class:`~totalmrvi.encoders.EncoderUZ`.
    **model_kwargs
        Additional keyword arguments passed to :class:`~totalmrvi.module.TOTALMRVAE`.
    """

    _module_cls = TOTALMRVAE
    _data_splitter_cls = DataSplitter
    _training_plan_cls = TrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 30,
        n_latent_u: int | None = None,
        n_hidden: int = 128,
        encoder_n_layers: int = 2,
        decoder_n_hidden_attn_mlp: int = 32,
        decoder_n_layers_attn_mlp: int = 1,
        decoder_n_heads_attn: int = 2,
        decoder_dropout_rate_attn: float = 0.0,
        n_latent_batch_embed: int = 16,
        dispersion_rna: Literal["gene", "gene-cell"] = "gene-cell",
        dispersion_pro: Literal["protein", "protein-cell"] = "protein",
        empirical_protein_background_prior: bool | None = None,
        qu_kwargs: dict | None = None,
        qz_kwargs: dict | None = None,
        **model_kwargs,
    ):
        super().__init__(adata)

        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (self.summary_stats.n_proteins > 10)
        )

        protein_background_prior_mean = None
        protein_background_prior_scale = None
        if emp_prior:
            logger.debug("TOTALMRVI __init__: emp_prior is True, attempting to calculate empirical priors.")
            try:
                protein_background_prior_mean, protein_background_prior_scale = (
                    self._get_protein_background_empirical_prior(
                        n_cells_for_prior_fitting=100
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to compute empirical protein background priors: {e}. "
                    "Using default (random) initialization for priors in the module."
                )
        else:
            logger.debug("TOTALMRVI __init__: emp_prior is False, skipping empirical prior calculation.")

        n_batch = self.summary_stats.n_batch
        n_sample = self.summary_stats.n_sample
        n_labels_for_module = self.summary_stats.n_labels

        self.module = self._module_cls(
            n_input_genes=self.summary_stats.n_vars,
            n_input_proteins=self.summary_stats.n_proteins,
            n_sample=n_sample,
            n_batch=n_batch,
            n_labels=n_labels_for_module,
            n_latent=n_latent,
            n_latent_u=n_latent_u,
            n_hidden=n_hidden,
            encoder_n_layers=encoder_n_layers,
            decoder_n_hidden_attn_mlp=decoder_n_hidden_attn_mlp,
            decoder_n_layers_attn_mlp=decoder_n_layers_attn_mlp,
            decoder_n_heads_attn=decoder_n_heads_attn,
            decoder_dropout_rate_attn=decoder_dropout_rate_attn,
            n_latent_batch_embed=n_latent_batch_embed,
            dispersion_pro=dispersion_pro,
            dispersion_rna=dispersion_rna,
            protein_background_prior_mean=protein_background_prior_mean,
            protein_background_prior_scale=protein_background_prior_scale,
            qu_kwargs=qu_kwargs,
            qz_kwargs=qz_kwargs,
            **model_kwargs,
        )

        self._model_summary_string = (
            f"TOTALMRVI Model with the following params: \n"
            f"n_latent: {n_latent}, n_latent_u: {self.module.n_latent_u_eff}, "
            f"n_sample: {n_sample}, n_batch: {n_batch}, \n"
            f"dispersion_rna: {dispersion_rna}, dispersion_pro: {dispersion_pro}."
        )
        self.init_params_ = self._get_init_params(locals())

    def _get_protein_background_empirical_prior(
        self, n_cells_for_prior_fitting: int = 100
    ):
        """
        Calculate empirical priors for protein background mean and scale.
        Uses `self.adata_manager` to access protein and batch data from `self.adata`.
        The shape of the returned priors is determined by `self.summary_stats.n_batch`.
        """
        logger.info("Calculating empirical priors for protein background using self.adata_manager.")
        pro_exp_array = self.adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
        if isinstance(pro_exp_array, pd.DataFrame):
            pro_exp_array = pro_exp_array.to_numpy()

        batch_indices_array = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).ravel()
        unique_batch_codes_in_data = np.unique(batch_indices_array)

        n_proteins = self.summary_stats.n_proteins
        n_batch_for_prior_shaping = self.summary_stats.n_batch

        target_prior_mean_shape = (
            (n_proteins, n_batch_for_prior_shaping) if n_batch_for_prior_shaping > 0 else (n_proteins,)
        )
        target_prior_scale_shape = (
            (n_proteins, n_batch_for_prior_shaping) if n_batch_for_prior_shaping > 0 else (n_proteins,)
        )

        batch_prior_means = np.zeros(target_prior_mean_shape, dtype=np.float32)
        batch_prior_scales = np.full(target_prior_scale_shape, 0.2, dtype=np.float32)

        if n_batch_for_prior_shaping == 0:
            all_pro_exp_sampled = pro_exp_array
            n_cells_to_sample_global = min(n_cells_for_prior_fitting, pro_exp_array.shape[0])
            if n_cells_to_sample_global > 0:
                sampled_cell_indices = np.random.choice(
                    np.arange(pro_exp_array.shape[0]),
                    size=n_cells_to_sample_global,
                    replace=False,
                )
                all_pro_exp_sampled = pro_exp_array[sampled_cell_indices, :]
            elif pro_exp_array.shape[0] == 0:
                logger.warning("No cells available for global prior calculation. Using defaults.")
                return batch_prior_means, batch_prior_scales

            gmm = GaussianMixture(n_components=2, random_state=0, reg_covar=1e-4)
            global_mus = []
            global_scales = []
            for p_idx in range(n_proteins):
                protein_values = all_pro_exp_sampled[:, p_idx]
                if np.all(protein_values < 1e-5) or len(np.unique(protein_values)) < 2:
                    global_mus.append(0.0)
                    global_scales.append(0.2)
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        gmm.fit(np.log1p(protein_values.reshape(-1, 1)))
                    if not gmm.converged_:
                        raise ConvergenceWarning(f"GMM (global, prot {p_idx}) did not converge.")
                    means_gmm = gmm.means_.ravel()
                    bg_mean_idx = np.argmin(means_gmm)
                    mu = means_gmm[bg_mean_idx]
                    scale = np.sqrt(max(gmm.covariances_[bg_mean_idx].ravel()[0], 1e-5))
                    global_mus.append(mu)
                    global_scales.append(scale if scale > 1e-3 else 0.2)
                except (ConvergenceWarning, ValueError) as e:
                    logger.debug(f"Global GMM failed for protein {p_idx}: {e}. Using default prior.")
                    global_mus.append(0.0)
                    global_scales.append(0.2)
            batch_prior_means = np.array(global_mus, dtype=np.float32)
            batch_prior_scales = np.array(global_scales, dtype=np.float32)
        else:
            for b_code_model_expects in range(n_batch_for_prior_shaping):
                if b_code_model_expects not in unique_batch_codes_in_data:
                    logger.debug(
                        f"Model expects batch category {b_code_model_expects}, but it's not in GMM fitting data. Using default priors for this batch."
                    )
                    batch_prior_means[:, b_code_model_expects] = 0.0
                    batch_prior_scales[:, b_code_model_expects] = 0.2
                    continue

                cells_in_this_batch_mask = batch_indices_array == b_code_model_expects
                pro_exp_this_batch = pro_exp_array[cells_in_this_batch_mask, :]

                if pro_exp_this_batch.shape[0] == 0:
                    logger.debug(f"No cells found for batch code {b_code_model_expects}. Using default priors.")
                    batch_prior_means[:, b_code_model_expects] = 0.0
                    batch_prior_scales[:, b_code_model_expects] = 0.2
                    continue

                n_cells_to_sample_this_batch = min(
                    n_cells_for_prior_fitting, pro_exp_this_batch.shape[0]
                )
                sampled_cell_indices_this_batch = np.random.choice(
                    np.arange(pro_exp_this_batch.shape[0]),
                    size=n_cells_to_sample_this_batch,
                    replace=False,
                )
                pro_exp_this_batch_sampled = pro_exp_this_batch[sampled_cell_indices_this_batch, :]

                gmm = GaussianMixture(n_components=2, random_state=0, reg_covar=1e-4)
                current_batch_mus = []
                current_batch_scales = []
                for p_idx in range(n_proteins):
                    protein_values = pro_exp_this_batch_sampled[:, p_idx]
                    if np.all(protein_values < 1e-5) or len(np.unique(protein_values)) < 2:
                        current_batch_mus.append(0.0)
                        current_batch_scales.append(0.2)
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            gmm.fit(np.log1p(protein_values.reshape(-1, 1)))
                        if not gmm.converged_:
                            raise ConvergenceWarning(
                                f"GMM (batch {b_code_model_expects}, prot {p_idx}) did not converge."
                            )
                        means_gmm = gmm.means_.ravel()
                        bg_mean_idx = np.argmin(means_gmm)
                        mu = means_gmm[bg_mean_idx]
                        scale = np.sqrt(max(gmm.covariances_[bg_mean_idx].ravel()[0], 1e-5))
                        current_batch_mus.append(mu)
                        current_batch_scales.append(scale if scale > 1e-3 else 0.2)
                    except (ConvergenceWarning, ValueError) as e:
                        logger.debug(f"GMM for batch {b_code_model_expects}, protein {p_idx} failed: {e}. Defaulting.")
                        current_batch_mus.append(0.0)
                        current_batch_scales.append(0.2)
                batch_prior_means[:, b_code_model_expects] = np.array(current_batch_mus, dtype=np.float32)
                batch_prior_scales[:, b_code_model_expects] = np.array(current_batch_scales, dtype=np.float32)

        return batch_prior_means, batch_prior_scales

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        protein_expression_obsm_key: str,
        layer: str | None = None,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        sample_key: str | None = None,
        labels_key: str | None = None,
        # Removed categorical_covariate_keys and continuous_covariate_keys from signature
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            Key in `adata.obsm` for protein expression data (counts).
        layer
            Key from `adata.layers` for gene expression counts. If `None`, `adata.X` is used.
        protein_names_uns_key
            Optional key in `adata.uns` where a list of protein names is stored.
            If `None`, and `adata.obsm[protein_expression_obsm_key]` is a DataFrame,
            its column names are used. Otherwise, proteins are named `protein_0`, `protein_1`, etc.
        batch_key
            Key in `adata.obs` for batch information. Used for batch effect correction and
            batch-specific protein background priors. If `None`, model assumes a single batch.
        sample_key
            Key in `adata.obs` for sample information. Crucial for the MrVI hierarchical latent
            space conditioning (u is sample-specific). If `None`, model assumes a single sample.
        labels_key
            Key in `adata.obs` for label information (e.g., cell types). Passed to the module.
            If `None`, model assumes a single label category.
        """ # Removed %(param_cat_cov_keys)s and %(param_cont_cov_keys)s
        setup_method_args = cls._get_setup_method_args(**locals())

        batch_ann_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        sample_ann_field = fields.CategoricalObsField(REGISTRY_KEYS.SAMPLE_KEY, sample_key)
        labels_ann_field = fields.CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key)

        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            sample_ann_field,
            batch_ann_field,
            labels_ann_field,
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=False,
                batch_field=batch_ann_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        lr: float = 2e-3,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = None,
        reduce_lr_on_plateau: bool = True,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 50,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `scvi.model._utils.get_max_epochs_heuristic(self.adata.n_obs)`.
        lr
            Learning rate for optimization.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range `[0.0, 1.0]`.
        validation_size
            Size of the validation set. If `None`, defaults to `1 - train_size`.
        shuffle_set_split
            Whether to shuffle indices before splitting.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping.
        check_val_every_n_epoch
            Check validation loss every `n` epochs. If `None` and `early_stopping` or
            `reduce_lr_on_plateau` is `True`, defaults to 1.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale KL weight from 0 to 1.
            Only used if `n_epochs_kl_warmup` is `None`. If both are `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale KL weight from 0 to 1. Overrides `n_steps_kl_warmup`
            if both are provided.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Overrides defaults set by
            this method's parameters (e.g., `lr`, `n_epochs_kl_warmup`).
        **trainer_kwargs
            Other keyword arguments passed to the :class:`~scvi.train.TrainRunner` and subsequently
            to PyTorch Lightning's `Trainer`. Examples include `enable_checkpointing`,
            `logger`, `callbacks`, etc.
        """
        if n_epochs_kl_warmup is None and n_steps_kl_warmup is None and self.adata is not None:
            n_steps_kl_warmup = int(0.75 * self.adata.n_obs)
        elif n_epochs_kl_warmup is not None:
            n_steps_kl_warmup = None

        _plan_kwargs_from_train_signature = {
            "lr": lr,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            _plan_kwargs_from_train_signature.update(plan_kwargs)
        final_plan_kwargs = _plan_kwargs_from_train_signature

        if max_epochs is None:
            if self.adata is not None:
                max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            else:
                max_epochs = 200
                logger.warning("adata not available for max_epochs heuristic. Defaulting to 200.")

        datasplitter_kwargs = trainer_kwargs.pop("datasplitter_kwargs", {})

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            **datasplitter_kwargs,
        )

        training_plan = self._training_plan_cls(self.module, **final_plan_kwargs)

        if (early_stopping or reduce_lr_on_plateau) and check_val_every_n_epoch is None:
            check_val_every_n_epoch = 1

        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **trainer_kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 1,
        batch_size: int | None = None,
        representation_kind: Literal["u", "z", "z_base", "eps"] = "z",
        return_dist: bool = False,
    ) -> Union[np.ndarray, torch.distributions.Normal]:
        """
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in `adata` to use. If `None`, all cells are used.
        give_mean
            Give the mean of the latent distribution for representation. If `False`, samples from
            the distribution. `mc_samples` controls the number of samples if `give_mean` is `False`.
        mc_samples
            Number of Monte Carlo samples to generate, if `give_mean` is `False`.
            If `give_mean` is `True`, this parameter is ignored for sampling.
            When `give_mean=False`, this is the number of samples drawn.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        representation_kind
            Which latent representation to return:
            - "u": The first-level latent variable.
            - "z": The second-level latent variable (z_base + eps).
            - "z_base": The base component of z, derived from u before adding sample-specific eps.
            - "eps": The sample-specific deviation `eps` that is added to `z_base`.
        return_dist
            If True and `representation_kind` is "u" or "eps" (and `EncoderUZ.use_map` is False for "eps"),
            returns the distribution object (`torch.distributions.Normal`). `give_mean` and `mc_samples` are ignored.

        Returns
        -------
        latent_representation
            If `return_dist` is `True`, returns `torch.distributions.Normal` object.
            Otherwise, returns `np.ndarray`.
            If `give_mean` is `True` or (`give_mean` is `False` and `mc_samples` is 1),
            shape is `(n_cells, n_latent_dim)`.
            If `give_mean` is `False` and `mc_samples` > 1, shape is `(mc_samples, n_cells, n_latent_dim)`.
            `n_latent_dim` depends on `representation_kind`.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        all_reps_list = []
        loc_list_dist, scale_list_dist = [], []
        self.module.eval()

        for tensors in scdl:
            _mc_samples_for_module = 0
            _use_mean_for_module = True

            if not return_dist:
                if give_mean:
                    _mc_samples_for_module = 0
                    _use_mean_for_module = True
                else:
                    _mc_samples_for_module = mc_samples
                    _use_mean_for_module = False

            inf_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(
                **inf_inputs, mc_samples=_mc_samples_for_module, use_mean=_use_mean_for_module
            )

            if return_dist:
                dist_to_collect = None
                if representation_kind == "u":
                    dist_to_collect = outputs["qu"]
                elif representation_kind == "eps":
                    dist_to_collect = outputs["qeps"]
                    if dist_to_collect is None:
                        raise ValueError(
                            "Cannot return distribution for 'eps' if EncoderUZ's 'use_map' is True. "
                            "Try `representation_kind='eps'` with `return_dist=False`, or set `use_map=False` in qz_kwargs."
                        )
                elif representation_kind in ["z", "z_base"]:
                    raise ValueError(f"Cannot return distribution for '{representation_kind}'. Try return_dist=False.")
                else:
                    raise ValueError(f"Invalid representation_kind for return_dist=True: {representation_kind}")

                loc_list_dist.append(dist_to_collect.loc.detach().cpu())
                scale_list_dist.append(dist_to_collect.scale.detach().cpu())

            else:
                rep_tensor = None
                if representation_kind == "u":
                    rep_tensor = outputs["u"]
                elif representation_kind == "z":
                    rep_tensor = outputs["z"]
                elif representation_kind == "z_base":
                    rep_tensor = outputs["z_base"]
                elif representation_kind == "eps":
                    rep_tensor = outputs["eps"]
                else:
                    raise ValueError(f"Invalid representation_kind: {representation_kind}")

                if not give_mean and mc_samples == 1 and rep_tensor.ndim == 3 and rep_tensor.shape[0] == 1:
                    rep_tensor = rep_tensor.squeeze(0)
                all_reps_list.append(rep_tensor.detach().cpu())

        if return_dist:
            if not loc_list_dist:
                return torch.distributions.Normal(torch.empty(0), torch.empty(0))
            final_locs = torch.cat(loc_list_dist, dim=0)
            final_scales = torch.cat(scale_list_dist, dim=0)
            return torch.distributions.Normal(final_locs, final_scales)
        else:
            if not all_reps_list:
                return np.array([])

            if all_reps_list[0].ndim == 3:
                full_rep_tensor = torch.cat(all_reps_list, dim=1)
            else:
                full_rep_tensor = torch.cat(all_reps_list, dim=0)
            return full_rep_tensor.numpy()

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_samples: int = 1,
        transform_batch: str | int | Sequence[str | int] | None = None,
        gene_list: Sequence[str] | None = None,
        protein_list: Sequence[str] | None = None,
        library_size: float | Literal["latent"] = 1.0,
        protein_expression_type: Literal["foreground", "corrected_total", "total"] = "foreground",
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
    ) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
        """
        Computes the model's normalized expression for RNA and protein.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to draw for each cell. For `n_samples > 1`, the mean
            or all an array of samples is returned (see `return_mean`).
        transform_batch
            Batch category to condition on for the generative model. If `None`, uses the batch
            cells were originally in. If a category is not present in the setup AnnData,
            an error will be raised. Can be a sequence, in which case outputs are averaged
            over the categories.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets.
        protein_list
            Return protein expression for a subset of proteins.
        library_size
            For RNA: If "latent", returns the model's estimated rate (`px_rate`).
            If a float, scales the estimated proportions (`px_scale`) by this float.
        protein_expression_type
            For Protein:
            - "foreground": Computes `rate_fore * P(foreground)`. Denoised signal.
            - "total": Computes `rate_fore * P(foreground) + rate_back * P(background)`. Total expected count.
            - "corrected_total": Computes "total", then scales sum over all proteins to 1 for each cell.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            If `True` and `n_samples > 1`, returns the mean of the `n_samples` samples.
            Otherwise (if `n_samples > 1`), returns all `n_samples` samples.
        return_numpy
            If `True`, returns `np.ndarray` instead of `pd.DataFrame`.

        Returns
        -------
        Tuple with two elements:
        - **RNA normalized expression**: `pd.DataFrame` or `np.ndarray`.
          Shape is `(n_cells, n_genes)` if `n_samples == 1` or `return_mean == True`,
          else `(n_samples, n_cells, n_genes)`.
        - **Protein normalized expression**: `pd.DataFrame` or `np.ndarray`.
          Shape is `(n_cells, n_proteins)` if `n_samples == 1` or `return_mean == True`,
          else `(n_samples, n_cells, n_proteins)`.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )

        # Prepare gene and protein masks using BOOLEAN logic
        all_gene_names = _get_var_names_from_manager(adata_manager) # np.ndarray
        if gene_list is None:
            gene_mask_bool = np.ones(len(all_gene_names), dtype=bool)
            genes_to_return = all_gene_names
        else:
            gene_set = set(gene_list)
            gene_mask_bool = np.array([g in gene_set for g in all_gene_names]) # Boolean mask
            genes_to_return = all_gene_names[gene_mask_bool]
            if len(genes_to_return) == 0:
                raise ValueError("No genes from `gene_list` were found in the model.")
        n_genes_out = len(genes_to_return)

        protein_state_reg = adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
        all_protein_names = np.array(protein_state_reg.column_names) # Use numpy for easier masking
        if protein_list is None:
            protein_mask_bool = np.ones(len(all_protein_names), dtype=bool)
            proteins_to_return = all_protein_names
        else:
            protein_set = set(protein_list)
            protein_mask_bool = np.array([p in protein_set for p in all_protein_names]) # Boolean mask
            proteins_to_return = all_protein_names[protein_mask_bool]
            if len(proteins_to_return) == 0:
                raise ValueError("No proteins from `protein_list` were found in the model.")
        n_proteins_out = len(proteins_to_return)

        # Prepare transform_batch
        if transform_batch is None:
            batch_categories_to_transform = [None]
        else:
            batch_categories_to_transform = _get_batch_code_from_category(
                adata_manager, transform_batch
            )

        all_rna_outputs_list = []
        all_protein_outputs_list = []

        self.module.eval()
        for tensors in scdl:
            n_cells_in_batch = tensors[REGISTRY_KEYS.X_KEY].shape[0]
            
            rna_minibatch_all_samples = torch.zeros(
                n_samples, n_cells_in_batch, n_genes_out, device=self.device
            )
            protein_minibatch_all_samples = torch.zeros(
                n_samples, n_cells_in_batch, n_proteins_out, device=self.device
            )

            # --- Loop over posterior samples ---
            for i in range(n_samples):
                inference_kwargs_iter = {"mc_samples": 1, "use_mean": False} 
                inf_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(**inf_inputs, **inference_kwargs_iter)

                rna_current_sample_sum_over_batches = torch.zeros(
                    n_cells_in_batch, n_genes_out, device=self.device
                )
                protein_current_sample_sum_over_batches = torch.zeros(
                    n_cells_in_batch, n_proteins_out, device=self.device
                )

                # --- Loop over target batch conditions ---
                for batch_idx_in_transform_loop, target_batch_code in enumerate(batch_categories_to_transform): 
                    gen_inputs = { # Prepare inputs for generative model
                        "z": inference_outputs["z"], 
                        "library": inference_outputs["library"],
                        "logbeta": inference_outputs["logbeta"],
                        "batch_index": (
                            tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1).long()
                            if target_batch_code is None
                            else torch.full((n_cells_in_batch,), 
                                             fill_value=target_batch_code, dtype=torch.long)
                        ).to(self.device)
                    }
                    generative_outputs = self.module.generative(**gen_inputs) # Call generative model

                    # --- RNA Calculation ---
                    px_scale_full = generative_outputs["px_scale"].squeeze(0) 
                    px_rate_full = generative_outputs["px_rate"].squeeze(0)   
                    if library_size == "latent":
                        rna_expr_one_batch = px_rate_full[:, gene_mask_bool]
                    else:
                        rna_expr_one_batch = px_scale_full[:, gene_mask_bool] * float(library_size)
                    rna_current_sample_sum_over_batches += rna_expr_one_batch

                    # --- Protein Calculation (Raw Components) ---
                    pi_logit_background_full = generative_outputs["py_mixing"].squeeze(0)
                    p_fg_full = torch.sigmoid(-pi_logit_background_full) # P(foreground)
                    p_bg_full = 1.0 - p_fg_full                          # P(background)
                    rate_fore_full = generative_outputs["py_rate_fore"].squeeze(0)
                    rate_back_full = generative_outputs["py_rate_back"].squeeze(0)
                    current_foreground_contribution = rate_fore_full * p_fg_full
                    current_background_contribution = rate_back_full * p_bg_full

                    # ============================================================== #
                    # ==== START DEBUG BLOCK FOR PROTEIN (Correct Placement) ======= #
                    # ============================================================== #
                    if i == 0 and batch_idx_in_transform_loop == 0 and n_cells_in_batch > 0:
                        debug_cell_in_batch_idx = 0 
                        original_prot_idx_to_debug = 0 
                        num_proteins_in_tensor = rate_fore_full.shape[1] 

                        if num_proteins_in_tensor > original_prot_idx_to_debug:
                            print(f"    [ DEBUG get_norm_expr - Sample {i}, Target Batch {target_batch_code} ] CellIdx(batch) {debug_cell_in_batch_idx}, OrigProtIdx {original_prot_idx_to_debug}:")
                            
                            raw_logit = pi_logit_background_full[debug_cell_in_batch_idx, original_prot_idx_to_debug].item()
                            raw_rate_fore = rate_fore_full[debug_cell_in_batch_idx, original_prot_idx_to_debug].item()
                            raw_rate_back = rate_back_full[debug_cell_in_batch_idx, original_prot_idx_to_debug].item()
                            print(f"      Raw logit_bg: {raw_logit:.4f} (Is NaN: {np.isnan(raw_logit)})")
                            print(f"      Raw rate_fore: {raw_rate_fore:.4f} (Is NaN: {np.isnan(raw_rate_fore)})")
                            print(f"      Raw rate_back: {raw_rate_back:.4f} (Is NaN: {np.isnan(raw_rate_back)})")
                            
                            p_fg_val = p_fg_full[debug_cell_in_batch_idx, original_prot_idx_to_debug].item()
                            p_bg_val = p_bg_full[debug_cell_in_batch_idx, original_prot_idx_to_debug].item()
                            print(f"      p_fg: {p_fg_val:.4f} (Is NaN: {np.isnan(p_fg_val)}) ([0,1]: {0 <= p_fg_val <= 1})")
                            print(f"      p_bg: {p_bg_val:.4f} (Is NaN: {np.isnan(p_bg_val)}) ([0,1]: {0 <= p_bg_val <= 1})")

                            fg_contrib_val = current_foreground_contribution[debug_cell_in_batch_idx, original_prot_idx_to_debug].item()
                            bg_contrib_val = current_background_contribution[debug_cell_in_batch_idx, original_prot_idx_to_debug].item()
                            sum_contrib_val = fg_contrib_val + bg_contrib_val
                            
                            print(f"      fg_contrib: {fg_contrib_val:.4f} (Is NaN: {np.isnan(fg_contrib_val)})")
                            print(f"      bg_contrib: {bg_contrib_val:.4f} (Is NaN: {np.isnan(bg_contrib_val)}) (>=0: {bg_contrib_val >= -1e-6})") 
                            print(f"      sum_contrib (total): {sum_contrib_val:.4f} (Is NaN: {np.isnan(sum_contrib_val)})")
                            print(f"      Is fg_contrib <= sum_contrib + 1e-5? {fg_contrib_val <= sum_contrib_val + 1e-5}")
                    # ============================================================ #
                    # ==== END DEBUG BLOCK FOR PROTEIN =========================== #
                    # ============================================================ #

                    # --- Continue Protein Calculation based on type ---
                    protein_expr_to_calc_full = torch.zeros_like(rate_fore_full)
                    if protein_expression_type == "foreground":
                        protein_expr_to_calc_full = current_foreground_contribution
                    elif protein_expression_type == "total":
                        protein_expr_to_calc_full = current_foreground_contribution + current_background_contribution
                    elif protein_expression_type == "corrected_total":
                        total_unscaled_expr_full = current_foreground_contribution + current_background_contribution
                        protein_expr_to_calc_full = F.normalize(total_unscaled_expr_full, p=1, dim=-1) 
                    
                    protein_expr_one_batch_masked = protein_expr_to_calc_full[:, protein_mask_bool]
                    protein_current_sample_sum_over_batches += protein_expr_one_batch_masked
                # --- End of inner loop over target_batch_code ---

                # Average over target batches and store results for sample i
                rna_minibatch_all_samples[i] = rna_current_sample_sum_over_batches / len(batch_categories_to_transform)
                protein_minibatch_all_samples[i] = protein_current_sample_sum_over_batches / len(batch_categories_to_transform)
            # --- End of outer loop over n_samples ---
            
            # Append results for this minibatch (all samples)
            all_rna_outputs_list.append(rna_minibatch_all_samples.cpu())
            all_protein_outputs_list.append(protein_minibatch_all_samples.cpu())

        final_rna_expr = torch.cat(all_rna_outputs_list, dim=1)
        final_protein_expr = torch.cat(all_protein_outputs_list, dim=1)

        if return_mean and n_samples > 1:
            final_rna_expr_np = torch.mean(final_rna_expr, dim=0).numpy()
            final_protein_expr_np = torch.mean(final_protein_expr, dim=0).numpy()
        elif n_samples == 1:
            final_rna_expr_np = final_rna_expr.squeeze(0).numpy()
            final_protein_expr_np = final_protein_expr.squeeze(0).numpy()
        else: # n_samples > 1 and not return_mean
            final_rna_expr_np = final_rna_expr.numpy()
            final_protein_expr_np = final_protein_expr.numpy()

        if return_numpy:
            return final_rna_expr_np, final_protein_expr_np
        else:
            obs_names_subset = adata_manager.adata.obs_names[indices]
            if final_rna_expr_np.ndim == 2:
                rna_df = pd.DataFrame(
                    final_rna_expr_np, index=obs_names_subset, columns=genes_to_return
                )
                protein_df = pd.DataFrame(
                    final_protein_expr_np, index=obs_names_subset, columns=proteins_to_return
                )
                return rna_df, protein_df
            else: # ndim == 3
                warnings.warn(
                    "Returning DataFrames for n_samples > 1 and return_mean=False is ambiguous. "
                    "Returning numpy arrays instead. Use return_numpy=True to silence this warning.", UserWarning)
                return final_rna_expr_np, final_protein_expr_np

    @torch.inference_mode()
    def get_protein_foreground_probability(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_samples: int = 1,
        transform_batch: str | int | Sequence[str | int] | None = None,
        protein_list: Sequence[str] | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
    ) -> pd.DataFrame | np.ndarray:
        """
        Returns the probability that the observed protein expression comes from
        the foreground distribution.

        This is denoted as :math:`1 - \pi_{nt}` in the TOTALVI paper,
        computed as :math:`\sigma(-m_{nt})` where :math:`m_{nt}` are the logits
        for the background probability.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to draw for each cell. For `n_samples > 1`, the mean
            or all an array of samples is returned (see `return_mean`).
        transform_batch
            Batch category to condition on for the generative model. If `None`, uses the batch
            cells were originally in. If a category is not present in the setup AnnData,
            an error will be raised. Can be a sequence, in which case outputs are averaged
            over the categories.
        protein_list
            Return probabilities for a subset of proteins.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            If `True` and `n_samples > 1`, returns the mean of the `n_samples` probabilities.
            Otherwise (if `n_samples > 1`), returns array of all `n_samples` probabilities.
        return_numpy
            If `True`, returns `np.ndarray` instead of `pd.DataFrame`.

        Returns
        -------
        Foreground probability estimates.
        If `n_samples == 1` or `return_mean == True`, shape is `(n_cells, n_proteins)`.
        Otherwise, shape is `(n_samples, n_cells, n_proteins)`. Returns `pd.DataFrame` unless
        `return_numpy` is True and output is 2D.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )

        # Prepare protein mask
        protein_state_reg = adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
        all_protein_names = np.array(protein_state_reg.column_names)
        if protein_list is None:
            protein_mask_bool = np.ones(len(all_protein_names), dtype=bool)
            proteins_to_return = all_protein_names
        else:
            protein_set = set(protein_list)
            protein_mask_bool = np.array([p in protein_set for p in all_protein_names])
            proteins_to_return = all_protein_names[protein_mask_bool]
            if len(proteins_to_return) == 0:
                raise ValueError("No proteins from `protein_list` were found in the model.")
        n_proteins_out = len(proteins_to_return)

        # Prepare transform_batch
        if transform_batch is None:
            batch_categories_to_transform = [None]
        else:
            batch_categories_to_transform = _get_batch_code_from_category(
                adata_manager, transform_batch
            )

        # List to store results
        all_fg_prob_list = []

        self.module.eval()
        for tensors in scdl:
            n_cells_in_batch = tensors[REGISTRY_KEYS.X_KEY].shape[0]

            fg_prob_minibatch_all_samples = torch.zeros(
                n_samples, n_cells_in_batch, n_proteins_out, device=self.device
            )

            for i in range(n_samples):
                inference_kwargs_iter = {"mc_samples": 1, "use_mean": False}
                inf_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(**inf_inputs, **inference_kwargs_iter)

                fg_prob_current_sample_sum_over_batches = torch.zeros(
                    n_cells_in_batch, n_proteins_out, device=self.device
                )

                for target_batch_code in batch_categories_to_transform:
                    gen_inputs = {
                        "z": inference_outputs["z"],
                        "library": inference_outputs["library"], # Still needed by generative signature
                        "logbeta": inference_outputs["logbeta"],
                        "batch_index": (
                            tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1).long()
                            if target_batch_code is None
                            else torch.full((n_cells_in_batch,),
                                             fill_value=target_batch_code, dtype=torch.long)
                        ).to(self.device)
                    }
                    generative_outputs = self.module.generative(**gen_inputs)

                    # Calculate foreground probability
                    pi_logit_background_full = generative_outputs["py_mixing"].squeeze(0)
                    p_fg_full = torch.sigmoid(-pi_logit_background_full) # Shape (B, P_all)

                    # Apply mask and accumulate
                    p_fg_masked = p_fg_full[:, protein_mask_bool] # Shape (B, P_out)
                    fg_prob_current_sample_sum_over_batches += p_fg_masked

                # Average over target batches for this sample
                fg_prob_minibatch_all_samples[i] = fg_prob_current_sample_sum_over_batches / len(batch_categories_to_transform)

            all_fg_prob_list.append(fg_prob_minibatch_all_samples.cpu())

        # Concatenate results across all minibatches
        final_fg_prob = torch.cat(all_fg_prob_list, dim=1) # Shape (n_samples, n_total_indices, n_proteins_out)

        # Handle return_mean logic
        if return_mean and n_samples > 1:
            final_fg_prob_np = torch.mean(final_fg_prob, dim=0).numpy() # Shape (n_cells, n_proteins_out)
        elif n_samples == 1:
            final_fg_prob_np = final_fg_prob.squeeze(0).numpy() # Shape (n_cells, n_proteins_out)
        else: # n_samples > 1 and not return_mean
            final_fg_prob_np = final_fg_prob.numpy() # Shape (n_samples, n_cells, n_proteins_out)

        # Handle return_numpy logic
        if return_numpy:
            return final_fg_prob_np
        else:
            obs_names_subset = adata_manager.adata.obs_names[indices]
            if final_fg_prob_np.ndim == 2: # Return DataFrame only if 2D
                return pd.DataFrame(
                    final_fg_prob_np, index=obs_names_subset, columns=proteins_to_return
                )
            else: # ndim == 3
                warnings.warn(
                    "Returning DataFrames for n_samples > 1 and return_mean=False is ambiguous. "
                    "Returning numpy arrays instead. Use return_numpy=True to silence this warning.", UserWarning)
                return final_fg_prob_np
            
    @torch.inference_mode()
    def posterior_predictive_sample(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
        gene_list: Sequence[str] | None = None,
        protein_list: Sequence[str] | None = None,
        transform_batch: str | int | Sequence[str | int] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate samples from the posterior predictive distribution.

        This is denoted :math:`p(\\hat{x}, \\hat{y} \\mid x, y)`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of required posterior predictive samples for each cell.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        gene_list
            Names of genes of interest. If `None`, all genes are sampled.
        protein_list
            Names of proteins of interest. If `None`, all proteins are sampled.
        transform_batch
            Batch category (or categories) to condition on for generation. If `None`, uses
            the original batch associated with each cell.

        Returns
        -------
        Dictionary with keys "rna" and "protein". The values are `np.ndarray` objects:
        - If `n_samples == 1`, shape is `(n_cells, n_features)`.
        - If `n_samples > 1`, shape is `(n_cells, n_features, n_samples)`.
          Note: This differs from some methods where samples are the first dimension.
        """

        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )

        # --- Prepare gene and protein masks using BOOLEAN logic ---
        all_gene_names = _get_var_names_from_manager(adata_manager) # np.ndarray
        if gene_list is None:
            gene_mask_bool = np.ones(len(all_gene_names), dtype=bool)
        else:
            gene_set = set(gene_list)
            gene_mask_bool = np.array([g in gene_set for g in all_gene_names])
            if not np.any(gene_mask_bool): # Check if any genes matched
                 raise ValueError("No genes from `gene_list` were found in the model.")
        # No need for n_genes_out explicitly here, shape is inferred

        protein_state_reg = adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
        all_protein_names = np.array(protein_state_reg.column_names)
        if protein_list is None:
            protein_mask_bool = np.ones(len(all_protein_names), dtype=bool)
        else:
            protein_set = set(protein_list)
            protein_mask_bool = np.array([p in protein_set for p in all_protein_names])
            if not np.any(protein_mask_bool): # Check if any proteins matched
                 raise ValueError("No proteins from `protein_list` were found in the model.")
        # No need for n_proteins_out explicitly here

        # --- Prepare transform_batch ---
        target_batch_code = None
        if transform_batch is not None:
            batch_codes = _get_batch_code_from_category(adata_manager, transform_batch)
            if len(batch_codes) > 1:
                warnings.warn("Multiple batches provided to `transform_batch` in posterior predictive sampling. Using the first batch provided.", UserWarning)
            target_batch_code = batch_codes[0]

        rna_samples_list = []
        protein_samples_list = []

        self.module.eval()
        for tensors in scdl:
            n_cells_in_batch = tensors[REGISTRY_KEYS.X_KEY].shape[0]
            
            inference_kwargs = {"mc_samples": n_samples, "use_mean": False}
            inf_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inf_inputs, **inference_kwargs)

            z = inference_outputs["z"]
            library = inference_outputs["library"]
            logbeta = inference_outputs["logbeta"]
            
            # Ensure correct dimensions based on n_samples
            if n_samples == 1 and z.ndim == 3:
                 z = z.squeeze(0)
                 library = library.squeeze(0)
                 logbeta = logbeta.squeeze(0)
            elif n_samples > 1 and z.ndim == 2:
                 z = z.unsqueeze(0)
                 library = library.unsqueeze(0)
                 logbeta = logbeta.unsqueeze(0)

            if target_batch_code is None:
                gen_batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1).long().to(self.device)
            else:
                gen_batch_index = torch.full((n_cells_in_batch,), 
                                             fill_value=target_batch_code, dtype=torch.long, device=self.device)
            
            gen_inputs = {
                "z": z, "library": library, "logbeta": logbeta, "batch_index": gen_batch_index
            }
            generative_outputs = self.module.generative(**gen_inputs)

            # Sample from distributions
            px_rate = generative_outputs["px_rate"]
            px_r = generative_outputs["px_r"]
            rna_dist = NegativeBinomial(mu=px_rate, theta=px_r)
            rna_sample_batch_full = rna_dist.sample().int() 

            py_rate_back = generative_outputs["py_rate_back"]
            py_rate_fore = generative_outputs["py_rate_fore"]
            py_mixing = generative_outputs["py_mixing"]
            py_r = generative_outputs["py_r"]
            protein_dist = NegativeBinomialMixture(
                mu1=py_rate_back, mu2=py_rate_fore, theta1=py_r, mixture_logits=py_mixing
            )
            protein_sample_batch_full = protein_dist.sample().int()

            # --- Subset features using BOOLEAN masks ---
            # Note: PyTorch boolean indexing works along the specified dimension
            # If rna_sample_batch_full is (S, B, G_all), [..., gene_mask_bool] indexes the LAST dim.
            # If rna_sample_batch_full is (B, G_all), [:, gene_mask_bool] indexes the LAST dim.
            # If n_samples=1, the shape is (B, G_all/P_all) after squeeze.
            # If n_samples>1, the shape is (S, B, G_all/P_all).
            if n_samples == 1:
                rna_sample_batch = rna_sample_batch_full[:, gene_mask_bool]
                protein_sample_batch = protein_sample_batch_full[:, protein_mask_bool]
            else: # n_samples > 1
                rna_sample_batch = rna_sample_batch_full[..., gene_mask_bool] # Ellipsis handles (S, B) dimensions
                protein_sample_batch = protein_sample_batch_full[..., protein_mask_bool]

            # Store samples
            if n_samples == 1:
                 rna_samples_list.append(rna_sample_batch.cpu().numpy())
                 protein_samples_list.append(protein_sample_batch.cpu().numpy())
            else:
                 rna_samples_list.append(rna_sample_batch.permute(1, 2, 0).cpu().numpy()) # (S,B,F) -> (B,F,S)
                 protein_samples_list.append(protein_sample_batch.permute(1, 2, 0).cpu().numpy())

        final_rna_samples = np.concatenate(rna_samples_list, axis=0)
        final_protein_samples = np.concatenate(protein_samples_list, axis=0)

        return {"rna": final_rna_samples, "protein": final_protein_samples}

    @torch.inference_mode()
    def get_protein_background_mean(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: str | int | Sequence[str | int] | None = None,
        protein_list: Sequence[str] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True, # If n_samples > 1 (for z), return mean of calculated background rates
        return_numpy: bool = False,
    ) -> pd.DataFrame | np.ndarray:
        """
        Returns the mean of the estimated protein background expression rate.

        This is :math:`\mathbb{E}[e^{\\beta_{nt}}] = e^{\mu_{\\beta_{nt}} + \sigma^2_{\\beta_{nt}}/2}`
        where :math:`\\beta_{nt} \sim \mathcal{N}(\mu_{\\beta_{nt}}, \sigma^2_{\\beta_{nt}})`,
        and :math:`\mu_{\\beta_{nt}}, \sigma^2_{\\beta_{nt}}` are the outputs of the
        BackgroundProteinEncoder module conditioned on latent variable `z` and `batch_index`.
        The expectation is taken over `n_samples` draws of `z`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch category (or categories) to condition on for the `BackgroundProteinEncoder`.
            If `None`, uses the original batch associated with each cell. If a sequence is given,
            the results are averaged over these batch conditions.
        protein_list
            Return background mean for a subset of proteins.
        n_samples
            Number of posterior samples of `z` to average over.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            If `True` and `n_samples > 1`, returns the mean of the `n_samples` computed background rates.
            Otherwise (if `n_samples > 1`), returns array of all `n_samples` background rates.
        return_numpy
            If `True`, returns `np.ndarray` instead of `pd.DataFrame`.

        Returns
        -------
        Protein background mean estimates.
        If `n_samples == 1` or `return_mean == True`, shape is `(n_cells, n_proteins)`.
        Otherwise, shape is `(n_samples, n_cells, n_proteins)`. Returns `pd.DataFrame` unless
        `return_numpy` is True and output is 2D.
        """
        # --- Input Validation ---
        if n_samples <= 0: # n_samples here is for z, should be positive
            raise ValueError("n_samples must be a positive integer.")
        # --- End Input Validation ---

        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )

        # Prepare protein mask
        protein_state_reg = adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
        all_protein_names = np.array(protein_state_reg.column_names)
        if protein_list is None:
            protein_mask_bool = np.ones(len(all_protein_names), dtype=bool)
            proteins_to_return = all_protein_names
        else:
            protein_set = set(protein_list)
            protein_mask_bool = np.array([p in protein_set for p in all_protein_names])
            proteins_to_return = all_protein_names[protein_mask_bool]
            if len(proteins_to_return) == 0:
                raise ValueError("No proteins from `protein_list` were found in the model.")
        n_proteins_out = len(proteins_to_return)

        # Prepare transform_batch
        if transform_batch is None:
            batch_categories_to_transform = [None]
        else:
            batch_categories_to_transform = _get_batch_code_from_category(
                adata_manager, transform_batch
            )
        
        all_bg_mean_list = []

        self.module.eval()
        for tensors in scdl:
            n_cells_in_batch = tensors[REGISTRY_KEYS.X_KEY].shape[0]

            # Accumulator for all posterior samples for this minibatch
            bg_mean_minibatch_all_samples = torch.zeros(
                n_samples, n_cells_in_batch, n_proteins_out, device=self.device
            )

            for s_idx in range(n_samples):
                # 1. Inference to get one sample of z
                inference_kwargs_iter = {"mc_samples": 1, "use_mean": False}
                inf_inputs = self.module._get_inference_input(tensors)
                # Ensure sample_index and batch_index are long from _get_inference_input
                inference_outputs = self.module.inference(**inf_inputs, **inference_kwargs_iter)
                
                z_current_sample = inference_outputs["z"] # Shape (1, B, D) or (B,D)
                if z_current_sample.ndim == 3 and z_current_sample.shape[0] == 1:
                    z_current_sample = z_current_sample.squeeze(0) # Now (B, D)

                # Accumulator for averaging over transform_batch conditions for this z_current_sample
                bg_mean_current_z_sum_over_batches = torch.zeros(
                    n_cells_in_batch, n_proteins_out, device=self.device
                )
                
                for target_batch_code in batch_categories_to_transform:
                    # 2. Determine batch_index for BackgroundProteinEncoder
                    if target_batch_code is None:
                        # Use original batch from data, already long from _get_inference_input
                        batch_index_for_encoder = inf_inputs["batch_index"] 
                    else:
                        batch_index_for_encoder = torch.full(
                            (n_cells_in_batch,), fill_value=target_batch_code, 
                            dtype=torch.long, device=self.device
                        )
                    
                    # 3. Get qbeta from BackgroundProteinEncoder
                    qbeta = self.module.background_encoder(z_current_sample, batch_index_for_encoder)
                    
                    # 4. Calculate E[exp(logbeta)] = exp(mu + sigma^2/2)
                    logbeta_loc = qbeta.loc     # Shape (B, P_all)
                    logbeta_scale = qbeta.scale # Shape (B, P_all)
                    current_bg_rate_mean_full = torch.exp(logbeta_loc + (logbeta_scale.pow(2)) / 2) # Shape (B, P_all)

                    # 5. Apply protein mask and accumulate
                    current_bg_rate_mean_masked = current_bg_rate_mean_full[:, protein_mask_bool] # Shape (B, P_out)
                    bg_mean_current_z_sum_over_batches += current_bg_rate_mean_masked
                
                # Average over transform_batch conditions and store for this sample s_idx
                bg_mean_minibatch_all_samples[s_idx] = bg_mean_current_z_sum_over_batches / len(batch_categories_to_transform)

            all_bg_mean_list.append(bg_mean_minibatch_all_samples.cpu())

        # Concatenate results across all minibatches
        final_bg_mean = torch.cat(all_bg_mean_list, dim=1) # Shape (n_samples, n_total_indices, n_proteins_out)

        # Handle return_mean logic
        if return_mean and n_samples > 1:
            final_bg_mean_np = torch.mean(final_bg_mean, dim=0).numpy()
        elif n_samples == 1:
            final_bg_mean_np = final_bg_mean.squeeze(0).numpy()
        else: # n_samples > 1 and not return_mean
            final_bg_mean_np = final_bg_mean.numpy()

        # Handle return_numpy logic
        if return_numpy:
            return final_bg_mean_np
        else:
            obs_names_subset = adata_manager.adata.obs_names[indices]
            if final_bg_mean_np.ndim == 2:
                return pd.DataFrame(
                    final_bg_mean_np, index=obs_names_subset, columns=proteins_to_return
                )
            else: 
                warnings.warn(
                    "Returning DataFrames for n_samples > 1 and return_mean=False is ambiguous for this method. "
                    "Returning numpy array instead. Use return_numpy=True to silence this warning.", UserWarning)
                return final_bg_mean_np
            
    def update_sample_info(self, adata: AnnData | None = None):
        """
        Update the sample information stored in the model, creating one row per unique sample.

        This method re-extracts data from `adata.obs`. For each unique sample category
        (identified by the registered `sample_key`), it takes the row of covariates
        from the *first cell encountered* for that sample. The resulting `self.sample_info`
        DataFrame is indexed by the integer-encoded sample categories.

        If `sample_key` was not provided during `setup_anndata` (i.e., `sample_key=None`),
        a warning is issued, and a minimal `sample_info` based on the default single
        sample category is created.

        Parameters
        ----------
        adata
            AnnData object to update the sample info from. If `None`, defaults to
            `self.adata`. The `adata` must have been set up with the model.
        """
        if adata is None:
            adata = self.adata
        
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        # Get the key in adata.obs that stores the scvi-tools integer-encoded sample IDs
        # This is typically "_scvi_sample"
        manager_encoded_sample_key = adata_manager.data_registry[REGISTRY_KEYS.SAMPLE_KEY].attr_key

        # Check if a user-provided sample_key was given during setup
        user_original_sample_key = adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).original_key

        print(f"[DEBUG update_sample_info] User's original sample_key from setup: {user_original_sample_key}") # DEBUG
        print(f"[DEBUG update_sample_info] Manager's encoded sample key in obs: {manager_encoded_sample_key}") # DEBUG

        if user_original_sample_key is None:
            # This case means sample_key=None was used in setup_anndata.
            # AnnDataManager creates a default _scvi_sample column (all 0s).
            print("[DEBUG update_sample_info] No original sample key provided at setup. Creating minimal sample_info.") # DEBUG
            warnings.warn(
                "Original sample key was not provided during setup_anndata. "
                "Sample info will be based on the default single sample category.", 
                UserWarning
            )
            if manager_encoded_sample_key in adata.obs:
                 unique_encoded_samples = pd.Series(adata.obs[manager_encoded_sample_key].unique()).sort_values()
                 self.sample_info = pd.DataFrame(index=unique_encoded_samples)
                 # Use the manager's encoded key name for the index name for consistency
                 self.sample_info.index.name = manager_encoded_sample_key 
            else: 
                self.sample_info = pd.DataFrame() 
            logger.info(f"Sample info updated (default single sample). Found {len(self.sample_info)} unique samples.")
            return

        # Proceed if a user_original_sample_key was provided (it might be the same as manager_encoded_sample_key
        # if the user's column was already named e.g. "_scvi_sample", but that's fine).
        print(f"[DEBUG update_sample_info] Using '{user_original_sample_key}' for identifying unique samples and '{manager_encoded_sample_key}' for indexing.") # DEBUG

        obs_df_copy = adata.obs.copy()
        
        # Drop duplicates based on the *integer-encoded* sample key to ensure one row per *encoded* sample.
        # This is what MRVI does with `obs_df[self.scvi_sample_col].duplicated("first")`.
        # We use `~` to keep the first occurrences.
        sample_info_df = obs_df_copy.loc[~obs_df_copy[manager_encoded_sample_key].duplicated(keep="first")]
        
        # Set the index to be the scvi-tools integer-encoded sample category and sort
        sample_info_df = sample_info_df.set_index(manager_encoded_sample_key)
        sample_info_df = sample_info_df.sort_index()
        
        self.sample_info = sample_info_df
        logger.info(f"Sample info updated. Found {len(self.sample_info)} unique samples. "
                    f"Columns: {list(self.sample_info.columns)}")
        
    @torch.inference_mode()
    def get_local_sample_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_u_samples: int = 1, # Number of u samples to average over / return
        give_u_mean: bool = True, # Whether to use mean of u or sample u
        batch_size: int | None = None,
    ) -> xr.DataArray:
        """
        Computes the local sample representation for each cell.

        For each cell, this method generates its latent representation `u`, and then
        for that `u` (or samples of `u`), it generates counterfactual second-level
        latent representations `z` as if the cell belonged to every sample category
        defined in the model.

        Parameters
        ----------
        adata
            AnnData object. If `None`, uses `self.adata`.
        indices
            Indices of cells in `adata` to use. If `None`, all cells.
        n_u_samples
            Number of Monte Carlo samples of `u` to generate for each cell if `give_u_mean` is False.
            If `give_u_mean` is True, this is ignored for `u` but the resulting `z` representations
            for each counterfactual sample will still be based on one `z` sample (or mean if `qz.use_map=True`).
        give_u_mean
            If `True`, use the mean of `q(u)` distribution. If `False`, draw `n_u_samples` from `q(u)`.
        batch_size
            Minibatch size for data loading.

        Returns
        -------
        xr.DataArray
            An xarray DataArray of local sample representations.
            Shape is `(n_cells, n_samples_total, n_latent_z)` if `give_u_mean` is True or `n_u_samples` is 1.
            Shape is `(n_u_samples, n_cells, n_samples_total, n_latent_z)` if `give_u_mean` is False and `n_u_samples` > 1.
            `n_samples_total` is the total number of unique sample categories in the model.
            Coordinates include 'cell_name' and 'sample_cat' (original sample category names).
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if indices is None:
            indices = np.arange(adata.n_obs)
        
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)

        n_model_samples = self.summary_stats.n_sample
        original_sample_categories = adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        
        # Get the latent dimension of z from the module
        n_latent_z_dim = self.module.n_latent # <--- USE MODULE'S ATTRIBUTE

        effective_n_u_samples = 1 if give_u_mean else n_u_samples
        all_local_zs_list = []

        self.module.eval()
        for tensors in scdl:
            n_cells_in_batch = tensors[REGISTRY_KEYS.X_KEY].shape[0]
            inf_inputs = self.module._get_inference_input(tensors)

            qu = self.module.qu(inf_inputs["x"], inf_inputs["y"], inf_inputs["sample_index"])
            if give_u_mean:
                u_for_qz = qu.mean.unsqueeze(0)
            else:
                u_for_qz = qu.rsample(sample_shape=(n_u_samples,))
            
            batch_local_zs = torch.zeros(
                effective_n_u_samples, n_cells_in_batch, n_model_samples, n_latent_z_dim, # <--- Use n_latent_z_dim
                device=self.device
            )

            for u_sample_idx in range(effective_n_u_samples):
                current_u = u_for_qz[u_sample_idx]
                for target_sample_code in range(n_model_samples):
                    target_sample_tensor = torch.full(
                        (n_cells_in_batch,), fill_value=target_sample_code, 
                        dtype=torch.long, device=self.device
                    )
                    z_base, eps_val_or_params = self.module.qz(current_u, target_sample_tensor)
                    current_eps = None
                    if self.module.qz.use_map:
                        current_eps = eps_val_or_params
                    else:
                        qeps_mean, qeps_scale_chunk = eps_val_or_params
                        qeps_scale = F.softplus(qeps_scale_chunk) + 1e-5
                        current_eps = qeps_mean
                    z_counterfactual = z_base + current_eps
                    batch_local_zs[u_sample_idx, :, target_sample_code, :] = z_counterfactual
            
            all_local_zs_list.append(batch_local_zs.cpu())

        final_local_zs = torch.cat(all_local_zs_list, dim=1)

        obs_names_subset = adata_manager.adata.obs_names[indices]
        
        if effective_n_u_samples == 1:
            final_local_zs = final_local_zs.squeeze(0)
            coords = {
                "cell_name": obs_names_subset,
                "sample_cat": original_sample_categories,
                "latent_dim": np.arange(n_latent_z_dim) # <--- Use n_latent_z_dim
            }
            dims = ["cell_name", "sample_cat", "latent_dim"]
        else:
             coords = {
                "u_sample_idx": np.arange(effective_n_u_samples),
                "cell_name": obs_names_subset,
                "sample_cat": original_sample_categories,
                "latent_dim": np.arange(n_latent_z_dim) # <--- Use n_latent_z_dim
            }
             dims = ["u_sample_idx", "cell_name", "sample_cat", "latent_dim"]
            
        return xr.DataArray(
            final_local_zs.numpy(),
            coords=coords,
            dims=dims,
            name="local_sample_z_representations"
        )
    
    @torch.inference_mode()
    def get_local_sample_distances(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_u_samples: int = 1,
        give_u_mean: bool = True,
        batch_size: int | None = None,
        norm: Literal["l1", "l2", "linf"] = "l2",
        # groupby, keep_cell from MRVI are more complex due to xarray; start without
    ) -> xr.DataArray: # Returns cell-specific (sample x sample) distance matrices
        """
        Computes cell-specific distances between counterfactual sample representations.

        For each cell, this method first generates its local sample representations
        (counterfactual `z` for each possible sample category). Then, it computes
        the pairwise distances between these representations.

        Parameters
        ----------
        adata
            AnnData object. If `None`, uses `self.adata`.
        indices
            Indices of cells in `adata` to use. If `None`, all cells.
        n_u_samples
            Number of Monte Carlo samples of `u` to average over.
        give_u_mean
            If `True`, use the mean of `q(u)`. If `False`, draw `n_u_samples` from `q(u)`
            and average the resulting distance matrices.
        batch_size
            Minibatch size for data loading.
        norm
            Norm to use for computing distances between z vectors.
            One of "l1", "l2", "linf".

        Returns
        -------
        xr.DataArray
            An xarray DataArray of cell-specific sample distance matrices.
            Shape is `(n_cells, n_samples_total, n_samples_total)`.
            Coordinates include 'cell_name', 'sample_cat_x', 'sample_cat_y'.
        """

        if n_u_samples <= 0:
            raise ValueError("n_u_samples must be a positive integer.")
        if norm not in ["l1", "l2", "linf"]:
            raise ValueError(f"Unsupported norm: {norm}. Must be 'l1', 'l2', or 'linf'.")

        self._check_if_trained(warn=False)
        adata_manager = self.get_anndata_manager(adata if adata is not None else self.adata, required=True)

        # 1. Get local sample representations (counterfactual z's)
        # local_zs shape: (eff_n_u_samples, total_cells, n_model_samples, n_latent_z)
        # or (total_cells, n_model_samples, n_latent_z) if eff_n_u_samples=1
        local_zs_xarray = self.get_local_sample_representation(
            adata=adata,
            indices=indices,
            n_u_samples=n_u_samples,
            give_u_mean=give_u_mean,
            batch_size=batch_size
        )
        local_zs = torch.from_numpy(local_zs_xarray.data).to(self.device)

        # If local_zs has u_sample dimension, average over it before distance calc,
        # or calculate distances per u_sample and then average distances.
        # MRVI averages representations if give_u_mean=True and n_u_samples=1 for u.
        # If multiple u_samples, it implies we want to average the final distances.
        
        # Let's calculate distances for each u_sample, then average the distance matrices.
        # Expand dims if local_zs is (total_cells, n_model_samples, n_latent_z)
        if local_zs.ndim == 3:
            local_zs = local_zs.unsqueeze(0) # (1, total_cells, n_model_samples, n_latent_z)
        
        n_eff_u_samples, n_total_cells, n_model_samples, _ = local_zs.shape
        
        all_dist_matrices_per_u_sample = []

        for u_idx in range(n_eff_u_samples):
            current_u_zs = local_zs[u_idx] # Shape (total_cells, n_model_samples, n_latent_z)
            
            # Calculate pairwise distances for each cell
            # Output should be (total_cells, n_model_samples, n_model_samples)
            cell_specific_distances = torch.zeros(
                (n_total_cells, n_model_samples, n_model_samples), device=self.device
            )
            
            for cell_i in range(n_total_cells):
                zs_for_cell_i = current_u_zs[cell_i] # Shape (n_model_samples, n_latent_z)
                
                # Expand for broadcasting:
                # zs_for_cell_i_x: (n_model_samples, 1, n_latent_z)
                # zs_for_cell_i_y: (1, n_model_samples, n_latent_z)
                zs_x = zs_for_cell_i.unsqueeze(1)
                zs_y = zs_for_cell_i.unsqueeze(0)
                
                delta_mat = zs_x - zs_y # Shape (n_model_samples, n_model_samples, n_latent_z)
                
                if norm == "l2":
                    dist_pairs = torch.norm(delta_mat, p=2, dim=-1)
                elif norm == "l1":
                    dist_pairs = torch.norm(delta_mat, p=1, dim=-1)
                elif norm == "linf":
                    dist_pairs = torch.norm(delta_mat, p=float('inf'), dim=-1)
                else:
                    raise ValueError(f"Unsupported norm: {norm}")
                cell_specific_distances[cell_i] = dist_pairs
            all_dist_matrices_per_u_sample.append(cell_specific_distances)
        
        # Stack and average over u_samples dimension
        avg_dist_matrices = torch.stack(all_dist_matrices_per_u_sample).mean(dim=0)
        # avg_dist_matrices shape: (total_cells, n_model_samples, n_model_samples)

        # Prepare for xarray DataArray
        if indices is None:
            indices = np.arange(self.adata.n_obs if adata is None else adata.n_obs)
        obs_names_subset = adata_manager.adata.obs_names[indices]
        original_sample_categories = adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        
        return xr.DataArray(
            avg_dist_matrices.cpu().numpy(),
            coords={
                "cell_name": obs_names_subset,
                "sample_cat_x": original_sample_categories,
                "sample_cat_y": original_sample_categories,
            },
            dims=["cell_name", "sample_cat_x", "sample_cat_y"],
            name="local_sample_distances"
        )
    
    @torch.inference_mode()
    def get_aggregated_posterior_u(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None, # Cells to contribute to the aggregate posterior
        sample_label: str | int | None = None, # If specified, indices are further filtered by this sample_label
        n_u_draws_per_cell: int = 1, # How many u to draw per cell to form the mixture components
        batch_size: int | None = None,
    ) -> MixtureSameFamily:
        """
        Computes the aggregated posterior over the `u` latent representations for a specified sample.

        This forms a mixture distribution where each component is q(u | x_i, y_i, s_i)
        for cells `i` belonging to the specified `sample_label` (or all cells in `indices`).

        Parameters
        ----------
        adata
            AnnData object. If `None`, uses `self.adata`.
        indices
            Indices of cells in `adata` to use for forming the aggregate posterior.
            If `None`, all cells are used.
        sample_label
            If provided, `indices` will be further filtered to include only cells
            belonging to this specific sample category (original name or integer code).
        n_u_draws_per_cell
            For each cell contributing to the posterior, how many samples of `u` from
            `q(u|x,y,s)` should be included as components in the mixture.
            If 1 (default), uses the mean of `q(u)`.
        batch_size
            Minibatch size for data loading.

        Returns
        -------
        torch.distributions.MixtureSameFamily
            A mixture of Normal distributions representing p(u | Sample).
        """
        self._check_if_trained(warn=False)
        adata_orig = adata
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if indices is None:
            indices = np.arange(adata.n_obs)
        
        if sample_label is not None:
            # Convert sample_label to its integer code if it's a string
            sample_state_reg = adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY)
            all_sample_cats = sample_state_reg.categorical_mapping
            
            target_sample_code = None
            if isinstance(sample_label, str):
                if sample_label in all_sample_cats:
                    target_sample_code = list(all_sample_cats).index(sample_label)
                else:
                    raise ValueError(f"Sample label '{sample_label}' not found in model setup.")
            elif isinstance(sample_label, int):
                if 0 <= sample_label < self.summary_stats.n_sample:
                    target_sample_code = sample_label
                else:
                    raise ValueError(f"Sample code {sample_label} is out of bounds.")
            else:
                raise TypeError("sample_label must be str or int.")

            # Filter indices by this target_sample_code
            cell_sample_codes = adata_manager.get_from_registry(REGISTRY_KEYS.SAMPLE_KEY)[indices].ravel()
            indices = indices[cell_sample_codes == target_sample_code]
            if len(indices) == 0:
                raise ValueError(f"No cells found for sample_label '{sample_label}' within the provided indices.")

        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)

        all_u_locs = []
        all_u_scales = []
        all_u_samples_for_mixture = [] # if n_u_draws_per_cell > 1

        self.module.eval()
        for tensors in scdl:
            inf_inputs = self.module._get_inference_input(tensors)
            qu = self.module.qu(inf_inputs["x"], inf_inputs["y"], inf_inputs["sample_index"])
            
            if n_u_draws_per_cell == 1: # Use mean
                all_u_locs.append(qu.loc.detach().cpu())
                all_u_scales.append(qu.scale.detach().cpu())
            else: # Draw samples
                u_samps = qu.rsample(sample_shape=(n_u_draws_per_cell,)).detach().cpu() # (n_draws, B, D_u)
                # Reshape to (n_draws * B, D_u) for Mixture components
                u_samps_flat = u_samps.permute(1,0,2).reshape(-1, u_samps.shape[-1])
                all_u_samples_for_mixture.append(u_samps_flat)
                # For sampled components, assume a very small fixed scale for the Normal components
                # This makes it a mixture of dirac-like deltas centered at the samples
                # Alternatively, one could fit a GMM to these samples, but that's more complex.
                # A simpler approach is a mixture of normals with tiny variance.
                small_scale = torch.full_like(u_samps_flat, 1e-3) # Small fixed scale
                all_u_locs.append(u_samps_flat)
                all_u_scales.append(small_scale)


        if not all_u_locs:
            raise ValueError("No cells found to form the aggregated posterior.")

        component_locs = torch.cat(all_u_locs, dim=0)
        component_scales = torch.cat(all_u_scales, dim=0)
        
        n_components = component_locs.shape[0]
        if n_components == 0:
             raise ValueError("No components for mixture model (no cells selected).")

        mix_logits = torch.ones(n_components, device=component_locs.device)
        mixture_dist = Categorical(logits=mix_logits)
        
        # Create a Normal distribution where the last dimension (D_u) is the event shape
        base_component_dist = Normal(loc=component_locs, scale=component_scales)
        independent_component_dist = Independent(base_component_dist, 1) # <--- CORRECTED HERE

        return MixtureSameFamily(mixture_dist, independent_component_dist)

    @torch.inference_mode()
    def get_log_affinity_u(
        self,
        adata: AnnData | None = None, # Query adata
        indices: Sequence[int] | None = None,
        sample_subset_for_posteriors: Sequence[str] | None = None, # Original sample names to build posteriors for
        n_u_draws_per_cell_for_posterior: int = 1,
        use_mean_u_for_query_cells: bool = True,
        n_u_samples_for_query_cells: int = 1, 
        batch_size: int | None = None,
        min_cells_for_posterior: int = 1, # Default to 1, meaning posterior is built if at least 1 cell
    ) -> pd.DataFrame:
        """
        Computes the log probability of each cell's u representation under the
        aggregated posterior p(u|S=s) for different sample categories S.

        This can be seen as a measure of affinity of each cell to each sample category's
        characteristic u-space distribution.

        Parameters
        ----------
        adata
            AnnData object for which to compute affinities. If `None`, uses `self.adata`.
        indices
            Indices of cells in `adata` to compute affinities for. If `None`, all cells.
        sample_subset_for_posteriors
            A list of original sample category names for which to build aggregated posteriors.
            If `None`, posteriors are built for all sample categories in `self.sample_info`
            that meet `min_cells_for_posterior`.
        n_u_draws_per_cell_for_posterior
            Number of `u` samples per cell used to construct each sample's aggregated posterior.
        use_mean_u_for_query_cells
            For the query cells, whether to use the mean of their `q(u)` or sample from it.
        n_u_samples_for_query_cells
            If `use_mean_u_for_query_cells` is False, number of `u` samples to draw per query cell.
            The log probabilities will be averaged over these samples.
        batch_size
            Minibatch size.
        min_cells_for_posterior
            Minimum number of cells required for a sample category to be included
            when forming an aggregated posterior.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape (n_query_cells, n_sample_categories_for_posterior)
            containing log p(u_cell | S=s). Indexed by query cell names, columns by
            original sample category names.
        """
        self._check_if_trained(warn=False)
        adata_query = self._validate_anndata(adata)
        # Manager for the query adata (used for getting query cell u's and obs_names)
        adata_manager_query = self.get_anndata_manager(adata_query, required=True) 

        # sample_info and manager for building posteriors should come from self.adata
        if not hasattr(self, "sample_info") or self.sample_info is None or self.sample_info.empty:
            self.update_sample_info(self.adata) 
        if self.sample_info.empty:
            raise ValueError("self.sample_info is empty.")

        # Determine which sample categories to build posteriors for, using self.adata_manager for mappings
        target_sample_info_for_posteriors = self.sample_info.copy() # Based on self.adata
        original_sample_key_col_name = self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).original_key
        
        if sample_subset_for_posteriors is not None:
            if original_sample_key_col_name and original_sample_key_col_name in target_sample_info_for_posteriors.columns:
                target_sample_info_for_posteriors = target_sample_info_for_posteriors[
                    target_sample_info_for_posteriors[original_sample_key_col_name].isin(sample_subset_for_posteriors)
                ]
            else:
                all_sample_cats_map = {
                    name: code for code, name in enumerate(self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping)
                }
                codes_to_keep = [all_sample_cats_map[s] for s in sample_subset_for_posteriors if s in all_sample_cats_map]
                target_sample_info_for_posteriors = target_sample_info_for_posteriors[target_sample_info_for_posteriors.index.isin(codes_to_keep)]
            if target_sample_info_for_posteriors.empty:
                raise ValueError("No samples for posterior construction after applying `sample_subset_for_posteriors`.")

        # Build aggregated posteriors for the selected sample categories
        aggregated_posteriors = {}
        skipped_samples_log = []
        
        # Get mapping from model's sample codes to original names for columns
        model_sample_code_to_name_map = {
            code: name for code, name in enumerate(self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping)
        }
        manager_enc_sample_key_for_self_adata = self.adata_manager.data_registry[REGISTRY_KEYS.SAMPLE_KEY].attr_key


        for s_code in target_sample_info_for_posteriors.index: # s_code is integer code
            s_name = model_sample_code_to_name_map.get(s_code, f"unknown_sample_code_{s_code}")
            
            # Check cell count for this sample_code in self.adata
            cells_in_sample_for_posterior = np.sum(self.adata.obs[manager_enc_sample_key_for_self_adata] == s_code)
            if cells_in_sample_for_posterior < min_cells_for_posterior:
                logger.warning(f"Sample '{s_name}' (code {s_code}) has {cells_in_sample_for_posterior} cells in self.adata, "
                               f"which is < min_cells_for_posterior ({min_cells_for_posterior}). Skipping posterior construction.")
                skipped_samples_log.append(s_name)
                continue
            
            try:
                # Use self.adata (model's reference adata) to build these posteriors
                agg_post = self.get_aggregated_posterior_u(
                    adata=self.adata, 
                    sample_label=s_code, # Pass integer code directly
                    n_u_draws_per_cell=n_u_draws_per_cell_for_posterior,
                    batch_size=batch_size,
                )
                if agg_post.mixture_distribution.logits.numel() > 0: # Check if components exist
                     aggregated_posteriors[s_name] = agg_post
                else: # Should be caught by cell count check above, but defensive
                    logger.warning(f"Aggregated posterior for sample '{s_name}' (code {s_code}) resulted in no components. Skipping.")
                    skipped_samples_log.append(s_name)
            except ValueError as e: 
                logger.warning(f"Could not build aggregated posterior for sample '{s_name}' (code {s_code}): {e}. Skipping.")
                skipped_samples_log.append(s_name)

        if not aggregated_posteriors:
            raise ValueError("No aggregated posteriors could be successfully built for any specified sample category.")
        
        active_posterior_sample_names = list(aggregated_posteriors.keys())

        # Get U representations for query cells
        if indices is None:
            indices = np.arange(adata_query.n_obs)
        
        u_query_cells_list = []
        query_scdl = self._make_data_loader(adata=adata_query, indices=indices, batch_size=batch_size, shuffle=False)
        self.module.eval() # Ensure module is in eval mode
        for tensors in query_scdl:
            inf_inputs = self.module._get_inference_input(tensors)
            qu = self.module.qu(inf_inputs["x"], inf_inputs["y"], inf_inputs["sample_index"])
            if use_mean_u_for_query_cells:
                u_for_logp = qu.mean.unsqueeze(0) 
            else:
                u_for_logp = qu.rsample(sample_shape=(n_u_samples_for_query_cells,))
            u_query_cells_list.append(u_for_logp.cpu())
        
        if not u_query_cells_list: # Should not happen if indices is not empty
            return pd.DataFrame(columns=active_posterior_sample_names, index=adata_manager_query.adata.obs_names[indices])

        u_query_cells_all = torch.cat(u_query_cells_list, dim=1) 
        if use_mean_u_for_query_cells and u_query_cells_all.ndim == 3 : u_query_cells_all = u_query_cells_all.squeeze(0)

        # Calculate log_prob of query cell u's under each aggregated posterior
        log_probs_matrix = np.zeros((len(indices), len(active_posterior_sample_names)))

        for col_idx, s_name in enumerate(active_posterior_sample_names):
            agg_posterior = aggregated_posteriors[s_name]
            
            log_probs_for_this_posterior_list = []
            # Number of u_samples taken for query cells
            n_query_u_effective_samples = u_query_cells_all.shape[0] if u_query_cells_all.ndim == 3 else 1
            
            for s_query_idx in range(n_query_u_effective_samples):
                current_u_slice = u_query_cells_all[s_query_idx] if u_query_cells_all.ndim == 3 else u_query_cells_all
                current_u_slice_cpu = current_u_slice.cpu() # Aggregated posterior components are on CPU
                log_p = agg_posterior.log_prob(current_u_slice_cpu)
                log_probs_for_this_posterior_list.append(log_p.numpy())
            
            log_probs_matrix[:, col_idx] = np.mean(np.stack(log_probs_for_this_posterior_list), axis=0)

        return pd.DataFrame(
            log_probs_matrix,
            index=adata_manager_query.adata.obs_names[indices],
            columns=active_posterior_sample_names
        )
    
    @torch.inference_mode()
    def differential_abundance(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None, # Sample-level covariate key in self.sample_info
        group1: str | Sequence[str] | None = None, # Value(s) in groupby column for group1 samples
        group2: str | Sequence[str] | None = None, # Value(s) in groupby column for group2 samples (or all others if None)
        sample_subset: Sequence[str] | None = None, # Subset of original sample categories to consider for posteriors
        n_u_draws_per_cell_for_posterior: int = 1, # For get_aggregated_posterior_u
        use_mean_u_for_cells: bool = True, # Use u.mean for cells whose DA is being computed
        n_u_samples_for_cells: int = 1, # If use_mean_u_for_cells=False, how many u samples per cell
        batch_size: int | None = None,
        min_cells_for_posterior: int = 5, # Min cells in a sample to form its aggregated posterior
    ) -> pd.DataFrame:
        """
        Performs differential abundance analysis in the `u` latent space.

        For each cell, this method computes the log probability of its `u` representation
        under the aggregated posterior `p(u|S)` for different sample groups `S`.
        It can compare two specific groups of samples or one group against all others,
        based on sample-level covariates defined in `self.sample_info`.

        Parameters
        ----------
        adata
            AnnData object. If `None`, uses `self.adata`.
        groupby
            A key from `self.sample_info.columns` that defines sample groups.
            For example, "condition" if `self.sample_info` has a "condition" column
            with values like "control", "treated".
        group1
            One or more categories from `self.sample_info[groupby]` to define the first group.
        group2
            One or more categories from `self.sample_info[groupby]` to define the second group.
            If `None`, group1 is compared against all other samples not in group1 (that are
            part of `sample_subset`, if specified, and have enough cells).
        sample_subset
            A list of original sample category names to restrict the analysis to.
            Aggregated posteriors will only be built for these samples. If `None`, all
            samples in `self.sample_info` are considered.
        n_u_draws_per_cell_for_posterior
            Number of `u` samples per cell used to construct each sample's aggregated posterior.
            If 1, uses the mean of `q(u)`.
        use_mean_u_for_cells
            For the cells whose differential abundance is being computed, whether to use the
            mean of their `q(u)` distribution or sample from it.
        n_u_samples_for_cells
            If `use_mean_u_for_cells` is False, number of `u` samples to draw per cell.
            The log probabilities will be averaged over these samples.
        batch_size
            Minibatch size.
        min_cells_for_posterior
            Minimum number of cells required for a sample category to be included
            when forming an aggregated posterior or the "rest" group.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by cell names, with columns including:
            - `log_prob_group1`: Log probability of cell's `u` under aggregated posterior of group1.
            - `log_prob_group2`: Log probability of cell's `u` under aggregated posterior of group2/rest.
            - `log_fc_abundance`: `log_prob_group1 - log_prob_group2`.
            - Potentially other stats depending on future enhancements.
        """
        self._check_if_trained(warn=False)
        adata_orig_passed = adata
        adata = self._validate_anndata(adata_orig_passed)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if not hasattr(self, "sample_info") or self.sample_info is None:
            self.update_sample_info(adata) # Ensure sample_info is populated
        if self.sample_info.empty:
            raise ValueError("self.sample_info is empty. Run update_sample_info() or ensure samples are set up.")

        if groupby is None:
            raise ValueError("`groupby` must be provided to define sample groups.")
        if groupby not in self.sample_info.columns:
            raise ValueError(f"`groupby` key '{groupby}' not found in self.sample_info.columns. Run update_sample_info().")
        if group1 is None:
            raise ValueError("`group1` must be specified.")

        if isinstance(group1, str): group1 = [group1]
        if isinstance(group2, str): group2 = [group2]

       # --- Determine samples belonging to group1, group2, and sample_subset ---
        target_sample_info = self.sample_info.copy()
        if sample_subset is not None:
            original_sample_key_col = adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).original_key
            if original_sample_key_col is None or original_sample_key_col not in target_sample_info.columns:
                 # If original_sample_key was None at setup, it might be stored as the encoded key name
                 # For safety, try to use the actual sample category names if available on sample_info index's values
                 # This part is tricky if original_sample_key_col is not reliable.
                 # Assuming sample_info's index map to original categories for now for subsetting
                 # This may need more robust handling if original_sample_key_col itself is not in sample_info
                 # For now, this relies on sample_info having a column with original sample names if subsetting by name.
                 # A better way: sample_info index IS the encoded sample. We need to map sample_subset names to codes.
                
                sample_name_to_code_map = {
                    name: code for code, name in enumerate(adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping)
                }
                sample_codes_to_keep = [sample_name_to_code_map[name] for name in sample_subset if name in sample_name_to_code_map]
                target_sample_info = target_sample_info[target_sample_info.index.isin(sample_codes_to_keep)]

            elif original_sample_key_col in target_sample_info.columns :
                target_sample_info = target_sample_info[target_sample_info[original_sample_key_col].isin(sample_subset)]
            else:
                 warnings.warn(f"Could not apply sample_subset as original sample key column '{original_sample_key_col}' "
                               f"not found in sample_info. Using all samples from sample_info.")

            if target_sample_info.empty:
                raise ValueError("No samples remaining after applying `sample_subset`.")

        samples_group1 = target_sample_info[target_sample_info[groupby].isin(group1)].index.tolist()
        if not samples_group1: 
            raise ValueError(f"No samples found for group1 with categories {group1} in '{groupby}'. Available: {target_sample_info[groupby].unique()}")

        samples_group2 = []
        if group2 is not None: # group2 is specified (now a list of strings)
            samples_group2 = target_sample_info[target_sample_info[groupby].isin(group2)].index.tolist()
            if not samples_group2: 
                raise ValueError(f"No samples found for group2 with categories {group2} in '{groupby}'. Available: {target_sample_info[groupby].unique()}")
        else: # group2 is None, so compare group1 against all other samples in target_sample_info
            samples_group2 = target_sample_info[~target_sample_info.index.isin(samples_group1)].index.tolist()
            if not samples_group2: 
                raise ValueError("No samples found for group2 (rest) after selecting group1. Is group1 all samples in target_sample_info?")
                    
        # Filter groups by min_cells_for_posterior
        # This requires cell counts per sample_code, which AnnDataManager has
        # For simplicity, let's assume all listed samples have enough cells.
        # A more robust implementation would filter here based on adata.obs['_scvi_sample'].value_counts().

        # --- Get U representations for all query cells ---
        # These are the cells for which we'll calculate differential abundance
        # Uses adata_orig_passed, or self.adata by default
        u_query_cells_list = []
        query_scdl = self._make_data_loader(adata=adata, indices=None, batch_size=batch_size, shuffle=False)
        for tensors in query_scdl:
            inf_inputs = self.module._get_inference_input(tensors)
            qu = self.module.qu(inf_inputs["x"], inf_inputs["y"], inf_inputs["sample_index"])
            if use_mean_u_for_cells:
                u_for_logp = qu.mean.unsqueeze(0) # (1, B, D_u)
            else:
                u_for_logp = qu.rsample(sample_shape=(n_u_samples_for_cells,)) # (S, B, D_u)
            u_query_cells_list.append(u_for_logp.cpu())
        
        u_query_cells = torch.cat(u_query_cells_list, dim=1) # (S_query, N_total, D_u)
        if use_mean_u_for_cells: u_query_cells = u_query_cells.squeeze(0) # (N_total, D_u)

        # --- Build aggregated posteriors and calculate log_probs ---
        results = {}
        print(f"\n[DA DEBUG] Group1 sample codes: {samples_group1}") # DEBUG
        print(f"[DA DEBUG] Group2 sample codes: {samples_group2}") # DEBUG

        for group_name, sample_codes_for_group in [("group1", samples_group1), ("group2", samples_group2)]:
            logger.info(f"Building aggregated posterior for {group_name} using {len(sample_codes_for_group)} sample categories.")
            print(f"[DA DEBUG] Processing {group_name} with sample_codes: {sample_codes_for_group}") # DEBUG
            
            if not sample_codes_for_group: # If the list of sample codes for this group is empty
                logger.warning(f"No sample codes provided for {group_name}. Skipping log_prob calculation.")
                results[f"log_prob_{group_name}"] = np.full(u_query_cells.shape[-2] if u_query_cells.ndim > 1 else u_query_cells.shape[0], np.nan)
                continue

            group_component_locs = []
            group_component_scales = []
            valid_samples_in_group_count = 0 # Count samples that actually contribute

            for s_code in sample_codes_for_group:
                print(f"[DA DEBUG]   {group_name} - checking sample_code: {s_code}") # DEBUG
                manager_enc_sample_key = self.adata_manager.data_registry[REGISTRY_KEYS.SAMPLE_KEY].attr_key
                # Ensure we are looking at self.adata for cell counts for posterior building
                s_indices = np.where(self.adata.obs[manager_enc_sample_key] == s_code)[0]
                
                print(f"[DA DEBUG]     Cells for sample_code {s_code}: {len(s_indices)}") # DEBUG

                if len(s_indices) < min_cells_for_posterior:
                    log_msg = f"Sample code {s_code} has < {min_cells_for_posterior} cells, skipping for {group_name} posterior."
                    print(f"[DA DEBUG]     {log_msg}") # DEBUG
                    logger.warning(log_msg)
                    continue # Skip this sample_code for building the group posterior
                
                valid_samples_in_group_count += 1
                # ... (rest of the loop to get locs/scales for this s_code) ...
                # ... (as in your existing working code) ...
                s_scdl = self._make_data_loader(adata=self.adata, indices=s_indices, batch_size=batch_size, shuffle=False)
                for tensors_s in s_scdl:
                    inf_inputs_s = self.module._get_inference_input(tensors_s)
                    qu_s = self.module.qu(inf_inputs_s["x"], inf_inputs_s["y"], inf_inputs_s["sample_index"])
                    if n_u_draws_per_cell_for_posterior == 1:
                        group_component_locs.append(qu_s.loc.detach().cpu())
                        group_component_scales.append(qu_s.scale.detach().cpu())
                    else:
                        u_samps = qu_s.rsample(sample_shape=(n_u_draws_per_cell_for_posterior,)).detach().cpu()
                        u_samps_flat = u_samps.permute(1,0,2).reshape(-1, u_samps.shape[-1])
                        group_component_locs.append(u_samps_flat)
                        group_component_scales.append(torch.full_like(u_samps_flat, 1e-3))
            
            if not group_component_locs: # This means no samples in the group met min_cells
                log_msg_empty_group = f"No valid samples with enough cells found for {group_name}. Skipping log_prob calculation for this group."
                print(f"[DA DEBUG]   {log_msg_empty_group}") # DEBUG
                logger.warning(log_msg_empty_group)
                results[f"log_prob_{group_name}"] = np.full(u_query_cells.shape[-2] if u_query_cells.ndim > 1 else u_query_cells.shape[0], np.nan)
                continue

            agg_locs = torch.cat(group_component_locs, dim=0)
            agg_scales = torch.cat(group_component_scales, dim=0)

            if agg_locs.shape[0] == 0: # Double check after concatenation
                logger.warning(f"No components for mixture model for {group_name} after concatenation. Skipping.")
                results[f"log_prob_{group_name}"] = np.full(u_query_cells.shape[-2] if u_query_cells.ndim > 1 else u_query_cells.shape[0], np.nan)
                continue
                
            agg_mix_logits = torch.ones(agg_locs.shape[0], device=agg_locs.device)
            
            base_agg_component_dist = Normal(loc=agg_locs, scale=agg_scales)
            independent_agg_component_dist = Independent(base_agg_component_dist, 1) # <--- CORRECTED HERE
            
            agg_posterior = MixtureSameFamily(
                Categorical(logits=agg_mix_logits),
                independent_agg_component_dist
            )

            # Calculate log_prob of query cell u's under this aggregated posterior
            # u_query_cells: (S_query, N_total, D_u) or (N_total, D_u)
            # agg_posterior.log_prob expects (..., D_u)
            # If u_query_cells is (S_query, N_total, D_u), log_prob needs to be applied per sample S_query and averaged
            log_probs_for_group_list = []
            n_query_u_samples = u_query_cells.shape[0] if u_query_cells.ndim == 3 else 1
            
            for s_query_idx in range(n_query_u_samples):
                current_u_slice = u_query_cells[s_query_idx] if u_query_cells.ndim == 3 else u_query_cells
                current_u_slice_dev = current_u_slice.to(agg_locs.device) # Move to device of mixture components
                log_p = agg_posterior.log_prob(current_u_slice_dev) # Shape (N_total)
                log_probs_for_group_list.append(log_p.cpu().numpy())
            
            results[f"log_prob_{group_name}"] = np.mean(np.stack(log_probs_for_group_list), axis=0)

        # --- Combine results into a DataFrame ---
        df_index = adata_manager.adata.obs_names # For all cells in the input adata
        if adata_orig_passed is not None and adata_orig_passed is not self.adata: # If user passed a specific adata
            df_index = adata_manager.adata.obs_names[self._validate_anndata(adata_orig_passed, copy_if_view=False).obs.index]


        da_df = pd.DataFrame(index=df_index)
        if f"log_prob_group1" in results:
            da_df["log_prob_group1"] = results["log_prob_group1"]
        if f"log_prob_group2" in results:
            da_df["log_prob_group2"] = results["log_prob_group2"]
        
        if "log_prob_group1" in da_df.columns and "log_prob_group2" in da_df.columns:
            da_df["log_fc_abundance"] = da_df["log_prob_group1"] - da_df["log_prob_group2"]
            
        return da_df
    
    def _construct_design_matrix(
        self,
        sample_cov_keys: Sequence[str],
        sample_info_df: pd.DataFrame,
        normalize_design_matrix: bool = True,
        add_batch_specific_offsets: bool = False,
        adata_manager_for_cats: AnnDataManager | None = None, # To get full category lists
        lfc_categorical_cov_key_original_name: str | None = None, # Name of the obs key being forced
        lfc_categorical_cov_forced_value: str | None = None, # The single value it's being forced to
    ) -> tuple[torch.Tensor, list[str], np.ndarray | None]:
        """
        Constructs a design matrix from sample-level covariates.

        Parameters
        ----------
        sample_cov_keys
            List of columns in `sample_info_df` to use as covariates.
        sample_info_df
            DataFrame containing sample-level annotations (typically a filtered self.sample_info).
            Index should be the scvi-tools encoded sample IDs.
        normalize_design_matrix
            Whether to normalize continuous covariates in the design matrix (0-1 scaling).
        add_batch_specific_offsets
            If True, and if 'batch_key_col' (or the original batch key) is in `sample_info_df`
            and model `n_batch > 0`, adds one-hot encoded batch offsets.

        Returns
        -------
        tuple
            - X_matrix: torch.Tensor of shape (n_samples_in_df, n_encoded_covariates)
            - X_matrix_col_names: list of names for the columns in X_matrix
            - offset_col_indices: np.ndarray of indices for offset columns, or None
        """
        X_list = []
        X_col_names_list = []
        
        # sample_info_df is already sorted by index if called from differential_expression
        sample_info_df_sorted = sample_info_df 

        offset_col_indices = None
        if add_batch_specific_offsets and self.summary_stats.n_batch > 0:
            original_model_batch_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key
            if original_model_batch_key and original_model_batch_key in sample_info_df_sorted.columns:
                sample_batch_categories = sample_info_df_sorted[original_model_batch_key].astype("category")
                model_batch_mapping = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
                sample_batch_codes = pd.Categorical(
                    sample_batch_categories, categories=model_batch_mapping, ordered=True
                ).codes
                valid_codes_mask = (sample_batch_codes != -1)
                if not np.all(valid_codes_mask):
                    warnings.warn("Some samples in design matrix have batch categories not in model's setup. No batch offset for these.", UserWarning)
                if np.any(valid_codes_mask):
                    offsets_raw = F.one_hot(
                        torch.tensor(sample_batch_codes[valid_codes_mask], dtype=torch.long),
                        num_classes=self.summary_stats.n_batch 
                    ).float()
                    if not np.all(valid_codes_mask):
                        padded_offsets = torch.zeros(len(sample_info_df_sorted), self.summary_stats.n_batch, dtype=torch.float32)
                        padded_offsets[valid_codes_mask] = offsets_raw
                        offsets_final = padded_offsets
                    else:
                        offsets_final = offsets_raw
                    X_list.append(offsets_final)
                    offset_names = [f"_offset_batch_{b_code}" for b_code in range(self.summary_stats.n_batch)]
                    X_col_names_list.extend(offset_names)
                    offset_col_indices = np.arange(len(offset_names))
            else:
                warnings.warn("`add_batch_specific_offsets` is True, but could not determine batch for samples. No offsets added.", UserWarning)
        # Use the correct adata_manager for resolving categories
        manager_for_categories = adata_manager_for_cats if adata_manager_for_cats is not None else self.adata_manager

        for cov_key in sample_cov_keys:
            if cov_key not in sample_info_df_sorted.columns:
                warnings.warn(f"Covariate '{cov_key}' not found in sample_info_df. Skipping.", UserWarning)
                continue
            
            cov_data_series = sample_info_df_sorted[cov_key] # This is the series from sample_info_df

            # Special handling for categorical LFC:
            # Re-type the series using all original categories to ensure get_dummies is consistent
            if lfc_categorical_cov_key_original_name and \
               cov_key == lfc_categorical_cov_key_original_name and \
               lfc_categorical_cov_forced_value is not None:
                
                original_categories = None
                # Attempt to get original categories from the *main model's anndata* via manager_for_categories
                # This assumes cov_key (lfc_categorical_cov_key_original_name) exists in manager_for_categories.adata.obs
                if cov_key in manager_for_categories.adata.obs:
                    original_column_from_adata = manager_for_categories.adata.obs[cov_key]
                    if pd.api.types.is_categorical_dtype(original_column_from_adata.dtype):
                        original_categories = original_column_from_adata.cat.categories.tolist()
                    elif original_column_from_adata.dtype == 'object' or pd.api.types.is_bool_dtype(original_column_from_adata.dtype):
                        original_categories = sorted(list(original_column_from_adata.unique()))
                        # Ensure baseline and perturbed categories are in these original categories
                        if lfc_categorical_cov_forced_value not in original_categories:
                             original_categories.append(lfc_categorical_cov_forced_value)
                             original_categories.sort() # Keep it sorted if possible
                        # We also need to ensure the *other* category (baseline or perturbed) is present
                        # This part is tricky. The LFC baseline/perturbed matrices are built independently.
                        # The key is that get_dummies sees all *possible* categories.

                if original_categories:
                    # Create a new series for this covariate that has the forced value,
                    # but is typed with all original categories.
                    # The sample_info_df ALREADY has the forced value in this column if this call
                    # is for e.g. sample_info_for_lfc_baseline.
                    # We just need to ensure its .cat.categories is correct before get_dummies.
                    current_col_as_cat = pd.Categorical(
                        cov_data_series, # This series already has the forced value
                        categories=original_categories, 
                        ordered=False
                    )
                    # If cov_data_series contains a value not in original_categories (should not happen if set correctly),
                    # pd.Categorical will make it NaN. We should ensure forced_value is in original_categories.
                    if lfc_categorical_cov_forced_value not in current_col_as_cat.categories:
                         warnings.warn(f"Forced LFC category '{lfc_categorical_cov_forced_value}' for key '{cov_key}' "
                                       f"is not among original categories '{original_categories}'. Dummification might be problematic.")

                    one_hot_cov = get_dummies(current_col_as_cat, prefix=cov_key, drop_first=True, dtype=np.float32)
                else:
                    warnings.warn(f"Could not determine original categories for LFC key '{cov_key}'. "
                                  f"Using categories present in current sample_info_df slice.", UserWarning)
                    # Fallback: dummify based on whatever categories are in cov_data_series
                    # This was the cause of the LFC design matrix column mismatch error.
                    # If cov_data_series only has one unique value (the forced one), get_dummies(drop_first=True) is empty.
                    # We MUST use the full set of categories.
                    # If original_categories could not be found, this will lead to an empty one_hot_cov.
                    # It's better to raise an error or ensure original_categories is always found.
                    # For now, this path will likely result in [] for perturbed_cols if only one cat is present.
                    # Let's force it to use categories from the original manager if they can't be found on the current adata.obs
                    # This assumes cov_key is a registered field with categorical_mapping
                    try:
                        state_reg_cat_mapping = manager_for_categories.get_state_registry(cov_key).categorical_mapping
                        if state_reg_cat_mapping is not None:
                            cat_type = pd.CategoricalDtype(categories=state_reg_cat_mapping, ordered=False)
                            cov_data_series_retyped = pd.Series(cov_data_series.values, index=cov_data_series.index, dtype=cat_type)
                            one_hot_cov = get_dummies(cov_data_series_retyped, prefix=cov_key, drop_first=True, dtype=np.float32)
                        else: # Should not happen if cov_key is a sample_cov_key meant to be categorical
                            one_hot_cov = pd.DataFrame(index=sample_info_df_sorted.index) # Empty
                    except KeyError: # cov_key might not be a directly registered field (e.g. from adata.obs directly)
                         one_hot_cov = pd.DataFrame(index=sample_info_df_sorted.index) # Empty
                         warnings.warn(f"Could not determine categorical mapping for '{cov_key}' from manager. LFC dummification may be empty.")


            elif pd.api.types.is_categorical_dtype(cov_data_series) or \
                 cov_data_series.dtype == 'object' or \
                 pd.api.types.is_bool_dtype(cov_data_series): # Standard categorical handling
                cov_data_series = cov_data_series.astype('category')
                one_hot_cov = get_dummies(cov_data_series, prefix=cov_key, drop_first=True, dtype=np.float32)
            
            # Numeric or already processed one_hot_cov
            if 'one_hot_cov' in locals() and isinstance(one_hot_cov, pd.DataFrame):
                X_list.append(torch.from_numpy(one_hot_cov.values))
                X_col_names_list.extend(one_hot_cov.columns.tolist())
                del one_hot_cov # Clear for next iteration
            elif pd.api.types.is_numeric_dtype(cov_data_series):
                numeric_cov = cov_data_series.values.astype(np.float32).reshape(-1, 1)
                if normalize_design_matrix:
                    min_val, max_val = numeric_cov.min(), numeric_cov.max()
                    if (max_val - min_val) > 1e-6: 
                        numeric_cov = (numeric_cov - min_val) / (max_val - min_val)
                    elif np.abs(max_val) > 1e-6 : 
                        numeric_cov = numeric_cov / max_val 
                X_list.append(torch.from_numpy(numeric_cov))
                X_col_names_list.append(cov_key)
            elif not (lfc_categorical_cov_key_original_name and cov_key == lfc_categorical_cov_key_original_name):
                # Warning only if not handled by the LFC specific logic already (which might lead to empty one_hot_cov)
                warnings.warn(f"Covariate '{cov_key}' with dtype {cov_data_series.dtype} was not processed as categorical or numeric. Skipping.", UserWarning)


        if not X_list:
            return torch.empty(len(sample_info_df_sorted), 0, dtype=torch.float32), [], None

        X_matrix = torch.cat(X_list, dim=1)
        return X_matrix, X_col_names_list, offset_col_indices
    
    @torch.inference_mode()
    def differential_expression(
        self,
        adata: AnnData | None = None,
        sample_cov_keys: Sequence[str] | None = None, # Covariates from self.sample_info to model
        sample_subset: Sequence[str] | None = None,   # Filter samples by original names
        filter_inadmissible_samples: bool = False,    # NEW: Placeholder for now
        admissibility_threshold: float = 0.0,     # NEW: For filter_inadmissible_samples
        normalize_design_matrix: bool = True,
        add_batch_specific_offsets_to_design: bool = False,
        compute_lfc: bool = False,
        lfc_covariate_name: str | None = None, # The single covariate (original or dummified) to perturb for LFC
        lfc_val_perturbed: float | str = 1.0,  # Value/category for perturbed state
        lfc_val_baseline: float | str = 0.0,   # Value/category for baseline state
        store_lfc_expression_values: bool = False,
        n_u_samples: int = 1,
        give_u_mean: bool = True,
        batch_size: int | None = None,
        correction_method: Literal["fdr_bh", "bonferroni"] | None = "fdr_bh",
        lfc_reg_eps: float = 1e-8,
        beta_cov_reg: float = 1e-6,
        lambda_reg: float = 0.0,
    ) -> xr.Dataset:
        """
        Performs cell-specific multivariate differential expression analysis on latent shifts.

        For each cell, this method identifies how sample-level covariates (defined in
        `self.sample_info` via `sample_cov_keys`) influence the cell's state shift
        in the z-latent space. This shift is conceptualized as `eps = z_counterfactual - z_base_of_u`,
        where `z_base_of_u` is derived from the cell's intrinsic state `u`, and
        `z_counterfactual` is the cell's state if it belonged to a specific sample context.
        A linear model `eps_vector = DesignMatrix @ beta_cell` is fit for each cell.

        Optionally, this method can compute log-fold changes (LFCs) in RNA and protein
        expression space for a specified covariate.

        Parameters
        ----------
        adata
            AnnData object. If `None`, uses `self.adata`.
        sample_cov_keys
            List of column names in `self.sample_info` to use as covariates in the
            linear model. These should represent sample-level annotations.
        sample_subset
            Optional list of original sample category names (from the column specified
            as `sample_key` during `setup_anndata`) to restrict the analysis to.
            Only these samples will be included in constructing the design matrix and
            in generating counterfactual epsilon vectors.
        filter_inadmissible_samples
            If True, attempts to filter out cell-sample pairs based on admissibility scores
            before fitting the linear model. (Currently a placeholder, issues a warning).
        admissibility_threshold
            Threshold for `filter_inadmissible_samples`. (Currently unused).
        lfc_covariate_name
            The column name from `self.sample_info` (if categorical and in `sample_cov_keys`)
            or from the design matrix (if continuous/binary or already dummified)
            for which LFC is to be calculated. Required if `compute_lfc` is True.
        lfc_val_perturbed
            Value or category name for the "perturbed" state of `lfc_covariate_name`.
        lfc_val_baseline
            Value or category name for the "baseline" state of `lfc_covariate_name`.
        store_lfc_expression_values
            If `True` and `compute_lfc` is `True`, stores the mean baseline and perturbed
            expression rates for RNA and protein in the output Dataset. These are the values
            used to compute the LFCs. Default is `False`.
        n_u_samples
            Number of `u` samples to draw per cell if `give_u_mean` is False.
            These `u` samples are used for calculating the epsilon vectors and, if `compute_lfc`
            is True, for generating baseline and perturbed expression.
        give_u_mean
            If True, use the mean of the `q(u)` distribution for each cell. Otherwise, draw
            `n_u_samples` from `q(u)` and average results (betas, LFCs) if `n_u_samples > 1`.
            For LFC calculation, the mean of `u` samples is used to get `z_base`.
        batch_size
            Minibatch size for data loading.
        correction_method
            Method for multiple testing correction on p-values (currently placeholder p-values).
            Options: "fdr_bh", "bonferroni", or None.
        lfc_reg_eps
            Small epsilon added for numerical stability during LFC calculation (`log2(x + eps)`).
        beta_cov_reg
            Small constant added to the diagonal of (X^T X) for numerical stability
            of the pseudo-inverse, used in beta estimation.
        lambda_reg
            L2 regularization strength for beta coefficient estimation (Ridge regression).
            A larger value will shrink beta coefficients towards zero. Default is 0.0 (no L2 regularization beyond pinv stability).

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the differential expression results.
            Coordinates typically include 'cell_obs_name', 'design_covariate',
            'latent_dim', 'gene_name', 'protein_name'.
            Data variables include:
            - 'betas': (cell_obs_name, design_covariate, latent_dim)
            - 'p_values': (cell_obs_name, design_covariate)
            - 'q_values': (cell_obs_name, design_covariate)
            - If `compute_lfc` is True:
                - 'lfc_rna': (cell_obs_name, gene_name)
                - 'lfc_protein': (cell_obs_name, protein_name)
            - If `store_lfc_expression_values` is True:
                - 'rna_rate_baseline': (cell_obs_name, gene_name)
                - 'rna_rate_perturbed': (cell_obs_name, gene_name)
                - 'protein_rate_baseline': (cell_obs_name, protein_name)
                - 'protein_rate_perturbed': (cell_obs_name, protein_name)
        """
        self._check_if_trained(warn=False)
        adata_query = self._validate_anndata(adata) 
        adata_manager = self.get_anndata_manager(adata_query, required=True)

        # --- Parameter Validation ---
        if compute_lfc and lfc_covariate_name is None: # Simplified from previous version for LFC params
             raise ValueError("If `compute_lfc` is True, `lfc_covariate_name` must be specified to indicate which covariate to perturb.")
        # The logic for lfc_condition_key_in_sample_info was removed in favor of a unified lfc_covariate_name

        if not hasattr(self, "sample_info") or self.sample_info is None or self.sample_info.empty:
            self.update_sample_info(self.adata)
        if self.sample_info.empty: raise ValueError("self.sample_info is empty.")
        if sample_cov_keys is None or not sample_cov_keys: raise ValueError("`sample_cov_keys` must be provided.")

        # --- Placeholder for filter_inadmissible_samples ---
        if filter_inadmissible_samples:
            warnings.warn(
                "`filter_inadmissible_samples` is not yet fully implemented in this version of TOTALMRVI. "
                "Proceeding without filtering inadmissible samples.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel
            )
            # When implemented, this section would:
            # 1. Call a method like self.get_admissibility_scores(adata_query, threshold=admissibility_threshold)
            #    which would return a boolean matrix: admissible_mask[cell_idx, target_sample_code_idx]
            # 2. This mask would then be used inside the main cell loop to filter
            #    Eps_for_cells_in_batch and X_matrix_torch on a per-cell basis.
            #    This makes n_samples_in_design and df_residuals cell-specific.

        # --- Prepare Sample Info and Design Matrix (as before) ---
        # ... (all the logic for _sample_info_for_design, X_matrix_torch, LFC design matrices) ...
        # ... (This part remains the same as the version that just passed all tests) ...
        _sample_info_for_design = self.sample_info.copy()
        original_sample_col_name_in_obs = adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).original_key
        if sample_subset is not None:
            if original_sample_col_name_in_obs and original_sample_col_name_in_obs in _sample_info_for_design.columns:
                _sample_info_for_design = _sample_info_for_design[_sample_info_for_design[original_sample_col_name_in_obs].isin(sample_subset)]
            else:
                mapping = adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
                sample_name_to_code_map = {name: i for i, name in enumerate(mapping)}
                codes_to_keep = [sample_name_to_code_map[s] for s in sample_subset if s in sample_name_to_code_map]
                _sample_info_for_design = _sample_info_for_design[_sample_info_for_design.index.isin(codes_to_keep)]
            if _sample_info_for_design.empty: raise ValueError("No samples left after `sample_subset`.")
        
        target_sample_codes_for_eps = _sample_info_for_design.index.to_numpy(dtype=int)
        n_samples_in_design = len(target_sample_codes_for_eps)
        if n_samples_in_design == 0: raise ValueError("No samples for design matrix after subsetting.")

        X_matrix_torch, X_matrix_col_names, _ = self._construct_design_matrix(
            sample_cov_keys, _sample_info_for_design, normalize_design_matrix, add_batch_specific_offsets_to_design,
            adata_manager_for_cats=self.adata_manager
        )
        X_matrix_torch = X_matrix_torch.to(self.device)
        n_design_covariates = X_matrix_torch.shape[1]
        if n_design_covariates == 0: raise ValueError("Design matrix has no columns.")

        X_matrix_baseline_lfc_global, X_matrix_perturbed_lfc_global = None, None
        lfc_cov_idx_in_design = -1 
        if compute_lfc:
            is_lfc_categorical = False
            original_lfc_cov_categories = None
            if lfc_covariate_name in _sample_info_for_design.columns:
                cov_data_type = _sample_info_for_design[lfc_covariate_name].dtype
                if pd.api.types.is_categorical_dtype(cov_data_type) or cov_data_type == 'object' or pd.api.types.is_bool_dtype(cov_data_type):
                    is_lfc_categorical = True
                    if lfc_covariate_name in self.adata_manager.adata.obs: # Use self.adata for original categories
                         original_column_from_adata = self.adata_manager.adata.obs[lfc_covariate_name]
                         if pd.api.types.is_categorical_dtype(original_column_from_adata.dtype):
                             original_lfc_cov_categories = original_column_from_adata.cat.categories.tolist()
                         elif original_column_from_adata.dtype == 'object' or pd.api.types.is_bool_dtype(original_column_from_adata.dtype):
                             original_lfc_cov_categories = sorted(list(original_column_from_adata.unique()))
                    if original_lfc_cov_categories is None: # Fallback if not in self.adata.obs (e.g. generated in sample_info)
                        original_lfc_cov_categories = sorted(list(_sample_info_for_design[lfc_covariate_name].unique()))
            
            if is_lfc_categorical:
                if not isinstance(lfc_val_baseline, str) or not isinstance(lfc_val_perturbed, str):
                    raise ValueError("For categorical LFC, `lfc_val_baseline` and `lfc_val_perturbed` must be category names (strings).")
                if original_lfc_cov_categories and lfc_val_baseline not in original_lfc_cov_categories:
                    raise ValueError(f"Baseline category '{lfc_val_baseline}' for LFC not found in original categories of '{lfc_covariate_name}': {original_lfc_cov_categories}")
                if original_lfc_cov_categories and lfc_val_perturbed not in original_lfc_cov_categories:
                    raise ValueError(f"Perturbed category '{lfc_val_perturbed}' for LFC not found in original categories of '{lfc_covariate_name}': {original_lfc_cov_categories}")

                sample_info_lfc_baseline_prep = _sample_info_for_design.copy()
                sample_info_lfc_baseline_prep[lfc_covariate_name] = lfc_val_baseline
                X_matrix_baseline_lfc_global, base_cols, _ = self._construct_design_matrix(
                    sample_cov_keys, sample_info_lfc_baseline_prep, normalize_design_matrix, add_batch_specific_offsets_to_design,
                    adata_manager_for_cats=self.adata_manager,
                    lfc_categorical_cov_key_original_name=lfc_covariate_name, 
                    lfc_categorical_cov_forced_value=str(lfc_val_baseline)
                )
                sample_info_lfc_perturbed_prep = _sample_info_for_design.copy()
                sample_info_lfc_perturbed_prep[lfc_covariate_name] = lfc_val_perturbed
                X_matrix_perturbed_lfc_global, pert_cols, _ = self._construct_design_matrix(
                    sample_cov_keys, sample_info_lfc_perturbed_prep, normalize_design_matrix, add_batch_specific_offsets_to_design,
                    adata_manager_for_cats=self.adata_manager,
                    lfc_categorical_cov_key_original_name=lfc_covariate_name,
                    lfc_categorical_cov_forced_value=str(lfc_val_perturbed)
                )
                if not base_cols == pert_cols or not base_cols == X_matrix_col_names:
                    raise RuntimeError(f"LFC design matrix columns mismatch. Main: {X_matrix_col_names}, Base: {base_cols}, Pert: {pert_cols}")
            elif lfc_covariate_name: 
                if lfc_covariate_name not in X_matrix_col_names:
                    raise ValueError(f"lfc_covariate_name '{lfc_covariate_name}' not in design: {X_matrix_col_names}")
                lfc_cov_idx_in_design = X_matrix_col_names.index(lfc_covariate_name)
                X_matrix_baseline_lfc_global = X_matrix_torch.clone()
                X_matrix_baseline_lfc_global[:, lfc_cov_idx_in_design] = float(lfc_val_baseline)
                X_matrix_perturbed_lfc_global = X_matrix_torch.clone()
                X_matrix_perturbed_lfc_global[:, lfc_cov_idx_in_design] = float(lfc_val_perturbed)
            if X_matrix_baseline_lfc_global is None or X_matrix_perturbed_lfc_global is None:
                raise ValueError("LFC: Failed to create baseline/perturbed design matrices.")
            X_matrix_baseline_lfc_global = X_matrix_baseline_lfc_global.to(self.device)
            X_matrix_perturbed_lfc_global = X_matrix_perturbed_lfc_global.to(self.device)


        # --- Main Loop (Beta, P-value, LFC calculation - as previously successful) ---
        # ... (Loop over scdl) ...
        # ... (Assemble xr.Dataset) ...
        # The entire loop and result assembly remains IDENTICAL to the version that just passed all tests.
        # I will paste it here for completeness, assuming it's unchanged from the one you have.

        all_betas_list, all_p_values_list = [], []
        all_rna_baseline_lfc_list, all_rna_perturbed_lfc_list = [], []
        all_pro_baseline_lfc_list, all_pro_perturbed_lfc_list = [], []

        scdl = self._make_data_loader(adata=adata_query, indices=None, batch_size=batch_size, shuffle=False)
        n_latent_z = self.module.n_latent
        df_chi2 = float(n_latent_z) 

        self.module.eval()
        for tensors in scdl:
            n_cells_in_batch = tensors[REGISTRY_KEYS.X_KEY].shape[0]
            inf_inputs = self.module._get_inference_input(tensors)
            original_cell_sample_codes = inf_inputs["sample_index"] 
            original_cell_batch_codes = inf_inputs["batch_index"]   
            qu = self.module.qu(inf_inputs["x"], inf_inputs["y"], original_cell_sample_codes)
            u_batch_samples = qu.mean.unsqueeze(0) if give_u_mean else qu.rsample(sample_shape=(n_u_samples,))
            Eps_for_cells_in_batch = torch.zeros(n_cells_in_batch, n_samples_in_design, n_latent_z, device=self.device)
            z_base_orig_avg_over_u = torch.zeros(n_cells_in_batch, n_latent_z, device=self.device)
            for u_idx in range(u_batch_samples.shape[0]):
                current_u_for_batch = u_batch_samples[u_idx]
                z_base_this_u, _ = self.module.qz(current_u_for_batch, original_cell_sample_codes)
                z_base_orig_avg_over_u += z_base_this_u
                for i, target_sample_code in enumerate(target_sample_codes_for_eps):
                    target_sample_tensor = torch.full((n_cells_in_batch,), fill_value=target_sample_code, dtype=torch.long, device=self.device)
                    _, eps_val_or_params_cf = self.module.qz(current_u_for_batch, target_sample_tensor)
                    current_eps_cf = eps_val_or_params_cf if self.module.qz.use_map else eps_val_or_params_cf[0]
                    Eps_for_cells_in_batch[:, i, :] += current_eps_cf
            Eps_for_cells_in_batch /= u_batch_samples.shape[0]
            z_base_orig_avg_over_u /= u_batch_samples.shape[0]

            XtX = X_matrix_torch.t() @ X_matrix_torch
            total_regularization_diag = (beta_cov_reg + lambda_reg) * torch.eye(XtX.shape[0], device=XtX.device)
            XtX_inv = torch.linalg.pinv(XtX + total_regularization_diag) 
            XtEps = torch.einsum('cs,bsd->cbd', X_matrix_torch.t(), Eps_for_cells_in_batch)
            beta_batch_T = torch.einsum('co,obd->cbd', XtX_inv, XtEps)
            beta_batch = beta_batch_T.permute(1,0,2)
            all_betas_list.append(beta_batch.cpu())
            
            df_residuals = n_samples_in_design - n_design_covariates
            batch_p_values = torch.full((n_cells_in_batch, n_design_covariates), float('nan'), device=self.device)
            if df_residuals <= 0:
                warnings.warn("DoF for residual variance <= 0. P-values remain NaN.", UserWarning, stacklevel=settings.warnings_stacklevel)
            else:
                Pred_Eps_for_batch = torch.einsum('sc,bcd->bsd', X_matrix_torch, beta_batch)
                Residuals_batch = Eps_for_cells_in_batch - Pred_Eps_for_batch
                sigma_sq_error_batch = torch.sum(Residuals_batch**2, dim=1) / df_residuals
                for k_cov in range(n_design_covariates):
                    beta_k_batch = beta_batch[:, k_cov, :]
                    if lambda_reg > 0:
                         XtX_inv_for_pval_calc_diag_term = torch.linalg.pinv(XtX + beta_cov_reg * torch.eye(XtX.shape[0], device=XtX.device))[k_cov, k_cov]
                         xtx_inv_kk = XtX_inv_for_pval_calc_diag_term
                    else: 
                         xtx_inv_kk = XtX_inv[k_cov, k_cov]
                    if xtx_inv_kk < beta_cov_reg / 2: xtx_inv_kk = torch.tensor(beta_cov_reg / 2 if beta_cov_reg > 0 else 1e-6, device=self.device)
                    sigma_sq_safe = torch.where(sigma_sq_error_batch > beta_cov_reg, sigma_sq_error_batch, torch.tensor(beta_cov_reg, device=self.device))
                    Wald_stats_k_cov = torch.full((n_cells_in_batch,), float('nan'), device=self.device)
                    if torch.isnan(sigma_sq_safe).any():
                        warnings.warn(f"Residual variance for cov {X_matrix_col_names[k_cov]} has NaNs.", UserWarning, stacklevel=settings.warnings_stacklevel)
                    else:
                        Wald_stats_k_cov = (1.0 / xtx_inv_kk) * torch.sum(beta_k_batch**2 / sigma_sq_safe, dim=1)
                    p_vals_for_this_cov = torch.full_like(Wald_stats_k_cov, float('nan'))
                    valid_wald_mask = ~torch.isnan(Wald_stats_k_cov)
                    if valid_wald_mask.any():
                        df_chi2_tensor = torch.tensor(df_chi2, dtype=torch.float32, device=self.device)
                        if df_chi2_tensor <= 0:
                            warnings.warn(f"Chi2 DoF ({df_chi2}) not positive. P-values for cov {X_matrix_col_names[k_cov]} remain NaN.", UserWarning, stacklevel=settings.warnings_stacklevel)
                        else:
                            cdf_input = torch.clamp(Wald_stats_k_cov[valid_wald_mask], min=0.0)
                            p_vals_for_this_cov[valid_wald_mask] = 1.0 - Chi2(df_chi2_tensor).cdf(cdf_input)
                    batch_p_values[:, k_cov] = p_vals_for_this_cov
            all_p_values_list.append(batch_p_values.cpu())

            if compute_lfc:
                library_rna_batch = torch.log(tensors[REGISTRY_KEYS.X_KEY].sum(dim=1, keepdim=True) + 1e-8).to(self.device)
                if library_rna_batch.ndim == 1: library_rna_batch = library_rna_batch.unsqueeze(1)
                pred_eps_baseline_all_contexts_lfc = torch.einsum('sc,bcd->bsd', X_matrix_baseline_lfc_global, beta_batch)
                pred_eps_perturbed_all_contexts_lfc = torch.einsum('sc,bcd->bsd', X_matrix_perturbed_lfc_global, beta_batch)
                eps_baseline_for_cells_lfc = torch.zeros(n_cells_in_batch, n_latent_z, device=self.device)
                eps_perturbed_for_cells_lfc = torch.zeros(n_cells_in_batch, n_latent_z, device=self.device)
                map_orig_code_to_design_idx = {code: i for i, code in enumerate(target_sample_codes_for_eps)}
                for cell_idx_in_batch in range(n_cells_in_batch):
                    orig_sample_code_of_cell = original_cell_sample_codes[cell_idx_in_batch].item()
                    if orig_sample_code_of_cell in map_orig_code_to_design_idx:
                        design_row_idx = map_orig_code_to_design_idx[orig_sample_code_of_cell]
                        eps_baseline_for_cells_lfc[cell_idx_in_batch, :] = pred_eps_baseline_all_contexts_lfc[cell_idx_in_batch, design_row_idx, :]
                        eps_perturbed_for_cells_lfc[cell_idx_in_batch, :] = pred_eps_perturbed_all_contexts_lfc[cell_idx_in_batch, design_row_idx, :]
                    else:
                        warnings.warn(f"Cell's original sample code {orig_sample_code_of_cell} not in LFC design. Using mean effect.", UserWarning, stacklevel=settings.warnings_stacklevel)
                        eps_baseline_for_cells_lfc[cell_idx_in_batch, :] = pred_eps_baseline_all_contexts_lfc[cell_idx_in_batch, :, :].mean(0)
                        eps_perturbed_for_cells_lfc[cell_idx_in_batch, :] = pred_eps_perturbed_all_contexts_lfc[cell_idx_in_batch, :, :].mean(0)
                z_baseline_eff = z_base_orig_avg_over_u + eps_baseline_for_cells_lfc
                z_perturbed_eff = z_base_orig_avg_over_u + eps_perturbed_for_cells_lfc
                q_beta_base = self.module.background_encoder(z_baseline_eff, original_cell_batch_codes)
                logbeta_base = q_beta_base.mean 
                gen_out_base = self.module.generative(z=z_baseline_eff, library=library_rna_batch, batch_index=original_cell_batch_codes, logbeta=logbeta_base)
                q_beta_pert = self.module.background_encoder(z_perturbed_eff, original_cell_batch_codes)
                logbeta_pert = q_beta_pert.mean
                gen_out_pert = self.module.generative(z=z_perturbed_eff, library=library_rna_batch, batch_index=original_cell_batch_codes, logbeta=logbeta_pert)
                all_rna_baseline_lfc_list.append(gen_out_base["px_rate"].cpu())
                all_rna_perturbed_lfc_list.append(gen_out_pert["px_rate"].cpu())
                pi_bg_base = torch.sigmoid(gen_out_base["py_mixing"])
                pro_rate_base = (gen_out_base["py_rate_fore"] * torch.sigmoid(-gen_out_base["py_mixing"])) + (gen_out_base["py_rate_back"] * pi_bg_base)
                all_pro_baseline_lfc_list.append(pro_rate_base.cpu())
                pi_bg_pert = torch.sigmoid(gen_out_pert["py_mixing"])
                pro_rate_pert = (gen_out_pert["py_rate_fore"] * torch.sigmoid(-gen_out_pert["py_mixing"])) + (gen_out_pert["py_rate_back"] * pi_bg_pert)
                all_pro_perturbed_lfc_list.append(pro_rate_pert.cpu())
        
        final_betas_np = torch.cat(all_betas_list, dim=0).numpy()       
        final_p_values_np = torch.cat(all_p_values_list, dim=0).numpy() 
        processed_indices = scdl.dataset.indices
        cell_obs_names = adata_manager.adata.obs_names[processed_indices].to_list()
        latent_dims_coords = [f"latent_dim_{j}" for j in range(n_latent_z)]
        data_vars = {
            "betas": xr.DataArray(final_betas_np, dims=["cell_obs_name", "design_covariate", "latent_dim"],
                                  coords={"cell_obs_name": cell_obs_names, "design_covariate": X_matrix_col_names, "latent_dim": latent_dims_coords}),
            "p_values": xr.DataArray(final_p_values_np, dims=["cell_obs_name", "design_covariate"],
                                     coords={"cell_obs_name": cell_obs_names, "design_covariate": X_matrix_col_names})
        }
        if correction_method:
            final_q_values_np = np.full_like(final_p_values_np, np.nan)
            for i in range(len(X_matrix_col_names)): # Iterate using index
                current_pvals = final_p_values_np[:, i]
                valid_pvals_mask = ~np.isnan(current_pvals)
                pvals_to_correct = np.clip(current_pvals[valid_pvals_mask], 0, 1)
                if len(pvals_to_correct) > 0:
                    if correction_method == "fdr_bh":
                        fdr_corrected = false_discovery_control(pvals_to_correct)
                        if len(fdr_corrected) == np.sum(valid_pvals_mask):
                           final_q_values_np[valid_pvals_mask, i] = fdr_corrected
                    elif correction_method == "bonferroni":
                         final_q_values_np[valid_pvals_mask, i] = np.clip(pvals_to_correct * len(pvals_to_correct), 0, 1)
            data_vars["q_values"] = xr.DataArray(final_q_values_np, dims=["cell_obs_name", "design_covariate"],
                                                 coords={"cell_obs_name": cell_obs_names, "design_covariate": X_matrix_col_names})
        if compute_lfc and all_rna_baseline_lfc_list: 
            final_rna_baseline_lfc = torch.cat(all_rna_baseline_lfc_list, dim=0).numpy()
            final_rna_perturbed_lfc = torch.cat(all_rna_perturbed_lfc_list, dim=0).numpy()
            final_pro_baseline_lfc = torch.cat(all_pro_baseline_lfc_list, dim=0).numpy()
            final_pro_perturbed_lfc = torch.cat(all_pro_perturbed_lfc_list, dim=0).numpy()
            lfc_rna = np.log2(final_rna_perturbed_lfc + lfc_reg_eps) - np.log2(final_rna_baseline_lfc + lfc_reg_eps)
            lfc_pro = np.log2(final_pro_perturbed_lfc + lfc_reg_eps) - np.log2(final_pro_baseline_lfc + lfc_reg_eps)
            gene_names_list = _get_var_names_from_manager(adata_manager).tolist()
            protein_names_list = list(adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
            data_vars["lfc_rna"] = xr.DataArray(lfc_rna, dims=["cell_obs_name", "gene_name"],
                                                coords={"cell_obs_name": cell_obs_names, "gene_name": gene_names_list})
            data_vars["lfc_protein"] = xr.DataArray(lfc_pro, dims=["cell_obs_name", "protein_name"],
                                                    coords={"cell_obs_name": cell_obs_names, "protein_name": protein_names_list})
            if store_lfc_expression_values:
                data_vars["rna_rate_baseline"] = xr.DataArray(final_rna_baseline_lfc, dims=["cell_obs_name", "gene_name"], coords={"cell_obs_name": cell_obs_names, "gene_name": gene_names_list})
                data_vars["rna_rate_perturbed"] = xr.DataArray(final_rna_perturbed_lfc, dims=["cell_obs_name", "gene_name"], coords={"cell_obs_name": cell_obs_names, "gene_name": gene_names_list})
                data_vars["protein_rate_baseline"] = xr.DataArray(final_pro_baseline_lfc, dims=["cell_obs_name", "protein_name"], coords={"cell_obs_name": cell_obs_names, "protein_name": protein_names_list})
                data_vars["protein_rate_perturbed"] = xr.DataArray(final_pro_perturbed_lfc, dims=["cell_obs_name", "protein_name"], coords={"cell_obs_name": cell_obs_names, "protein_name": protein_names_list})
        
        return xr.Dataset(data_vars)