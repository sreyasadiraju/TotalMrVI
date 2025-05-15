import pytest
import torch
import numpy as np
import pandas as pd
from anndata import AnnData
from scvi import REGISTRY_KEYS

@pytest.fixture
def dummy_totalmrvi_adata():
    """Fixture to generate dummy AnnData objects for TOTALMRVI tests."""
    def _dummy_totalmrvi_adata(
        n_obs=100,
        n_genes=50,
        n_proteins=20,
        n_batches=2,
        n_samples=3,
        n_labels=1,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0 for dummy_totalmrvi_adata")
        if n_labels <= 0:
            raise ValueError("n_labels must be > 0 for dummy_totalmrvi_adata")

        obs_dict = {
            "sample_key_col": np.random.choice([f"sample_{i}" for i in range(n_samples)], size=n_obs),
            "labels_key_col": np.random.choice([f"label_{i}" for i in range(n_labels)], size=n_obs),
        }

        if n_batches > 0:
            obs_dict["batch_key_col"] = np.random.choice([f"batch_{i}" for i in range(n_batches)], size=n_obs)
        else:
            obs_dict["batch_key_col"] = np.array([f"batch_default"] * n_obs, dtype=object)

        obs_names = pd.Index([f"cell_{i}" for i in range(n_obs)])
        var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])

        adata = AnnData(
            X=np.random.poisson(5, size=(n_obs, n_genes)).astype(np.float32),
            obsm={
                "protein_expression": np.random.poisson(10, size=(n_obs, n_proteins)).astype(np.float32)
            },
            obs=pd.DataFrame(obs_dict, index=obs_names),
            var=pd.DataFrame(index=var_names),
            uns={}
        )
        adata.uns["protein_names_col"] = [f"protein_{i}" for i in range(n_proteins)]
        return adata
    return _dummy_totalmrvi_adata

@pytest.fixture
def fake_data_fixture():
    """Fixture to generate fake batch data for tests."""
    def _fake_batch(B=16, G=100, P=20, n_samples_cat=5, n_batches_cat=4):
        # Ensure n_batches_cat >= 0. If 0, batch indices are still 0 but n_batch in module is 0.
        # For F.one_hot, num_classes must be > 0 if it's called.
        # Modules handle n_batch=0 by not calling F.one_hot with num_classes=0.
        
        actual_n_batches_for_randint = n_batches_cat if n_batches_cat > 0 else 1
        batch = torch.randint(0, actual_n_batches_for_randint, (B,1))
        if n_batches_cat == 0: # If model n_batch is 0, set dummy batch index to 0
            batch = torch.zeros(B,1, dtype=torch.long)


        x = torch.poisson(torch.rand(B, G) * 5)
        y = torch.poisson(torch.rand(B, P) * 3)
        sample_idx = torch.randint(0, n_samples_cat, (B, 1))
        
        tensors = {
            REGISTRY_KEYS.X_KEY: x,
            REGISTRY_KEYS.PROTEIN_EXP_KEY: y,
            REGISTRY_KEYS.BATCH_KEY: batch,
            REGISTRY_KEYS.SAMPLE_KEY: sample_idx,
        }
        return tensors
    return _fake_batch