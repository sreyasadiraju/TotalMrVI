import pytest
import torch
from scvi import REGISTRY_KEYS

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