import torch
import numpy as np
from scvi import REGISTRY_KEYS # If using the fixture
from totalmrvi.module import TOTALMRVAE # Main module to test

# test_totalmrvae_loss_scalar_and_shapes (from Cell 611)
def test_totalmrvae_loss_scalar_and_shapes(fake_data_fixture): # Use fixture
    N_SAMPLES = 5; N_BATCHES = 4; G = 60; P = 15; LATENT_DIM = 10; LATENT_U_DIM = 8
    mod = TOTALMRVAE(
        n_input_genes=G, n_input_proteins=P, n_sample=N_SAMPLES, n_batch=N_BATCHES, n_labels=1,
        n_latent=LATENT_DIM, n_latent_u=LATENT_U_DIM, dispersion_rna="gene-cell", dispersion_pro="protein",
        qz_kwargs={"use_map": True, "n_hidden": 20, "n_layers": 1}
    )
    scenarios = [
        {"name": "use_mean", "mc_samples": 0, "use_mean": True, "expected_S_dim_in_z": False},
        {"name": "mc_samples=0_no_mean", "mc_samples": 0, "use_mean": False, "expected_S_dim_in_z": False},
        {"name": "mc_samples=1", "mc_samples": 1, "use_mean": False, "expected_S_dim_in_z": True},
        {"name": "mc_samples=2", "mc_samples": 2, "use_mean": False, "expected_S_dim_in_z": True},
    ]
    for B_test in [1, 16]:
        print(f"\n--- Testing with Batch Size: {B_test} ---")
        tensors = fake_data_fixture(B=B_test, G=G, P=P, n_samples_cat=N_SAMPLES, n_batches_cat=N_BATCHES)
        inf_inputs = mod._get_inference_input(tensors)
        for scen in scenarios:
            print(f"  Scenario: {scen['name']}")
            inf_out = mod.inference(**inf_inputs, mc_samples=scen["mc_samples"], use_mean=scen["use_mean"])
            if scen["expected_S_dim_in_z"]:
                assert inf_out["z"].ndim == 3 and inf_out["z"].shape[0] == scen["mc_samples"] and \
                       inf_out["z"].shape[1] == B_test and inf_out["z"].shape[2] == LATENT_DIM
            else:
                assert inf_out["z"].ndim == 2 and inf_out["z"].shape[0] == B_test and \
                       inf_out["z"].shape[1] == LATENT_DIM
            if scen["expected_S_dim_in_z"] and scen["mc_samples"] > 0:
                assert inf_out["library"].ndim == 3 and inf_out["library"].shape[0] == scen["mc_samples"]
            else:
                assert inf_out["library"].ndim == 2 and inf_out["library"].shape[1] == 1
            gen_in  = mod._get_generative_input(tensors, inf_out)
            gen_out = mod.generative(**gen_in)
            expected_gen_shape_rna = (scen["mc_samples"], B_test, G) if scen["expected_S_dim_in_z"] else (B_test, G)
            expected_gen_shape_pro = (scen["mc_samples"], B_test, P) if scen["expected_S_dim_in_z"] else (B_test, P)
            assert gen_out["px_rate"].shape == expected_gen_shape_rna
            assert gen_out["py_rate_back"].shape == expected_gen_shape_pro
            loss_out = mod.loss(tensors, inf_out, gen_out)
            assert torch.isfinite(loss_out.loss)
            print(f"    ✅ Loss finite for {scen['name']}")
            for k, v_loss in loss_out.reconstruction_loss.items(): assert v_loss.shape == (B_test,)
            for k, v_loss in loss_out.kl_local.items(): assert v_loss.shape == (B_test,)
            print(f"    ✅ Logged loss shapes correct for {scen['name']}")

# test_totalmrvae_gradients (from Cell 611)
def test_totalmrvae_gradients(fake_data_fixture): # Use fixture
    N_SAMPLES=3; N_BATCHES=2; G=20; P=10
    for stop_grads_mlp_test_val in [True, False]:
        print(f"\n--- Testing Gradients with stop_gradients_mlp={stop_grads_mlp_test_val} ---")
        qz_kwargs_test = {"use_map": False, "stop_gradients_mlp": stop_grads_mlp_test_val}
        mod = TOTALMRVAE(
            n_input_genes=G, n_input_proteins=P, n_sample=N_SAMPLES, n_batch=N_BATCHES, n_labels=1,
            n_latent=8, n_latent_u=6, dispersion_rna="gene", dispersion_pro="protein-cell", qz_kwargs=qz_kwargs_test
        )
        tensors = fake_data_fixture(B=4, G=G, P=P, n_samples_cat=N_SAMPLES, n_batches_cat=N_BATCHES)
        inf_out = mod.inference(**mod._get_inference_input(tensors), mc_samples=1, use_mean=False)
        gen_out = mod.generative(**mod._get_generative_input(tensors, inf_out))
        loss = mod.loss(tensors, inf_out, gen_out).loss
        mod.zero_grad(); loss.backward()
        assert mod.qu.x_proj.linear.weight.grad is not None
        assert mod.qz.attention.query_proj.weight.grad is not None
        post_mlp_grad = mod.qz.attention.post_mlp.final.weight.grad
        if stop_grads_mlp_test_val: assert post_mlp_grad is None
        else: assert post_mlp_grad is not None
        print(f"    ✅ qz.attention.post_mlp grad check passed for stop_mlp={stop_grads_mlp_test_val}")
    print("\n✅ backward pass produces expected gradients for key parameters")


# test_totalmrvae_cf_sample (from Cell 611)
def test_totalmrvae_cf_sample(fake_data_fixture): # Use fixture
    torch.manual_seed(123); N_SAMPLES=5; N_BATCHES=2; G,P,D,Du = 20,10,8,6
    mod = TOTALMRVAE(G,P,N_SAMPLES,N_BATCHES,1,D,Du, qz_kwargs={"use_map":False})
    tensors = fake_data_fixture(B=4,G=G,P=P,n_samples_cat=N_SAMPLES,n_batches_cat=N_BATCHES)
    inf_inputs = mod._get_inference_input(tensors)
    sample_idx_orig = inf_inputs["sample_index"]
    qu = mod.qu(inf_inputs["x"],inf_inputs["y"],sample_idx_orig)
    u_sample = qu.rsample((1,))
    z_base_orig,eps_orig_params = mod.qz(u_sample,sample_idx_orig)
    cf_sample_idx = (sample_idx_orig+1)%N_SAMPLES
    z_base_cf,eps_cf_params = mod.qz(u_sample,cf_sample_idx)
    assert torch.allclose(z_base_orig,z_base_cf)
    assert isinstance(eps_orig_params,tuple) and isinstance(eps_cf_params,tuple)
    assert not torch.allclose(eps_orig_params[0],eps_cf_params[0])
    mod_map = TOTALMRVAE(G,P,N_SAMPLES,N_BATCHES,1,D,Du, qz_kwargs={"use_map":True})
    z_base_orig_map,eps_orig_map = mod_map.qz(u_sample,sample_idx_orig)
    z_base_cf_map,eps_cf_map = mod_map.qz(u_sample,cf_sample_idx)
    assert torch.allclose(z_base_orig_map,z_base_cf_map)
    assert not torch.allclose(eps_orig_map,eps_cf_map)
    print("✅ TOTALMRVAE counterfactual sample test passed!")

# test_totalmrvae_prior_init (from Cell 611)
def test_totalmrvae_prior_init(fake_data_fixture): # Use fixture
    N_SAMPLES=3; N_BATCHES_MULTI=2; N_BATCHES_SINGLE=1; N_BATCHES_ZERO=0; G,P,D,Du = 20,10,8,6
    mod_default = TOTALMRVAE(G,P,N_SAMPLES,N_BATCHES_MULTI,1,D,Du)
    assert mod_default.prior_logbeta_loc.shape==(P,N_BATCHES_MULTI)
    prior_mean_mb = np.random.randn(P,N_BATCHES_MULTI); prior_scale_mb = np.random.rand(P,N_BATCHES_MULTI)*0.5+0.1
    mod_provided_mb = TOTALMRVAE(G,P,N_SAMPLES,N_BATCHES_MULTI,1,D,Du,protein_background_prior_mean=prior_mean_mb,protein_background_prior_scale=prior_scale_mb)
    assert torch.allclose(mod_provided_mb.prior_logbeta_loc,torch.from_numpy(prior_mean_mb).float())
    mod_nb0_default = TOTALMRVAE(G,P,N_SAMPLES,n_batch=N_BATCHES_ZERO,n_labels=1,n_latent=D,n_latent_u=Du)
    assert mod_nb0_default.prior_logbeta_loc.shape==(P,)
    prior_mean_nb0=np.random.randn(P); prior_scale_nb0=np.random.rand(P)*0.5+0.1
    mod_nb0_provided = TOTALMRVAE(G,P,N_SAMPLES,n_batch=N_BATCHES_ZERO,n_labels=1,n_latent=D,n_latent_u=Du,protein_background_prior_mean=prior_mean_nb0,protein_background_prior_scale=prior_scale_nb0)
    assert mod_nb0_provided.prior_logbeta_loc.shape==(P,)
    if N_BATCHES_ZERO == 0:
        mod_for_loss_nb0 = TOTALMRVAE(G,P,N_SAMPLES,n_batch=0,n_labels=1,n_latent=D,n_latent_u=Du)
        tensors_nb0 = fake_data_fixture(B=2, G=G, P=P, n_samples_cat=N_SAMPLES, n_batches_cat=N_BATCHES_ZERO) # Use fixture
        inf_out_nb0 = mod_for_loss_nb0.inference(**mod_for_loss_nb0._get_inference_input(tensors_nb0))
        gen_out_nb0 = mod_for_loss_nb0.generative(**mod_for_loss_nb0._get_generative_input(tensors_nb0,inf_out_nb0))
        loss_out_nb0 = mod_for_loss_nb0.loss(tensors_nb0,inf_out_nb0,gen_out_nb0)
        assert torch.isfinite(loss_out_nb0.loss)
        print("    ✅ Loss calculation successful for n_batch=0")
    print("✅ TOTALMRVAE prior initialization tests passed!")