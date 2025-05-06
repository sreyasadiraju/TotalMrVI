import torch
from torch.distributions import Normal
from totalmrvi.encoders import EncoderXYU, EncoderUZ, BackgroundProteinEncoder
from totalmrvi.components import ModalityInputBlock, ConditionalLayerNorm # Needed if ModalityInputBlock is tested here or implicitly

# Test functions for EncoderXYU (from Cell 550)
def test_encoder_xyu():
    torch.manual_seed(0)
    batch_size = 3; n_genes = 5; n_proteins = 2; n_latent = 4; n_hidden = 8; n_sample = 3
    x = torch.rand(batch_size, n_genes); y = torch.rand(batch_size, n_proteins)
    sample_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    encoder = EncoderXYU(n_genes, n_proteins, n_latent, n_hidden, n_sample)
    dist = encoder(x, y, sample_idx)
    assert isinstance(dist, Normal)
    assert dist.loc.shape == (batch_size, n_latent)
    print("✅ EncoderXYU passed test!")

def test_encoder_xyu_conditioning_and_layers():
    torch.manual_seed(1)
    batch_size = 4; n_genes, n_proteins, n_latent, n_hidden, n_sample = 5,2,4,8,3
    x = torch.rand(batch_size, n_genes); y = torch.rand(batch_size, n_proteins)
    sample_idx = torch.tensor([0,1,2,0], dtype=torch.long)
    encoder_l2 = EncoderXYU(n_genes, n_proteins, n_latent, n_hidden, n_sample, n_layers=2)
    dist_l2 = encoder_l2(x, y, sample_idx)
    assert dist_l2.loc.shape == (batch_size, n_latent)
    encoder_l1 = EncoderXYU(n_genes, n_proteins, n_latent, n_hidden, n_sample, n_layers=1)
    dist1 = encoder_l1(x,y,sample_idx)
    sample_idx_alt = torch.tensor([1,2,0,1],dtype=torch.long)
    dist2 = encoder_l1(x,y,sample_idx_alt)
    assert not torch.allclose(dist1.loc, dist2.loc)
    print("✅ EncoderXYU passed conditioning and n_layers>1 tests!")

# Test functions for EncoderUZ (from Cell 556 and new tests)
def test_encoder_uz_map_false():
    encoder = EncoderUZ(n_latent=10, n_sample=4, n_latent_u=None, use_map=False)
    u = torch.randn(5,10); sample_idx = torch.randint(0,4,(5,))
    z_base, (mu,scale) = encoder(u,sample_idx)
    assert z_base.shape == (5,10) and mu.shape == (5,10) and scale.shape == (5,10)
    print("✅ EncoderUZ use_map=False test passed!")

def test_encoder_uz_map_true_behavior():
    torch.manual_seed(0)
    encoder = EncoderUZ(n_latent=12,n_sample=5,n_latent_u=None,use_map=True,n_latent_sample=8,n_hidden=16,n_layers=1)
    u = torch.randn(6,12); sample_idx=torch.tensor([0,1,2,3,4,0])
    z_base,eps = encoder(u,sample_idx)
    assert z_base.shape == (6,12) and eps.shape == (6,12)
    assert torch.allclose(z_base,u)
    _,eps2 = encoder(u, (sample_idx+1)%5)
    assert not torch.allclose(eps,eps2)
    print("✅ EncoderUZ use_map=True test passed!")

def test_encoder_uz_map_true_with_projection():
    torch.manual_seed(0)
    encoder = EncoderUZ(n_latent=12,n_sample=5,n_latent_u=8,use_map=True,n_latent_sample=6,n_hidden=16,n_layers=1)
    u = torch.randn(6,8); sample_idx=torch.tensor([0,1,2,3,4,0])
    z_base,eps = encoder(u,sample_idx)
    assert z_base.shape == (6,12) and eps.shape == (6,12)
    _,eps_alt = encoder(u, (sample_idx+2)%5)
    assert not torch.allclose(eps,eps_alt)
    print("✅ EncoderUZ map=True test with projection passed!")

def test_encoder_uz_independent_across_cells():
    torch.manual_seed(0)
    encoder = EncoderUZ(n_latent=6,n_sample=3,n_latent_u=None,use_map=True,n_latent_sample=4,n_hidden=8,n_layers=1)
    u_base = torch.randn(1,6); u = u_base.repeat(3,1); sample_idx = torch.tensor([0,0,0])
    z_base_orig, _ = encoder(u,sample_idx)
    u_modified = u.clone(); u_modified[1]+=torch.randn_like(u_modified[1])*0.1
    z_base_mod, _ = encoder(u_modified,sample_idx)
    assert torch.allclose(z_base_orig[0],z_base_mod[0],atol=1e-5)
    assert torch.allclose(z_base_orig[2],z_base_mod[2],atol=1e-5)
    assert not torch.allclose(z_base_orig[1],z_base_mod[1],atol=1e-4)
    print("✅ EncoderUZ is independent across cells in a batch")

def test_encoder_uz_s_dimension():
    S,B,D_u,D_z,N_sample,D_sample = 3,4,10,12,5,8
    encoder = EncoderUZ(n_latent=D_z,n_sample=N_sample,n_latent_u=D_u,n_latent_sample=D_sample,use_map=True)
    encoder_nomap = EncoderUZ(n_latent=D_z,n_sample=N_sample,n_latent_u=D_u,n_latent_sample=D_sample,use_map=False)
    u_sb = torch.randn(S,B,D_u); sample_idx_b = torch.randint(0,N_sample,(B,))
    z_base,eps = encoder(u_sb,sample_idx_b)
    assert z_base.shape==(S,B,D_z) and eps.shape==(S,B,D_z)
    z_base_nm,(eps_mean,eps_scale) = encoder_nomap(u_sb,sample_idx_b)
    assert z_base_nm.shape==(S,B,D_z) and eps_mean.shape==(S,B,D_z) and eps_scale.shape==(S,B,D_z)
    print("✅ EncoderUZ S-dimension propagation test passed!")

def test_encoder_uz_stop_gradients():
    encoder = EncoderUZ(n_latent=10, n_sample=4, n_latent_u=None, use_map=True, stop_gradients=True)
    u = torch.randn(5, 10, requires_grad=True)
    sample_idx = torch.randint(0, 4, (5,))
    z_base, eps = encoder(u, sample_idx)
    assert not z_base.requires_grad, "z_base should not require grad when stop_gradients=True"
    assert eps.requires_grad, "eps tensor SHOULD require grad (due to AttentionBlock params)"
    encoder_nomap = EncoderUZ(n_latent=10, n_sample=4, n_latent_u=None, use_map=False, stop_gradients=True)
    z_base_nm, eps_nm_tuple = encoder_nomap(u, sample_idx)
    assert not z_base_nm.requires_grad
    assert isinstance(eps_nm_tuple, tuple)
    assert eps_nm_tuple[0].requires_grad
    assert eps_nm_tuple[1].requires_grad
    print("✅ EncoderUZ stop_gradients test passed!")


# Test functions for BackgroundProteinEncoder (from Cell 575)
def test_background_protein_encoder_shapes():
    encoder = BackgroundProteinEncoder(n_latent=10,n_batch=3,n_proteins=7)
    z = torch.randn(5,10); batch_idx = torch.randint(0,3,(5,))
    dist = encoder(z,batch_idx)
    assert isinstance(dist,Normal) and dist.loc.shape==(5,7) and dist.scale.shape==(5,7)
    print("✅ BackgroundProteinEncoder shape test passed!")

def test_background_protein_encoder_batch_effect():
    encoder = BackgroundProteinEncoder(n_latent=10,n_batch=3,n_proteins=7)
    z = torch.randn(5,10); batch_idx = torch.randint(0,3,(5,))
    dist1 = encoder(z,batch_idx); dist2 = encoder(z,(batch_idx+1)%3)
    assert not torch.allclose(dist1.loc,dist2.loc)
    print("✅ BackgroundProteinEncoder batch effect test passed!")

def test_background_protein_encoder_s_dimension():
    S,B,D_latent,N_batch,P = 3,5,10,3,7
    encoder = BackgroundProteinEncoder(n_latent=D_latent,n_batch=N_batch,n_proteins=P)
    z_sb = torch.randn(S,B,D_latent); batch_idx_b = torch.randint(0,N_batch,(B,))
    dist = encoder(z_sb,batch_idx_b)
    assert isinstance(dist,Normal) and dist.loc.shape==(S,B,P) and dist.scale.shape==(S,B,P)
    print("✅ BackgroundProteinEncoder S-dimension test passed!")