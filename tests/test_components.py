import torch
import torch.nn as nn 
import pytest 

from totalmrvi.components import (
    ConditionalLayerNorm,
    ModalityInputBlock,
    MLP,
    AttentionBlock,
)

# Test functions for ConditionalLayerNorm (from Cell 544, 545)
def test_conditional_layer_norm():
    torch.manual_seed(0)
    batch_size = 4
    n_features = 8
    n_conditions = 3
    x = torch.randn(batch_size, n_features)
    condition = torch.tensor([0, 1, 2, 0])
    layer = ConditionalLayerNorm(n_features=n_features, n_conditions=n_conditions)
    out = layer(x, condition)
    assert out.shape == (batch_size, n_features), "Incorrect output shape"
    assert not torch.isnan(out).any(), "NaNs detected in output"
    print("✅ ConditionalLayerNorm passed test!")

def test_conditional_layer_norm_correctness():
    torch.manual_seed(0)
    batch_size = 2
    n_features = 4
    n_conditions = 2
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    condition = torch.tensor([0, 1], dtype=torch.long)
    layer = ConditionalLayerNorm(n_features=n_features, n_conditions=n_conditions)
    with torch.no_grad():
        layer.gamma.weight.data = torch.tensor([[1.0]*4, [2.0]*4])
        layer.beta.weight.data = torch.tensor([[0.0]*4, [1.0]*4])
    out = layer(x, condition)
    x0_norm = (x[0] - x[0].mean()) / x[0].std(unbiased=False)
    x1_norm = (x[1] - x[1].mean()) / x[1].std(unbiased=False)
    expected = torch.stack([1.0 * x0_norm + 0.0, 2.0 * x1_norm + 1.0])
    assert torch.allclose(out, expected, atol=1e-5), "Output does not match expected"
    print("✅ ConditionalLayerNorm passed correctness test!")

# Test function for ModalityInputBlock (from Cell 547)
def test_modality_input_block():
    torch.manual_seed(0)
    batch_size = 2
    in_features = 3
    out_features = 4
    n_conditions = 2
    eps = 1e-5
    x = torch.tensor([[1.0, 2.0, 3.0],[3.0, 2.0, 1.0]], dtype=torch.float32)
    condition = torch.tensor([0, 1], dtype=torch.long)
    block = ModalityInputBlock(in_features, out_features, n_conditions, activation=nn.Identity())
    with torch.no_grad():
        block.linear.weight.data.fill_(1.0)
        block.linear.bias.data.fill_(0.0)
        block.cond_norm.gamma.weight.data = torch.tensor([[1.0]*out_features, [2.0]*out_features])
        block.cond_norm.beta.weight.data = torch.tensor([[0.0]*out_features, [1.0]*out_features])
    out = block(x, condition)
    linear_out_0 = torch.tensor([6.0]*out_features)
    linear_out_1 = torch.tensor([6.0]*out_features)
    norm_0 = (linear_out_0 - linear_out_0.mean()) / (linear_out_0.std(unbiased=False) + eps)
    norm_1 = (linear_out_1 - linear_out_1.mean()) / (linear_out_1.std(unbiased=False) + eps)
    expected = torch.stack([norm_0 * 1.0 + 0.0, norm_1 * 2.0 + 1.0])
    assert torch.allclose(out, expected, atol=1e-5), "ModalityInputBlock output mismatch"
    print("✅ ModalityInputBlock passed test!")

# Test functions for MLP (from Cell 595 and new test)
def test_mlp_basic():
    mlp = MLP(in_dim=16, hidden_dim=32, out_dim=8, n_layers=1)
    x = torch.randn(10, 16)
    out = mlp(x)
    assert out.shape == (10, 8), f"Expected output shape (10, 8), got {out.shape}"
    assert not torch.isnan(out).any(), "MLP output contains NaNs"
    print("✅ MLP test passed!")

def test_mlp_residual_projection():
    mlp = MLP(in_dim=24, hidden_dim=16, out_dim=8, n_layers=1)
    x = torch.randn(10, 24)
    out = mlp(x)
    assert out.shape == (10, 8), "Residual projection in MLP failed"
    print("✅ MLP residual projection test passed!")

def test_mlp_multi_layer():
    mlp = MLP(in_dim=16, hidden_dim=32, out_dim=8, n_layers=3)
    x = torch.randn(10, 16)
    out = mlp(x)
    assert out.shape == (10, 8), f"Expected output shape (10, 8) for n_layers=3, got {out.shape}"
    assert not torch.isnan(out).any(), "MLP output contains NaNs for n_layers=3"
    print("✅ MLP multi-layer test passed!")
    
def test_mlp_multi_layer_residuals():
    torch.manual_seed(0)
    B, D_in, D_hidden, D_out = 2, 4, 8, 3
    mlp1 = MLP(D_in, D_hidden, D_out, n_layers=2)
    x1 = torch.randn(B, D_in)
    h0_block_input = x1
    h0_seq_out = mlp1.blocks[0](h0_block_input)
    h0_residual = mlp1.input_residual_proj(x1)
    h0_sum = h0_seq_out + h0_residual
    h1_block_input = h0_sum
    h1_seq_out = mlp1.blocks[1](h1_block_input)
    h1_residual = h1_block_input
    h1_sum = h1_seq_out + h1_residual
    expected_out1 = mlp1.final(h1_sum)
    actual_out1 = mlp1(x1)
    assert torch.allclose(actual_out1, expected_out1), "MLP n_layers=2 (D_in != D_hidden) output mismatch"
    mlp2 = MLP(D_hidden, D_hidden, D_out, n_layers=2)
    x2 = torch.randn(B, D_hidden)
    h0_block_input_c2 = x2
    h0_seq_out_c2 = mlp2.blocks[0](h0_block_input_c2)
    h0_residual_c2 = mlp2.input_residual_proj(x2)
    assert isinstance(mlp2.input_residual_proj, nn.Identity)
    h0_sum_c2 = h0_seq_out_c2 + h0_residual_c2
    h1_block_input_c2 = h0_sum_c2
    h1_seq_out_c2 = mlp2.blocks[1](h1_block_input_c2)
    h1_residual_c2 = h1_block_input_c2
    h1_sum_c2 = h1_seq_out_c2 + h1_residual_c2
    expected_out2 = mlp2.final(h1_sum_c2)
    actual_out2 = mlp2(x2)
    assert torch.allclose(actual_out2, expected_out2), "MLP n_layers=2 (D_in == D_hidden) output mismatch"
    print("✅ MLP multi-layer residual logic test passed!")

# Test functions for AttentionBlock (from Cell 595)
def test_attention_block_shapes():
    block = AttentionBlock(query_dim=32, out_dim=16, outerprod_dim=8, n_heads=2, n_hidden_mlp=64, n_layers_mlp=1)
    q = torch.randn(12, 32)
    kv = torch.randn(12, 32)
    out = block(q, kv)
    assert out.shape == (12, 16), f"Expected output shape (12, 16), got {out.shape}"
    assert not torch.isnan(out).any(), "AttentionBlock output contains NaNs"
    print("✅ AttentionBlock shape test passed!")

def test_attention_block_grad_stop():
    block = AttentionBlock(query_dim=32, out_dim=16, stop_gradients_mlp=True)
    for name, param in block.pre_mlp.named_parameters():
        assert not param.requires_grad
    for name, param in block.post_mlp.named_parameters():
        assert not param.requires_grad
    assert block.query_proj.weight.requires_grad
    assert block.kv_proj.weight.requires_grad
    q = torch.randn(4, 32, requires_grad=True)
    kv = torch.randn(4, 32)
    out = block(q, kv)
    assert out.requires_grad
    print("✅ AttentionBlock stop_gradients_mlp test passed (checks requires_grad flags)!")

def test_attention_block_use_map_false():
    block = AttentionBlock(query_dim=32, out_dim=20, use_map=False)
    q = torch.randn(6, 32)
    kv = torch.randn(6, 32)
    mean, scale = block(q, kv)
    assert mean.shape == (6, 10)
    assert scale.shape == (6, 10)
    print("✅ AttentionBlock use_map=False test passed!")

def test_attention_block_s_dimension():
    S, B, D_q, D_kv, D_out = 3, 4, 32, 16, 8
    block_map_true = AttentionBlock(query_dim=D_q, kv_dim=D_kv, out_dim=D_out, use_map=True)
    block_map_false = AttentionBlock(query_dim=D_q, kv_dim=D_kv, out_dim=2*D_out, use_map=False)
    query_sb = torch.randn(S, B, D_q)
    kv_b = torch.randn(B, D_kv)
    out_map_true = block_map_true(query_sb, kv_b)
    assert out_map_true.shape == (S, B, D_out)
    print("✅ AttentionBlock S-dim test (query 3D, kv 2D, map=True) passed!")
    mean, scale = block_map_false(query_sb, kv_b)
    assert mean.shape == (S, B, D_out)
    assert scale.shape == (S, B, D_out)
    print("✅ AttentionBlock S-dim test (query 3D, kv 2D, map=False) passed!")
    kv_sb = torch.randn(S, B, D_kv)
    out_map_true_kv_sb = block_map_true(query_sb, kv_sb)
    assert out_map_true_kv_sb.shape == (S, B, D_out)
    print("✅ AttentionBlock S-dim test (query 3D, kv 3D, map=True) passed!")
    mean_kv_sb, scale_kv_sb = block_map_false(query_sb, kv_sb)
    assert mean_kv_sb.shape == (S, B, D_out)
    assert scale_kv_sb.shape == (S, B, D_out)
    print("✅ AttentionBlock S-dim test (query 3D, kv 3D, map=False) passed!")