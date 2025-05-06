import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalLayerNorm(nn.Module):
    """
    Applies LayerNorm followed by condition-specific scale and shift.

    Equivalent to ConditionalNormalization(..., normalization_type='layer') in JAX.

    Parameters
    ----------
    n_features
        Number of input features.
    n_conditions
        Number of discrete condition labels (e.g., number of samples).
    eps
        Epsilon added for numerical stability.
    """
    def __init__(self, n_features: int, n_conditions: int, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.layer_norm = nn.LayerNorm(n_features, elementwise_affine=False, eps=eps)
        self.gamma = nn.Embedding(n_conditions, n_features)
        self.beta = nn.Embedding(n_conditions, n_features)
        nn.init.normal_(self.gamma.weight, mean=1.0, std=0.02)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Tensor of shape (batch_size, n_features).
        condition
            LongTensor of shape (batch_size,) or (batch_size, 1)
            Discrete condition index for each example in batch.
        """
        if condition.ndim == 2:
            condition = condition.squeeze(-1)
        x_norm = self.layer_norm(x)
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x_norm + beta

class ModalityInputBlock(nn.Module):
    """
    A single block: Linear -> ConditionalLayerNorm -> Activation.

    Used for processing modality-specific inputs.

    Parameters
    ----------
    in_features
        Input feature dimensionality.
    out_features
        Output feature dimensionality (typically `n_hidden`).
    n_conditions
        Number of sample-level conditions for ConditionalLayerNorm.
    activation
        Activation function (e.g., nn.GELU()).
    """
    def __init__(self, in_features: int, out_features: int, n_conditions: int, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.cond_norm = ConditionalLayerNorm(out_features, n_conditions)
        self.activation = activation

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_proj = self.linear(x)
        x_norm = self.cond_norm(x_proj, condition)
        return self.activation(x_norm)

class MLP(nn.Module):
    """
    Residual Multi-Layer Perceptron.

    Each block consists of:
    Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU.
    A skip connection adds the input of the block (or a projection of the
    original MLP input for the first block) to the block's core output.

    Parameters
    ----------
    in_dim
        Input dimension to the MLP.
    hidden_dim
        Dimension of hidden layers within blocks.
    out_dim
        Output dimension of the MLP (after the final linear layer).
    n_layers
        Number of residual blocks. Must be at least 1.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 1):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        self.blocks = nn.ModuleList()

        if in_dim != hidden_dim:
            self.input_residual_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.input_residual_proj = nn.Identity()

        for i in range(n_layers):
            layer_in_dim = in_dim if i == 0 else hidden_dim
            self.blocks.append(nn.Sequential(
                nn.Linear(layer_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ))
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_representation = x

        for i, block in enumerate(self.blocks):
            input_to_this_block = current_representation
            block_core_output = block(input_to_this_block)

            if i == 0:
                residual_for_addition = self.input_residual_proj(x)
            else:
                residual_for_addition = input_to_this_block
                assert input_to_this_block.shape[-1] == self.hidden_dim, \
                    "Input to non-first MLP block should be hidden_dim"

            if block_core_output.shape != residual_for_addition.shape:
                 raise AssertionError(
                     f"MLP residual shape mismatch at layer {i}: "
                     f"block_core_output: {block_core_output.shape} vs "
                     f"residual_for_addition: {residual_for_addition.shape}."
                 )
            current_representation = block_core_output + residual_for_addition
        
        return self.final(current_representation)

class AttentionBlock(nn.Module):
    """
    Cross-attention block inspired by MrVI.

    A query embedding attends to a key-value (kv) embedding.
    The process involves:
    1. Projecting query_embed to Q.
    2. Projecting kv_embed to K and V (K=V).
    3. MultiheadAttention(Q, K, V).
    4. Attended output -> pre_mlp.
    5. Concatenating original query_embed with pre_mlp output.
    6. Passing concatenated tensor through post_mlp.

    Parameters
    ----------
    query_dim
        Dimension of the query embedding.
    out_dim
        Output dimension of the block. If `use_map` is False, this is the
        combined dimension for mean and scale (e.g., 2 * latent_dim).
    outerprod_dim
        Dimension for Q, K, V projections in attention.
    n_heads
        Number of attention heads.
    dropout_rate
        Dropout rate for MultiheadAttention.
    n_hidden_mlp
        Hidden dimension for pre_mlp and post_mlp.
    n_layers_mlp
        Number of layers for post_mlp (pre_mlp is fixed to 1 layer).
    stop_gradients_mlp
        If True, parameters of pre_mlp and post_mlp are frozen (requires_grad=False).
    use_map
        If True, outputs a single tensor. If False, `out_dim` is expected to be
        2*actual_out, and output is (mean, scale_chunk).
    kv_dim
        Dimension of the key-value embedding. Defaults to `query_dim` if None.
    """
    def __init__(
        self,
        query_dim: int,
        out_dim: int,
        outerprod_dim: int = 16,
        n_heads: int = 2,
        dropout_rate: float = 0.0,
        n_hidden_mlp: int = 32,
        n_layers_mlp: int = 1,
        stop_gradients_mlp: bool = False,
        use_map: bool = True,
        kv_dim: int | None = None
    ):
        super().__init__()
        self.outerprod_dim = outerprod_dim
        self.use_map = use_map
        self.out_dim = out_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim or query_dim

        self.query_proj = nn.Linear(self.query_dim, outerprod_dim)
        self.kv_proj = nn.Linear(self.kv_dim, outerprod_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=outerprod_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.pre_mlp = MLP(
            in_dim=outerprod_dim,
            hidden_dim=n_hidden_mlp,
            out_dim=outerprod_dim,
            n_layers=1,
        )
        self.post_mlp = MLP(
            in_dim=self.query_dim + outerprod_dim,
            hidden_dim=n_hidden_mlp,
            out_dim=out_dim,
            n_layers=n_layers_mlp,
        )
        if stop_gradients_mlp:
            for param in self.pre_mlp.parameters():
                param.requires_grad = False
            for param in self.post_mlp.parameters():
                param.requires_grad = False

    def forward(self, query_embed: torch.Tensor, kv_embed: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        S_dim, B_dim = -1, -1
        original_ndim = query_embed.ndim

        if query_embed.ndim == 3:
            S_dim = query_embed.shape[0]
            B_dim = query_embed.shape[1]
            _query_embed_reshaped = query_embed.reshape(S_dim * B_dim, -1)
        elif query_embed.ndim == 2:
            B_dim = query_embed.shape[0]
            _query_embed_reshaped = query_embed
        else:
            raise ValueError(f"AttentionBlock: query_embed has unexpected ndim: {query_embed.ndim}, shape: {query_embed.shape}")
        
        assert _query_embed_reshaped.ndim == 2

        if kv_embed.ndim == 2:
            if S_dim != -1 :
                _kv_embed_expanded = kv_embed.unsqueeze(0).expand(S_dim, B_dim, -1)
                _kv_embed_reshaped = _kv_embed_expanded.reshape(S_dim * B_dim, -1)
            else:
                _kv_embed_reshaped = kv_embed
        elif kv_embed.ndim == 3 and S_dim != -1 and kv_embed.shape[0] == S_dim and kv_embed.shape[1] == B_dim:
             _kv_embed_reshaped = kv_embed.reshape(kv_embed.shape[0] * kv_embed.shape[1], -1)
        else:
            raise ValueError(f"AttentionBlock: kv_embed shape {kv_embed.shape} incompatible with query shape {query_embed.shape}")
        
        assert _kv_embed_reshaped.ndim == 2
        
        _original_query_for_concat = _query_embed_reshaped
        _forward_pass = self._core_forward(_query_embed_reshaped, _kv_embed_reshaped, _original_query_for_concat)

        if original_ndim == 3:
            if self.use_map:
                _forward_pass = _forward_pass.reshape(S_dim, B_dim, -1)
            else:
                mean, scale = _forward_pass
                mean = mean.reshape(S_dim, B_dim, -1)
                scale = scale.reshape(S_dim, B_dim, -1)
                _forward_pass = (mean, scale)
        return _forward_pass

    def _core_forward(self, query_embed_flat: torch.Tensor, kv_embed_flat: torch.Tensor, original_query_flat: torch.Tensor):
        _Q_projected = self.query_proj(query_embed_flat)
        _K_projected = self.kv_proj(kv_embed_flat)
        Q = _Q_projected.unsqueeze(1)
        K = _K_projected.unsqueeze(1)
        V = K
        attn_output, _ = self.attn(Q, K, V)
        attn_output = attn_output.squeeze(1)
        eps = self.pre_mlp(attn_output)
        combined = torch.cat([original_query_flat, eps], dim=-1)
        residual_out = self.post_mlp(combined)
        if self.use_map:
            out = residual_out
        else:
            mean, scale_chunk = torch.chunk(residual_out, 2, dim=-1)
            out = (mean, scale_chunk)
        return out