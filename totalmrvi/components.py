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
    n_channels
        The number of channels per embedding dimension computed by attention.
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
    kv_dim
        Dimension of the key-value embedding. Defaults to `query_dim` if None.
    """
    def __init__(
        self,
        query_dim: int,
        out_dim: int,
        outerprod_dim: int = 16,
        n_channels: int = 4,
        n_heads: int = 2,
        dropout_rate: float = 0.0,
        n_hidden: int = 32,
        n_layers: int = 1,
        stop_gradients_mlp: bool = False,
        kv_dim: int | None = None
    ):
        super().__init__()
        self.outerprod_dim = outerprod_dim
        self.n_channels = n_channels
        self.out_dim = out_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim or query_dim
        self.stop_gradients_mlp = stop_gradients_mlp

        self.query_proj = nn.Linear(self.query_dim, outerprod_dim)
        self.kv_proj = nn.Linear(self.kv_dim, outerprod_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=n_channels * n_heads,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attn_output_proj = nn.Linear(n_channels * n_heads, n_channels)
        self.pre_mlp = MLP(
            in_dim=outerprod_dim * n_channels,
            hidden_dim=n_hidden,
            out_dim=outerprod_dim,
            n_layers=1,
        )
        self.post_mlp = MLP(
            in_dim=self.query_dim + outerprod_dim,
            hidden_dim=n_hidden,
            out_dim=out_dim,
            n_layers=n_layers,
        )

    def forward(self, query_embed: torch.Tensor, kv_embed: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        S_dim, B_dim = -1, -1

        assert query_embed.ndim == kv_embed.ndim, "AttentionBlock: query_embed and kv_embed must have the same number of dimensions"
        if query_embed.ndim == 3:
            S_dim = query_embed.shape[0]
            B_dim = query_embed.shape[1]
            assert kv_embed.shape[0] == S_dim and kv_embed.shape[1] == B_dim, "AttentionBlock: query_embed and kv_embed must have the same batch and sample sizes"
            query_embed = query_embed.reshape(S_dim * B_dim, -1)
            kv_embed = kv_embed.reshape(S_dim * B_dim, -1)
        elif query_embed.ndim == 2:
            B_dim = query_embed.shape[0]
            assert kv_embed.shape[0] == B_dim, "AttentionBlock: query_embed and kv_embed must have the same batch size"
        else:
            raise ValueError(f"AttentionBlock: query_embed has unexpected ndim: {query_embed.ndim}, shape: {query_embed.shape}")
        
        query_embed_stop = query_embed if not self.stop_gradients_mlp else query_embed.detach()

        Q = self.query_proj(query_embed_stop).unsqueeze(-1) # shape (mc_samples*batch_size, outerprod_dim, 1)
        K = self.kv_proj(kv_embed).unsqueeze(-1) # shape (mc_samples*batch_size, outerprod_dim, 1)

        eps, _ = self.attn(Q, K, K) # shape (mc_samples*batch_size, outerprod_dim, n_channels*n_heads)
        eps = self.attn_output_proj(eps) # shape (mc_samples*batch_size, outerprod_dim, n_channels)

        eps = eps.reshape(-1, self.outerprod_dim * self.n_channels) # shape (mc_samples*batch_size, outerprod_dim*n_channels)
        eps = self.pre_mlp(eps) # shape (mc_samples*batch_size, outerprod_dim)

        combined = torch.cat([query_embed_stop, eps], dim=-1)
        residual_out = self.post_mlp(combined) # shape (mc_samples*batch_size, out_dim)
        
        if S_dim != -1:
            residual_out = residual_out.reshape(S_dim, B_dim, -1)
        return residual_out