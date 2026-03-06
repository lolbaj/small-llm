"""
MoE Transformer architecture implementation (RoPE, GQA, SwiGLU, RMSNorm).
"""

import math
from typing import Tuple, List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from config import SmallLLMConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Used in LLaMA/Mistral/DeepSeek for its computational efficiency
    and lack of bias term/mean centering.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x_in):
        """Helper to compute norm."""
        return x_in * torch.rsqrt(x_in.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x_in):
        """Forward pass with float cast for precision stability."""
        output = self._norm(x_in.float()).type_as(x_in)
        return output * self.weight


class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Embeds relative positional information into the query/key projections
    by rotating pairs of coordinates.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Precompute frequency parameters
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x_in: torch.Tensor, seq_len: int):
        """Computes RoPE cosine and sine components."""
        t_pos = torch.arange(seq_len, device=x_in.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t_pos, self.inv_freq)
        # Combine sine/cosine
        emb = torch.cat((freqs, freqs), dim=-1)
        # Cos/Sin embeddings: (seq_len, 1, 1, head_dim)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def apply_rotary_emb(x_in, cos, sin):
    """Applies precomputed RoPE to query/key tensors."""
    # Split features into two halves
    half_len = x_in.shape[-1] // 2
    x1 = x_in[..., :half_len]
    x2 = x_in[..., half_len:]
    # Slice cos/sin to match the half-dimension
    cos_half = cos[..., :half_len]
    sin_half = sin[..., :half_len]
    # Rotate: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    return torch.cat(
        (x1 * cos_half - x2 * sin_half, x1 * sin_half + x2 * cos_half), dim=-1
    )


# pylint: disable=too-many-instance-attributes
class GQAAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with KV Cache support.
    """

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            config.d_model, config.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.d_model, config.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def forward(
        self,
        x_in: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Attention forward pass."""
        b_sz, s_len, _ = x_in.shape
        q_proj, k_proj, v_proj = self.wq(x_in), self.wk(x_in), self.wv(x_in)

        q_proj = q_proj.view(b_sz, s_len, self.n_heads, self.head_dim)
        k_proj = k_proj.view(b_sz, s_len, self.n_kv_heads, self.head_dim)
        v_proj = v_proj.view(b_sz, s_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q_proj = apply_rotary_emb(q_proj, cos, sin)
        k_proj = apply_rotary_emb(k_proj, cos, sin)

        # KV Caching logic
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k_proj = torch.cat([prev_k, k_proj], dim=1)
            v_proj = torch.cat([prev_v, v_proj], dim=1)

        new_kv_cache = (k_proj, v_proj)

        # Repeat K, V heads for GQA compatibility
        k_rep = k_proj.repeat_interleave(self.n_rep, dim=2)
        v_rep = v_proj.repeat_interleave(self.n_rep, dim=2)

        # Scaled dot-product attention
        q_proj = q_proj.transpose(1, 2)
        k_rep = k_rep.transpose(1, 2)
        v_rep = v_rep.transpose(1, 2)

        scores = torch.matmul(q_proj, k_rep.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        if mask is not None:
            # Mask must be adjusted for the new sequence length if using cache
            scores = scores + mask[:, :, -q_proj.shape[2] :, :]

        attn = F.softmax(scores.float(), dim=-1).type_as(q_proj)
        out = torch.matmul(attn, v_rep)
        out = out.transpose(1, 2).contiguous().view(b_sz, s_len, -1)
        return self.wo(out), new_kv_cache


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: Swish(xW + b) * (xV + c).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x_in):
        """Activation forward pass."""
        return self.w2(F.silu(self.w1(x_in)) * self.w3(x_in))


class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts Layer with auxiliary loss and shared expert.
    """

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.use_shared = config.use_shared_expert

        # Router: projects d_model to expert logits
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Experts
        hidden_dim = 4 * config.d_model // 3  # Typical SwiGLU scaling
        self.experts = nn.ModuleList(
            [SwiGLU(config.d_model, hidden_dim) for _ in range(config.n_experts)]
        )

        # Shared expert path (always active)
        if self.use_shared:
            self.shared_expert = SwiGLU(config.d_model, hidden_dim)

    # pylint: disable=too-many-locals
    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """MoE forward pass with top-k routing and capacity limiting."""
        b_sz, s_len, d_dim = x_in.shape
        x_flat = x_in.view(-1, d_dim)
        num_tokens = b_sz * s_len

        # Router logits & softmax
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Re-normalize

        # Output accumulation
        moe_out = torch.zeros_like(x_flat)

        # Expert utilization tracking
        expert_counts = torch.zeros(self.n_experts, device=x_in.device)
        for idx in range(self.top_k):
            expert_counts.scatter_add_(
                0, indices[:, idx], torch.ones_like(indices[:, idx], dtype=torch.float)
            )

        utilization = (expert_counts > 0).float().mean().item() * 100

        # Load balancing auxiliary loss
        fraction = expert_counts / num_tokens
        mean_prob = router_probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(mean_prob * fraction)

        # Expert processing with capacity factor
        expert_capacity = int(
            (num_tokens * self.top_k / self.n_experts) * self.capacity_factor
        )

        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                if mask.sum() > expert_capacity:
                    mask_indices = torch.where(mask)[0][:expert_capacity]
                    mask = torch.zeros_like(mask)
                    mask[mask_indices] = True

                expert_mask = indices == i
                token_weights = (weights * expert_mask.float()).sum(
                    dim=-1, keepdim=True
                )
                moe_out[mask] += token_weights[mask] * expert(x_flat[mask])

        final_out = moe_out.view(b_sz, s_len, d_dim)
        if self.use_shared:
            final_out += self.shared_expert(x_in)

        return final_out, aux_loss, utilization


class TransformerBlock(nn.Module):
    """
    Standard Transformer block containing attention and MoE paths.
    """

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GQAAttention(config)
        self.moe_norm = RMSNorm(config.d_model, config.norm_eps)
        self.moe = MoELayer(config)

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def forward(
        self,
        x_in: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Block forward pass."""
        # Attention path
        attn_out, new_kv_cache = self.attention(
            self.attn_norm(x_in), cos, sin, mask, kv_cache
        )
        h_out = x_in + attn_out
        # MoE path
        moe_out, aux_loss, util = self.moe(self.moe_norm(h_out))
        out = h_out + moe_out
        return out, aux_loss, util, new_kv_cache


class MoETransformer(nn.Module):
    """
    Complete Decoder-Only MoE Transformer.
    """

    def __init__(self, config: SmallLLMConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RoPE(config.d_model // config.n_heads)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        # Weight tying: output projection shares weights with input embedding
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.tok_emb.weight

        # Causal mask cache
        mask_raw = torch.full(
            (config.context_length, config.context_length), float("-inf")
        )
        mask_raw = torch.triu(mask_raw, diagonal=1)
        self.register_buffer("mask", mask_raw)

    # pylint: disable=too-many-locals
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """Transformer forward pass with targets and cache support."""
        _, s_len = idx.shape
        x_tok = self.tok_emb(idx)

        # If using KV cache, we need to handle positional embeddings differently
        total_s = s_len
        if kv_caches is not None:
            total_s += kv_caches[0][0].shape[1]

        cos_all, sin_all = self.rope(x_tok, total_s)
        # Only take the relevant parts of cos/sin for current input s
        cos = cos_all[:, -s_len:, :, :]
        sin = sin_all[:, -s_len:, :, :]

        mask_all = self.mask[:total_s, :total_s]
        # Reshape mask for attention broadcasting: (1, 1, s, total_s)
        mask_attn = mask_all[None, None, -s_len:, :]

        total_aux_loss = 0
        total_util = 0
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x_tok, aux_loss, util, new_cache = layer(
                x_tok, cos, sin, mask_attn, layer_cache
            )
            total_aux_loss += aux_loss
            total_util += util
            new_kv_caches.append(new_cache)

        x_tok = self.norm(x_tok)
        logits_out = self.output(x_tok)

        avg_util = total_util / len(self.layers)

        loss_out = None
        if targets is not None:
            # Flatten for CrossEntropy
            loss_out = F.cross_entropy(
                logits_out.view(-1, self.config.vocab_size), targets.view(-1)
            )
            # Add total MoE load balance loss
            loss_out += self.config.moe_aux_loss_coeff * total_aux_loss

        return logits_out, loss_out, total_aux_loss, avg_util, new_kv_caches


if __name__ == "__main__":
    # Test block
    test_config = SmallLLMConfig(fast_test=True)
    test_model = MoETransformer(test_config)
    print(
        f"Model parameters: {sum(p.numel() for p in test_model.parameters()) / 1e6:.2f}M"
    )

    # Test forward pass
    test_x = torch.randint(0, test_config.vocab_size, (2, 16))
    test_logits, test_loss, test_aux, test_util, _ = test_model(test_x, targets=test_x)
    print(
        f"Logits: {test_logits.shape}, Loss: {test_loss.item():.4f}, "
        f"Aux: {test_aux.item():.4f}, Util: {test_util:.2f}%"
    )
