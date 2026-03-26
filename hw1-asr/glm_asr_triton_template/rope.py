"""
Triton Rotary Position Embeddings (RoPE)
compute_freqs_kernel: one Triton program per sequence position.
"""

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


def get_stream():
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


@triton.jit
def compute_freqs_kernel(
    positions_ptr, inv_freq_ptr,
    cos_ptr, sin_ptr,
    seq_len, half_dim,
    stride_pos, stride_inv,
    stride_cos0, stride_cos1,
    stride_sin0, stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos/sin tables for rotary embeddings.
    Grid: (seq_len,) — one program per position.

    For position p:
        freqs[i]               = p * inv_freq[i]    for i in [0, half_dim)
        cos_cache[p, i]        = cos(freqs[i])
        cos_cache[p, i+half]   = cos(freqs[i])      duplicated so apply_rope
        sin_cache[p, i]        = sin(freqs[i])      needs no extra concat
        sin_cache[p, i+half]   = sin(freqs[i])
    """
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # Step 1: load position scalar
    pos = tl.load(positions_ptr + pid * stride_pos)

    # Step 2: load inverse frequencies
    inv_freq = tl.load(inv_freq_ptr + offs * stride_inv,
                        mask=offs < half_dim, other=0.0)

    # Step 3: freqs = position * inv_freq
    freqs = pos * inv_freq

    # Step 4: cos and sin
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)

    # Step 5: store both halves (avoids concat in apply_rope)
    cos_row = cos_ptr + pid * stride_cos0
    sin_row = sin_ptr + pid * stride_sin0

    tl.store(cos_row + offs              * stride_cos1, cos_val, mask=offs < half_dim)
    tl.store(cos_row + (offs + half_dim) * stride_cos1, cos_val, mask=offs < half_dim)
    tl.store(sin_row + offs              * stride_sin1, sin_val, mask=offs < half_dim)
    tl.store(sin_row + (offs + half_dim) * stride_sin1, sin_val, mask=offs < half_dim)


class RotaryEmbedding:
    """Rotary Position Embedding using Triton kernel for cache computation."""

    def __init__(self, dim, max_position_embeddings=8192,
                 base=10000.0, partial_rotary_factor=1.0):
        self.dim         = dim
        self.max_position_embeddings = max_position_embeddings
        self.base        = base
        self.partial_rotary_factor   = partial_rotary_factor

        self.rotary_dim  = int(dim * partial_rotary_factor)
        self.rotary_dim -= self.rotary_dim % 2   # must be even

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                     / self.rotary_dim)
        )
        self.inv_freq = inv_freq
        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len, device=None):
        """Pre-compute cos/sin tables using Triton kernel (CUDA) or CPU fallback."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions, self.inv_freq, cos_cache, sin_cache,
                seq_len, half_dim,
                positions.stride(0), self.inv_freq.stride(0),
                cos_cache.stride(0), cos_cache.stride(1),
                sin_cache.stride(0), sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            # CPU fallback — mathematically identical to the Triton kernel
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]  # (seq, half_dim)
            cos_h = torch.cos(freqs)
            sin_h = torch.sin(freqs)
            cos_cache[:, :half_dim]            = cos_h
            cos_cache[:, half_dim:half_dim * 2] = cos_h
            sin_cache[:, :half_dim]            = sin_h
            sin_cache[:, half_dim:half_dim * 2] = sin_h

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(self, x, position_ids=None):
        """Return (cos, sin) for the given sequence length / position ids."""
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]; sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


def _apply_rope_single(x, cos, sin, half_dim, head_dim):
    """Apply RoPE to one tensor (Q or K) using PyTorch ops."""
    cos = cos[:x.shape[-2]]
    sin = sin[:x.shape[-2]]
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:half_dim * 2]
    ce = cos[None, None, :, :]
    se = sin[None, None, :, :]
    x1r = x1 * ce - x2 * se
    x2r = x2 * ce + x1 * se
    if head_dim > half_dim * 2:
        return torch.cat([x1r, x2r, x[..., half_dim * 2:]], dim=-1)
    return torch.cat([x1r, x2r], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, rotary_dim=None):
    """Apply rotary position embeddings to Q and K."""
    _, _, _, head_dim = q.shape
    if rotary_dim is None: rotary_dim = head_dim
    half_dim = rotary_dim // 2
    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]
    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()
    return (
        _apply_rope_single(q, cos, sin, half_dim, head_dim).to(q.dtype),
        _apply_rope_single(k, cos, sin, half_dim, head_dim).to(k.dtype),
    )


def apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    """Apply rotary embeddings to partial head dimensions (encoder uses 50%)."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, S, D = 2, 4, 16, 64

    rope = RotaryEmbedding(dim=D, max_position_embeddings=1024)
    q = torch.randn(B, H, S, D, device=device)
    k = torch.randn(B, H, S, D, device=device)

    cos, sin = rope(q)
    print(f"  Cos: {cos.shape}  Sin: {sin.shape}")

    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"  Q rotated: {q_r.shape}  K rotated: {k_r.shape}")

    rope_p = RotaryEmbedding(dim=D, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_p(q)
    q_p, k_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, D // 2)
    print(f"  Partial RoPE Q: {q_p.shape}")

    print("Triton RoPE working!")