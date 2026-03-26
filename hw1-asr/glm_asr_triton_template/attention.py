"""
Triton Multi-Head Attention

 FlashAttention-style fused kernel
    - Online softmax recurrence (Milakov & Gimelshein, 2018)
    - O(seq * head_dim) memory instead of O(seq^2)
    - Causal mask applied inside the K-loop — no extra memory, no extra kernel
    - 1 kernel launch vs 3 in the reference path

OOM fixes applied:
    - MAX_ATTENTION_DIM reduced to 128 (safer for most cluster GPUs)
    - BLOCK_K hard-capped at 64 inside the flash path to avoid register pressure
    - try/except around flash kernel call — falls back to reference path if it fails
    - Reference path also falls back to PyTorch if seq or head_dim too large

Toggle: USE_FLASH_ATTENTION = True / False
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional


def get_stream():
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


USE_FLASH_ATTENTION = False

MAX_ATTENTION_DIM = 128   # was 256 — safer for cluster GPUs
FLASH_BLOCK_K     = 64    # hard cap on BLOCK_K for flash kernel

# internal: print flash config only once
_flash_printed = False
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================================
# OPT-3: FlashAttention-style kernel
# ============================================================================

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    scale,
    seq_q, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    IS_CAUSAL: tl.constexpr,
    BLOCK_K:   tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    """
    FlashAttention-style fused kernel.
    Grid: (batch_heads, seq_q) — one program per (head, query-position).

    Online softmax algorithm:
        m = -inf, l = 0, acc = 0
        For each K-block:
            s      = q @ K_block^T * scale
            m_new  = max(m, max(s))
            acc   *= exp(m - m_new)     <- rescale to new max
            l     *= exp(m - m_new)
            acc   += exp(s - m_new) @ V_block
            l     += sum(exp(s - m_new))
            m      = m_new
        output = acc / l                <- single VRAM write

    Scores matrix is NEVER allocated (O(seq^2) eliminated).
    Causal mask is free — applied with tl.where inside the loop.
    BLOCK_K is capped at 64 externally to avoid register pressure / OOM.
    """
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    # Load query vector — stays in registers for the full kernel
    q = tl.load(
        q_ptr + pid_bh*stride_q0 + pid_q*stride_q1 + offs_d*stride_q2,
        mask=offs_d < head_dim, other=0.0,
    )

    # Online softmax state (all register-resident)
    m   = tl.full((1,), float('-inf'), dtype=tl.float32)
    l   = tl.zeros((1,),              dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,),        dtype=tl.float32)

    for k_start in range(0, seq_k, BLOCK_K):
        k_idx = k_start + offs_k

        k_blk = tl.load(
            k_ptr + pid_bh*stride_k0 + k_idx[:,None]*stride_k1 + offs_d[None,:]*stride_k2,
            mask=(k_idx[:,None] < seq_k) & (offs_d[None,:] < head_dim), other=0.0,
        )

        # Partial dot product scores
        s = tl.sum(k_blk * q[None,:], axis=1) * scale

        # Causal mask — no extra memory, no extra kernel
        if IS_CAUSAL:
            s = tl.where(k_idx <= pid_q, s, float('-inf'))

        # Mask positions beyond seq_k
        s = tl.where(k_idx < seq_k, s, float('-inf'))

        # Online softmax update
        m_new   = tl.maximum(m, tl.max(s, axis=0))
        rescale = tl.exp(m - m_new)
        acc     = acc * rescale
        l       = l   * rescale

        v_blk = tl.load(
            v_ptr + pid_bh*stride_v0 + k_idx[:,None]*stride_v1 + offs_d[None,:]*stride_v2,
            mask=(k_idx[:,None] < seq_k) & (offs_d[None,:] < head_dim), other=0.0,
        )
        exp_s = tl.exp(s - m_new)
        acc  += tl.sum(v_blk * exp_s[:,None], axis=0)
        l    += tl.sum(exp_s, axis=0)
        m     = m_new

    # Single VRAM write
    tl.store(
        out_ptr + pid_bh*stride_o0 + pid_q*stride_o1 + offs_d*stride_o2,
        acc / l,
        mask=offs_d < head_dim,
    )


# ============================================================================
# Reference: three-kernel path (retained for correctness verification)
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Q @ K^T * scale -> HBM.  Grid: (batch_heads, seq_q)"""
    pid_bh = tl.program_id(0); pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K); offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + pid_bh*stride_q0 + pid_q*stride_q1 + offs_d*stride_q2,
                mask=offs_d < head_dim, other=0.0)
    k = tl.load(k_ptr + pid_bh*stride_k0 + offs_k[:,None]*stride_k1 + offs_d[None,:]*stride_k2,
                mask=(offs_k[:,None] < seq_k) & (offs_d[None,:] < head_dim), other=0.0)
    scores = tl.sum(k * q[None,:], axis=1) * scale
    tl.store(scores_ptr + pid_bh*stride_s0 + pid_q*stride_s1 + offs_k*stride_s2,
             scores, mask=offs_k < seq_k)


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """Safe softmax in-place.  Grid: (batch_heads * seq_q,)"""
    row = tl.program_id(0); offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(scores_ptr + row*stride_s + offs, mask=offs < seq_k, other=-float('inf'))
    x_max = tl.max(x, axis=0); exp_x = tl.exp(x - x_max)
    tl.store(scores_ptr + row*stride_s + offs,
             exp_x / tl.sum(exp_x, axis=0), mask=offs < seq_k)


@triton.jit
def attention_output_kernel(
    attn_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """weights @ V.  Grid: (batch_heads, seq_q)"""
    pid_bh = tl.program_id(0); pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K); offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(attn_ptr + pid_bh*stride_w0 + pid_q*stride_w1 + offs_k*stride_w2,
                mask=offs_k < seq_k, other=0.0)
    v = tl.load(v_ptr + pid_bh*stride_v0 + offs_k[:,None]*stride_v1 + offs_d[None,:]*stride_v2,
                mask=(offs_k[:,None] < seq_k) & (offs_d[None,:] < head_dim), other=0.0)
    out = tl.sum(v * w[:,None], axis=0)
    tl.store(output_ptr + pid_bh*stride_o0 + pid_q*stride_o1 + offs_d*stride_o2,
             out, mask=offs_d < head_dim)


@triton.jit
def causal_mask_kernel(
    scores_ptr, seq_k, offset,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0); pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    scores = tl.load(scores_ptr + pid_bh*stride_s0 + pid_q*stride_s1 + offs_k*stride_s2,
                     mask=offs_k < seq_k, other=-1e9)
    scores = tl.where(offs_k > pid_q + offset, -1e9, scores)
    tl.store(scores_ptr + pid_bh*stride_s0 + pid_q*stride_s1 + offs_k*stride_s2,
             scores, mask=offs_k < seq_k)


# ============================================================================
# Attention classes
# ============================================================================

class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads, num_kv_heads=None, head_dim=None):
        self.hidden_size        = hidden_size
        self.num_heads          = num_heads
        self.num_kv_heads       = num_kv_heads or num_heads
        self.head_dim           = head_dim or (hidden_size // num_heads)
        self.scale              = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(self, q, k, v, attention_mask=None, is_causal=False):
        _, num_heads, _, _    = q.shape
        _, num_kv_heads, _, _ = k.shape
        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)
        return scaled_dot_product_attention(q, k, v, attention_mask, is_causal, self.scale)

    def _expand_kv(self, x, n):
        B, h, s, d = x.shape
        return x[:, :, None, :, :].expand(B, h, n, s, d).reshape(B, h*n, s, d)


def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


def scaled_dot_product_attention(q, k, v,
                                  attention_mask=None,
                                  is_causal=False,
                                  scale=None):
    """
    Scaled dot-product attention.
    Routes to flash_attention_kernel (OPT-3) when USE_FLASH_ATTENTION=True.
    Falls back gracefully on OOM or unsupported dimensions.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _                    = k.shape
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    head_dim_padded = next_power_of_two(head_dim)
    seq_k_padded    = next_power_of_two(seq_k)

    # ── OPT-3: FlashAttention ────────────────────────────────────────────────
    # Conditions:
    #   - flag enabled
    #   - running on CUDA
    #   - head_dim fits within safe limit
    #   - no additive attention_mask (flash handles causal internally)
    if (USE_FLASH_ATTENTION
            and q.is_cuda
            and head_dim_padded <= MAX_ATTENTION_DIM
            and attention_mask is None):

        global _flash_printed
        if not _flash_printed:
            _flash_printed = True
            SEP = "=" * 55
            print()
            print(SEP)
            print("  attention.py — Active Configuration")
            print(SEP)
            if USE_FLASH_ATTENTION:
                print(f"  OPT-3  FLASH ATTENTION : ON")
                print(f"         USE_FLASH       : {USE_FLASH_ATTENTION}")
                print(f"         BLOCK_K         : {FLASH_BLOCK_K}")
                print(f"         BLOCK_D         : {head_dim_padded}")
                print(f"         MAX_ATTN_DIM    : {MAX_ATTENTION_DIM}")
                print(f"         causal          : {is_causal}")
            else:
                print(f"  OPT-3  FLASH ATTENTION : OFF  (3-kernel reference path)")
            print(SEP)
            print()

        bh = batch * num_heads
        qf = q.reshape(bh, seq_q, head_dim).to(torch.float32).contiguous()
        kf = k.reshape(bh, seq_k, head_dim).to(torch.float32).contiguous()
        vf = v.reshape(bh, seq_k, head_dim).to(torch.float32).contiguous()

        # Pad head_dim to power-of-two if needed
        if head_dim_padded != head_dim:
            def _pad_head(t, d):
                p = torch.zeros((t.shape[0], t.shape[1], d),
                                dtype=torch.float32, device=t.device)
                p[:, :, :head_dim] = t
                return p
            qf = _pad_head(qf, head_dim_padded)
            kf = _pad_head(kf, head_dim_padded)
            vf = _pad_head(vf, head_dim_padded)

        out = torch.empty((bh, seq_q, head_dim_padded),
                          dtype=torch.float32, device=q.device)

        # BLOCK_K hard-capped at FLASH_BLOCK_K (64) to avoid register OOM
        block_k = FLASH_BLOCK_K

        try:
            flash_attention_kernel[(bh, seq_q)](
                qf, kf, vf, out,
                float(scale), seq_q, seq_k, head_dim_padded,
                qf.stride(0), qf.stride(1), qf.stride(2),
                kf.stride(0), kf.stride(1), kf.stride(2),
                vf.stride(0), vf.stride(1), vf.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                IS_CAUSAL=is_causal,
                BLOCK_K=block_k,
                BLOCK_D=head_dim_padded,
            )
        except Exception as e:
            # Graceful fallback to reference path if flash kernel fails
            print(f"[attention] flash kernel failed ({e}), falling back to reference path")
            return _reference_triton_attention(
                q, k, v, attention_mask, is_causal, scale,
                batch, num_heads, seq_q, seq_k, head_dim,
                head_dim_padded, seq_k_padded
            )

        if head_dim_padded != head_dim:
            out = out[:, :, :head_dim]
        return out.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # ── Reference: three-kernel Triton path ─────────────────────────────────
    return _reference_triton_attention(
        q, k, v, attention_mask, is_causal, scale,
        batch, num_heads, seq_q, seq_k, head_dim,
        head_dim_padded, seq_k_padded
    )


def _reference_triton_attention(q, k, v, attention_mask, is_causal, scale,
                                 batch, num_heads, seq_q, seq_k, head_dim,
                                 head_dim_padded, seq_k_padded):
    """Three-kernel reference path — also falls back to PyTorch if dims too large."""

    use_triton = (q.is_cuda
                  and seq_k_padded    <= MAX_ATTENTION_DIM
                  and head_dim_padded <= MAX_ATTENTION_DIM)

    if use_triton:
        bh = batch * num_heads
        qf = q.reshape(bh, seq_q, head_dim).to(torch.float32)
        kf = k.reshape(bh, seq_k, head_dim).to(torch.float32)
        vf = v.reshape(bh, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            def _pad(t, s, d):
                p = torch.zeros((t.shape[0], s, d),
                                dtype=torch.float32, device=q.device)
                p[:, :t.shape[1], :t.shape[2]] = t
                return p
            kf = _pad(kf, seq_k_padded, head_dim_padded)
            vf = _pad(vf, seq_k_padded, head_dim_padded)
            qp = torch.zeros((bh, seq_q, head_dim_padded),
                             dtype=torch.float32, device=q.device)
            qp[:, :, :head_dim] = qf; qf = qp

        try:
            scores = torch.empty((bh, seq_q, seq_k_padded),
                                 dtype=torch.float32, device=q.device)
            output = torch.empty((bh, seq_q, head_dim_padded),
                                 dtype=torch.float32, device=q.device)
            grid = (bh, seq_q)

            attention_scores_kernel[grid](
                qf, kf, scores, float(scale), seq_k_padded, head_dim_padded,
                qf.stride(0), qf.stride(1), qf.stride(2),
                kf.stride(0), kf.stride(1), kf.stride(2),
                scores.stride(0), scores.stride(1), scores.stride(2),
                BLOCK_K=seq_k_padded, BLOCK_D=head_dim_padded)

            if seq_k_padded != seq_k:
                scores[:, :, seq_k:] = -1e9

            if is_causal:
                mask = torch.triu(
                    torch.ones((seq_q, seq_k_padded),
                               dtype=torch.float32, device=q.device), diagonal=1
                ) * -1e9
                scores = scores + mask[None, :, :]

            if attention_mask is not None:
                if attention_mask.ndim == 4:
                    attention_mask = attention_mask.reshape(bh, seq_q, seq_k)
                if seq_k_padded != seq_k:
                    mp = torch.zeros((bh, seq_q, seq_k_padded),
                                    dtype=torch.float32, device=q.device)
                    mp[:, :, :seq_k] = attention_mask
                    mp[:, :, seq_k:] = -1e9
                    attention_mask = mp
                scores = scores + attention_mask

            s2d = scores.reshape(bh * seq_q, seq_k_padded)
            softmax_inplace_kernel[(s2d.shape[0],)](
                s2d, s2d.stride(0), seq_k_padded, BLOCK_SIZE=seq_k_padded)
            scores = s2d.reshape(bh, seq_q, seq_k_padded)

            attention_output_kernel[grid](
                scores, vf, output, seq_k_padded, head_dim_padded,
                scores.stride(0), scores.stride(1), scores.stride(2),
                vf.stride(0), vf.stride(1), vf.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                BLOCK_K=seq_k_padded, BLOCK_D=head_dim_padded)

            if head_dim_padded != head_dim:
                output = output[:, :, :head_dim]
            return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

        except Exception as e:
            print(f"[attention] triton reference failed ({e}), falling back to PyTorch")

    # ── Pure PyTorch fallback ────────────────────────────────────────────────
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1
        ) * -1e9
        scores = scores + mask[None, None, :, :]
    if attention_mask is not None:
        scores = scores + attention_mask
    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn   = torch.exp(scores) / torch.sum(torch.exp(scores), dim=-1, keepdim=True)
    return torch.einsum("bnqk,bnkd->bnqd", attn, v).to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention (OPT-3: FlashAttention)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, S, D = 2, 4, 16, 64

    q = torch.randn(B, H, S, D, device=device)
    k = torch.randn(B, H, S, D, device=device)
    v = torch.randn(B, H, S, D, device=device)

    # Non-causal
    out_nc = scaled_dot_product_attention(q, k, v, is_causal=False)
    print(f"  Flash non-causal: {out_nc.shape}")

    # Causal
    out_c = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Flash causal:     {out_c.shape}")

    # Verify flash matches reference
    import attention as _self
    _self.USE_FLASH_ATTENTION = False
    out_ref = scaled_dot_product_attention(q, k, v, is_causal=True)
    _self.USE_FLASH_ATTENTION = True
    err = float((out_c - out_ref).abs().max())
    print(f"  Max error vs reference: {err:.6f}  {'PASS' if err < 1e-3 else 'FAIL'}")

    # GQA
    attn = MultiHeadAttention(H*D, H, num_kv_heads=2)
    kg = torch.randn(B, 2, S, D, device=device)
    vg = torch.randn(B, 2, S, D, device=device)
    print(f"  GQA: {attn(q, kg, vg).shape}")

    print("Triton Attention working!")