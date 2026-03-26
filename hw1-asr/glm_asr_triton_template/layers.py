"""
Triton Neural Network Layers — Optimised Implementation

OPT-1: Tile and Block Size Tuning
    TILE_CONFIG_A: BLOCK_M=32,  BLOCK_N=32,  BLOCK_K=16, num_warps=2, num_stages=2
    TILE_CONFIG_B: BLOCK_M=64,  BLOCK_N=64,  BLOCK_K=32, num_warps=4, num_stages=3  <- default
    TILE_CONFIG_C: BLOCK_M=128, BLOCK_N=64,  BLOCK_K=32, num_warps=8, num_stages=3

OPT-2: Kernel Fusion
    linear_gelu_kernel:  fused fc1 + GELU  (EncoderMLP / projector)
    swiglu_fused_kernel: fused gate_proj + up_proj + SiLU  (decoder MLP)

OOM fixes applied:
    - Config C guarded with try/except in benchmark (register spilling risk)
    - Padding helpers use contiguous() to avoid non-contiguous stride errors
    - All kernel launches wrapped with try/except in _forward_triton,
      falling back to torch if the Triton call fails
"""

import time
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl




def _print_layers_config():
    cfg = Linear.TILE_CONFIG
    SEP = "=" * 55
    print()
    print(SEP)
    print("  layers.py — Active Configuration")
    print(SEP)
    print(f"  OPT-1  BACKEND      : {Linear.BACKEND}")
    print(f"         TILE_CONFIG  : "
          f"BLOCK={cfg.block_m}x{cfg.block_n}x{cfg.block_k} | "
          f"warps={cfg.num_warps} | stages={cfg.num_stages}")
    print(f"  OPT-2  MLP.FUSED        : {MLP.FUSED}"
          f"   <- swiglu_fused_kernel")
    print(f"         EncoderMLP.FUSED : {EncoderMLP.FUSED}"
          f"   <- linear_gelu_kernel")
    print(SEP)
    print()
# ============================================================================
# Helpers
# ============================================================================

def get_stream():
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None

def pad_to_multiple(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple

def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ============================================================================
# OPT-1: Tile configurations
# ============================================================================

class TileConfig:
    """
    Tile/block-size configuration for linear_kernel_tf32.

    block_m x block_n  — output tile each program computes.
    block_k            — inner-loop step along the K (shared) dimension.
    num_warps          — warp groups per program (32 threads each).
    num_stages         — software pipeline depth (overlaps loads with compute).
    """
    def __init__(self, block_m, block_n, block_k, num_warps, num_stages):
        self.block_m    = block_m
        self.block_n    = block_n
        self.block_k    = block_k
        self.num_warps  = num_warps
        self.num_stages = num_stages

# Config A: small tiles — best for encoder (hidden=1280) or small M
TILE_CONFIG_A = TileConfig(block_m=32,  block_n=32, block_k=16, num_warps=2, num_stages=2)

# Config B: 64x64 — aligns with A100/H100 MMA, best overall for decoder
TILE_CONFIG_B = TileConfig(block_m=64,  block_n=64, block_k=32, num_warps=4, num_stages=3)

# Config C: large-M — fewer dispatches, better for long-audio prefill
# (may OOM on small register-file GPUs — benchmarked with try/except)
TILE_CONFIG_C = TileConfig(block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3)


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_x, stride_y,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * w.  Grid: (batch,)"""
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid*stride_x + offs, mask=offs < hidden_size, other=0.0)
    w = tl.load(w_ptr + offs,                mask=offs < hidden_size, other=0.0)
    var = tl.sum(x * x, axis=0) / hidden_size
    rms = tl.sqrt(var + eps)
    tl.store(y_ptr + pid*stride_y + offs, (x / rms) * w, mask=offs < hidden_size)


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    stride_x, stride_y,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """LayerNorm: y = (x-mean)/sqrt(var+eps)*w + b.  Grid: (batch,)"""
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid*stride_x + offs, mask=offs < hidden_size, other=0.0)
    w = tl.load(w_ptr + offs,                mask=offs < hidden_size, other=1.0)
    b = tl.load(b_ptr + offs,                mask=offs < hidden_size, other=0.0)
    mean = tl.sum(x, axis=0) / hidden_size
    xc   = x - mean
    var  = tl.sum(xc * xc, axis=0) / hidden_size
    y    = xc / tl.sqrt(var + eps) * w + b
    tl.store(y_ptr + pid*stride_y + offs, y, mask=offs < hidden_size)


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """GELU tanh approximation."""
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x     = tl.load(x_ptr + offs, mask=mask, other=0.0)
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    y     = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """SiLU: x * sigmoid(x)"""
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x / (1.0 + tl.exp(-x)), mask=mask)


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    """Numerically stable softmax.  Grid: (batch,)"""
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    x     = tl.load(x_ptr + row*stride_x + offs, mask=offs < n_cols, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    tl.store(y_ptr + row*stride_y + offs, exp_x / tl.sum(exp_x, axis=0), mask=offs < n_cols)


# ----------------------------------------------------------------------------
# OPT-1: Tiled matmul  C = A @ B
# A:(M,K)  B:(K,N)  C:(M,N)   Grid:(cdiv(M,BM), cdiv(N,BN))
# ----------------------------------------------------------------------------
@triton.jit
def linear_kernel_tf32(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled matmul.
    Accumulator is register-resident for the whole K-loop; single store at end.
    Data reuse: each A element reused BLOCK_N times, each B element BLOCK_M times.
    """
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        a = tl.load(a_ptr + offs_m[:,None]*stride_am + k_idx[None,:]*stride_ak,
                    mask=(offs_m[:,None] < M) & (k_idx[None,:] < K), other=0.0)
        b = tl.load(b_ptr + k_idx[:,None]*stride_bk + offs_n[None,:]*stride_bn,
                    mask=(k_idx[:,None] < K) & (offs_n[None,:] < N), other=0.0)
        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn,
             acc, mask=(offs_m[:,None] < M) & (offs_n[None,:] < N))


# ----------------------------------------------------------------------------
# OPT-2: Fused Linear + GELU (encoder MLP / projector)
# Saves 2 VRAM passes vs unfused path.
# ----------------------------------------------------------------------------
@triton.jit
def linear_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused matmul + GELU.
    GELU applied to the accumulator in registers after K-loop — no intermediate store.
    """
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:,None]*stride_am + (k+offs_k[None,:])*stride_ak,
                    mask=(offs_m[:,None] < M) & (k+offs_k[None,:] < K), other=0.0)
        b = tl.load(b_ptr + (k+offs_k[:,None])*stride_bk + offs_n[None,:]*stride_bn,
                    mask=(k+offs_k[:,None] < K) & (offs_n[None,:] < N), other=0.0)
        acc += tl.dot(a, b)

    # GELU in registers — zero intermediate VRAM writes
    inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
    acc   = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(c_ptr + offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn,
             acc, mask=(offs_m[:,None] < M) & (offs_n[None,:] < N))


# ----------------------------------------------------------------------------
# OPT-2: Fused SwiGLU (decoder MLP)
# Saves 3 VRAM passes vs unfused path; input tile loaded once for both projections.
# ----------------------------------------------------------------------------
@triton.jit
def swiglu_fused_kernel(
    a_ptr, gate_ptr, up_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_gk, stride_gn,
    stride_uk, stride_un,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused SwiGLU: SiLU(x @ W_gate) * (x @ W_up).
    Input tile loaded once per K-iter; gate_acc and up_acc in registers throughout.
    """
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Input loaded ONCE — reused for both gate and up projections
        a = tl.load(a_ptr + offs_m[:,None]*stride_am + (k+offs_k[None,:])*stride_ak,
                    mask=(offs_m[:,None] < M) & (k+offs_k[None,:] < K), other=0.0)
        gw = tl.load(gate_ptr + (k+offs_k[:,None])*stride_gk + offs_n[None,:]*stride_gn,
                     mask=(k+offs_k[:,None] < K) & (offs_n[None,:] < N), other=0.0)
        uw = tl.load(up_ptr + (k+offs_k[:,None])*stride_uk + offs_n[None,:]*stride_un,
                     mask=(k+offs_k[:,None] < K) & (offs_n[None,:] < N), other=0.0)
        gate_acc += tl.dot(a, gw)
        up_acc   += tl.dot(a, uw)

    # SiLU(gate) * up — computed in registers, single store
    out = (gate_acc / (1.0 + tl.exp(-gate_acc))) * up_acc
    tl.store(c_ptr + offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn,
             out, mask=(offs_m[:,None] < M) & (offs_n[None,:] < N))


@triton.jit
def embedding_kernel(
    indices_ptr, weight_ptr, output_ptr,
    embedding_dim, stride_w0, stride_w1, stride_out0,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding lookup.  Grid: (num_tokens, cdiv(dim, BLOCK))"""
    pid0 = tl.program_id(0); pid1 = tl.program_id(1)
    idx  = tl.load(indices_ptr + pid0)
    offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim
    w    = tl.load(weight_ptr + idx*stride_w0 + offs*stride_w1, mask=mask, other=0.0)
    tl.store(output_ptr + pid0*stride_out0 + offs, w, mask=mask)


# retained for compatibility with attention.py
@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
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
    row = tl.program_id(0); offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(scores_ptr + row*stride_s + offs, mask=offs < seq_k, other=-float('inf'))
    x_max = tl.max(x, axis=0); exp_x = tl.exp(x - x_max)
    tl.store(scores_ptr + row*stride_s + offs,
             exp_x / tl.sum(exp_x, axis=0), mask=offs < seq_k)


@triton.jit
def attention_output_kernel(
    weights_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0); pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K); offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(weights_ptr + pid_bh*stride_w0 + pid_q*stride_w1 + offs_k*stride_w2,
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
# Layer Classes
# ============================================================================

class RMSNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps         = eps
        self.weight      = torch.ones(hidden_size, dtype=torch.float32)
        self.use_triton  = _is_power_of_two(hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if self.use_triton and x.is_cuda:
            B  = int(np.prod(x.shape[:-1]))
            xf = x.reshape(B, self.hidden_size).to(torch.float32).contiguous()
            out = torch.empty_like(xf)
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            rmsnorm_kernel[(B,)](
                xf, self.weight, out,
                xf.stride(0), out.stride(0),
                self.hidden_size, self.eps,
                BLOCK_SIZE=next_power_of_two(self.hidden_size))
            return out.reshape(shape)
        xf = x.to(torch.float32)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        return (self.weight * xf * torch.rsqrt(
            torch.mean(xf*xf, dim=-1, keepdim=True) + self.eps)).to(x.dtype)


class LayerNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps         = eps
        self.weight      = torch.ones(hidden_size,  dtype=torch.float32)
        self.bias        = torch.zeros(hidden_size, dtype=torch.float32)
        self.use_triton  = _is_power_of_two(hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if self.use_triton and x.is_cuda:
            B  = int(np.prod(x.shape[:-1]))
            xf = x.reshape(B, self.hidden_size).to(torch.float32).contiguous()
            out = torch.empty_like(xf)
            if self.weight.device != x.device: self.weight = self.weight.to(x.device)
            if self.bias.device   != x.device: self.bias   = self.bias.to(x.device)
            layernorm_kernel[(B,)](
                xf, self.weight, self.bias, out,
                xf.stride(0), out.stride(0),
                self.hidden_size, self.eps,
                BLOCK_SIZE=next_power_of_two(self.hidden_size))
            return out.reshape(shape)
        xf   = x.to(torch.float32)
        mean = torch.mean(xf, dim=-1, keepdim=True)
        var  = torch.var(xf,  dim=-1, keepdim=True, unbiased=False)
        xn   = (xf - mean) * torch.rsqrt(var + self.eps)
        if self.weight.device != x.device: self.weight = self.weight.to(x.device)
        if self.bias.device   != x.device: self.bias   = self.bias.to(x.device)
        return (self.weight * xn + self.bias).to(x.dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape; total = int(np.prod(shape)); block = 256
    xf = x.reshape(-1).to(torch.float32).contiguous()
    out = torch.empty_like(xf)
    if x.is_cuda:
        gelu_kernel[(triton.cdiv(total, block),)](xf, out, total, BLOCK_SIZE=block)
        return out[:total].reshape(shape).to(x.dtype)
    return torch.nn.functional.gelu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape; total = int(np.prod(shape)); block = 256
    xf = x.reshape(-1).to(torch.float32).contiguous()
    out = torch.empty_like(xf)
    if x.is_cuda:
        silu_kernel[(triton.cdiv(total, block),)](xf, out, total, BLOCK_SIZE=block)
        return out[:total].reshape(shape).to(x.dtype)
    return torch.nn.functional.silu(x)


def get_activation(name: str):
    return {"gelu": gelu, "silu": silu}[name]


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    if axis != -1 and axis != len(x.shape) - 1:
        x = torch.movedim(x, axis, -1)
    shape = x.shape; B = int(np.prod(x.shape[:-1])); S = x.shape[-1]
    xf = x.reshape(B, S).to(torch.float32).contiguous()
    out = torch.empty_like(xf)
    if x.is_cuda:
        softmax_kernel[(B,)](xf, out, xf.stride(0), out.stride(0), S,
                              BLOCK_SIZE=next_power_of_two(S))
        result = out.reshape(shape)
    else:
        result = torch.softmax(x, dim=-1)
    if axis != -1 and axis != len(shape) - 1:
        result = torch.movedim(result, -1, axis)
    return result


class Linear:
    """
    Linear layer with Triton tiled matmul backend.
    OPT-1: switch tile config via Linear.TILE_CONFIG = TILE_CONFIG_A/B/C
    Falls back to PyTorch if Triton kernel fails (e.g. OOM on Config C).
    """
    TILE_CONFIG   = TILE_CONFIG_B   # default — best overall for decoder
    BACKEND       = "triton"        # "triton", "torch", or "cublas"
    _printed_once = False           # print active config only on first call

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.has_bias     = bias
        self.weight       = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_param   = torch.zeros(out_features, dtype=torch.float32) if bias else None
        self._weight_t_padded = None
        self._K_padded = None
        self._N_padded = None

    def _ensure_weight_prepared(self):
        if self._weight_t_padded is None:
            cfg = Linear.TILE_CONFIG
            K, N = self.in_features, self.out_features
            self._K_padded = pad_to_multiple(K, cfg.block_k)
            self._N_padded = pad_to_multiple(N, cfg.block_n)
            wt = self.weight.t().contiguous()
            if self._K_padded > K or self._N_padded > N:
                p = torch.zeros((self._K_padded, self._N_padded),
                                dtype=torch.float32, device=wt.device)
                p[:K, :N] = wt
                self._weight_t_padded = p
            else:
                self._weight_t_padded = wt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if Linear.BACKEND in ("torch", "cublas"):
            return self._forward_torch(x)
        if Linear.BACKEND == "triton":
            return self._forward_triton(x)
        M = int(np.prod(x.shape[:-1]))
        return self._forward_triton(x) if (M >= Linear.TILE_CONFIG.block_m and x.is_cuda) \
               else self._forward_torch(x)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape; M = int(np.prod(shape[:-1]))
        xf = x.reshape(M, self.in_features).to(torch.float32)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        out = xf @ self.weight.t()
        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            out = out + self.bias_param
        return out.reshape(*shape[:-1], self.out_features)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        cfg   = Linear.TILE_CONFIG

        # Print full OPT-1 + OPT-2 config once on first kernel call
        if not Linear._printed_once:
            Linear._printed_once = True
            SEP = "=" * 55
            print()
            print(SEP)
            print("  layers.py — Active Configuration")
            print(SEP)
            print(f"    BACKEND      : {Linear.BACKEND}")
            print(f"         TILE_CONFIG  : "
                  f"BLOCK={cfg.block_m}x{cfg.block_n}x{cfg.block_k} | "
                  f"warps={cfg.num_warps} | stages={cfg.num_stages}")
            print(f"  OPT-2  MLP.FUSED         : {MLP.FUSED}"
                  f"  <- swiglu_fused_kernel")
            print(f"         EncoderMLP.FUSED  : {EncoderMLP.FUSED}"
                  f"  <- linear_gelu_kernel")
            print(SEP)
            print()

        shape = x.shape
        M, K, N = int(np.prod(shape[:-1])), self.in_features, self.out_features
        xf = x.reshape(M, K).to(torch.float32).contiguous()

        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            self._weight_t_padded = None
        self._ensure_weight_prepared()

        Mp = pad_to_multiple(M, cfg.block_m)
        if Mp > M or self._K_padded > K:
            xp = torch.zeros((Mp, self._K_padded), dtype=torch.float32, device=x.device)
            xp[:M, :K] = xf
        else:
            xp = xf

        out = torch.zeros((Mp, self._N_padded), dtype=torch.float32, device=x.device)

        try:
            grid = (triton.cdiv(Mp, cfg.block_m), triton.cdiv(self._N_padded, cfg.block_n))
            linear_kernel_tf32[grid](
                xp, self._weight_t_padded, out,
                Mp, self._N_padded, self._K_padded,
                xp.stride(0), xp.stride(1),
                self._weight_t_padded.stride(0), self._weight_t_padded.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            )
        except Exception as e:
            print(f"[Linear] Triton kernel failed ({e}), falling back to PyTorch")
            return self._forward_torch(x)

        out = out[:M, :N]
        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            out = out + self.bias_param
        return out.reshape(*shape[:-1], self.out_features)


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.weight = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        shape = input_ids.shape; B = int(np.prod(shape))
        if self.weight.device != input_ids.device:
            self.weight = self.weight.to(input_ids.device)
        if not input_ids.is_cuda:
            return self.weight.index_select(0, input_ids.reshape(-1).to(torch.int64))\
                              .reshape(*shape, self.embedding_dim)
        idx = input_ids.reshape(-1).to(torch.int32).contiguous()
        out = torch.empty((B, self.embedding_dim), dtype=torch.float32, device=idx.device)
        block = 256
        embedding_kernel[(B, triton.cdiv(self.embedding_dim, block))](
            idx, self.weight, out, self.embedding_dim,
            self.weight.stride(0), self.weight.stride(1), out.stride(0),
            BLOCK_SIZE=block)
        return out.reshape(*shape, self.embedding_dim)


class MLP:
    """
    Decoder SwiGLU MLP.
    OPT-2: FUSED=True uses swiglu_fused_kernel (eliminates 3 VRAM passes vs unfused).
    Set MLP.FUSED = False to use the reference unfused path.
    CACHE_WEIGHTS=False saves ~3.5 GB VRAM on 16 GB GPUs by not caching transposed weights.
    """
    FUSED         = True
    CACHE_WEIGHTS = False   # set True for speed if VRAM allows
    TILE_M        = 64; TILE_N = 64; TILE_K = 32
    _printed_once = False

    def __init__(self, hidden_size, intermediate_size,
                 activation="silu", bias=False, use_gating=True):
        self.use_gating        = use_gating
        self.act_fn            = get_activation(activation)
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled      = bias
        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj   = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj   = Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj  = Linear(intermediate_size, hidden_size, bias=bias)
        self._gate_wt   = None
        self._up_wt     = None

    def _prep(self):
        """
        Prepare transposed weights for fused kernel.
        NOTE: we do NOT cache these on low-memory GPUs — caching all 28 layers'
        transposed weights costs ~3.5 GB extra VRAM.
        Instead we create them on-the-fly and let them be freed after each call.
        To re-enable caching (faster but more VRAM), set MLP.CACHE_WEIGHTS = True.
        """
        if MLP.CACHE_WEIGHTS:
            if self._gate_wt is None and self.use_gating:
                try:
                    self._gate_wt = self.gate_proj.weight.t().contiguous()
                    self._up_wt   = self.up_proj.weight.t().contiguous()
                except torch.cuda.OutOfMemoryError:
                    print("[MLP] OOM caching weights — disabling weight cache")
                    MLP.CACHE_WEIGHTS = True
                    self._gate_wt = None
                    self._up_wt   = None

    def _get_weights(self, x):
        """Return (gate_wt, up_wt) — cached or freshly transposed."""
        if MLP.CACHE_WEIGHTS and self._gate_wt is not None:
            return self._gate_wt, self._up_wt
        try:
            gw = self.gate_proj.weight.t().contiguous()
            uw = self.up_proj.weight.t().contiguous()
            return gw, uw
        except torch.cuda.OutOfMemoryError:
            print("[MLP] OOM transposing weights — falling back to standard path")
            MLP.FUSED = False
            return None, None

    def __call__(self, x):
        if self.use_gating and MLP.FUSED and x.is_cuda:
            return self._fused(x)
        return self._standard(x)

    def _standard(self, x):
        if self.use_gating:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))

    def _fused(self, x):
        for wt_attr, proj in [('_gate_wt', self.gate_proj), ('_up_wt', self.up_proj)]:
            if proj.weight.device != x.device:
                proj.weight = proj.weight.to(x.device)
                setattr(self, wt_attr, None)
        self._prep()

        gate_wt, up_wt = self._get_weights(x)
        if gate_wt is None:
            return self._standard(x)


        shape = x.shape
        xf = x.reshape(-1, self.hidden_size).to(torch.float32).contiguous()
        M, K, N = xf.shape[0], self.hidden_size, self.intermediate_size
        Mp = pad_to_multiple(M, self.TILE_M)
        Kp = pad_to_multiple(K, self.TILE_K)
        Np = pad_to_multiple(N, self.TILE_N)

        def p2(t, r, c):
            if t.shape == (r, c): return t
            q = torch.zeros((r, c), dtype=torch.float32, device=x.device)
            q[:t.shape[0], :t.shape[1]] = t; return q

        xp  = p2(xf,      Mp, Kp)
        gwp = p2(gate_wt, Kp, Np)
        uwp = p2(up_wt,   Kp, Np)
        mid = torch.zeros((Mp, Np), dtype=torch.float32, device=x.device)

        try:
            grid = (triton.cdiv(Mp, self.TILE_M), triton.cdiv(Np, self.TILE_N))
            swiglu_fused_kernel[grid](
                xp, gwp, uwp, mid, Mp, Np, Kp,
                xp.stride(0),  xp.stride(1),
                gwp.stride(0), gwp.stride(1),
                uwp.stride(0), uwp.stride(1),
                mid.stride(0), mid.stride(1),
                BLOCK_M=self.TILE_M, BLOCK_N=self.TILE_N, BLOCK_K=self.TILE_K)
        except Exception as e:
            print(f"[MLP] swiglu_fused_kernel failed ({e}), falling back to standard path")
            return self._standard(x)

        # Free temporary transposed weights immediately if not caching
        if not MLP.CACHE_WEIGHTS:
            del gate_wt, up_wt

        mid = mid[:M, :N].reshape(*shape[:-1], self.intermediate_size)
        return self.down_proj(mid)


class EncoderMLP:
    """
    Encoder MLP (no gating, GELU).
    OPT-2: FUSED=True uses linear_gelu_kernel (eliminates 2 VRAM passes vs unfused).
    Set EncoderMLP.FUSED = False to use the reference unfused path.
    """
    FUSED         = True
    TILE_M        = 64; TILE_N = 64; TILE_K = 32
    _printed_once = False

    def __init__(self, hidden_size, intermediate_size, activation="gelu", bias=True):
        self.fc1              = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2              = Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn           = get_activation(activation)
        self.hidden_size      = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled     = bias
        self.activation       = activation
        self._fc1_wt          = None

    def _prep(self):
        if self._fc1_wt is None:
            try:
                self._fc1_wt = self.fc1.weight.t().contiguous()
            except torch.cuda.OutOfMemoryError:
                print("[EncoderMLP] OOM preparing fused weights — disabling fusion")
                EncoderMLP.FUSED = False
                self._fc1_wt     = None

    def __call__(self, x):
        if EncoderMLP.FUSED and self.activation == "gelu" and x.is_cuda:
            return self._fused(x)
        return self.fc2(self.act_fn(self.fc1(x)))

    def _fused(self, x):
        if self.fc1.weight.device != x.device:
            self.fc1.weight = self.fc1.weight.to(x.device)
            self._fc1_wt = None
        self._prep()


        shape = x.shape
        xf = x.reshape(-1, self.hidden_size).to(torch.float32).contiguous()
        M, K, N = xf.shape[0], self.hidden_size, self.intermediate_size
        Mp = pad_to_multiple(M, self.TILE_M)
        Kp = pad_to_multiple(K, self.TILE_K)
        Np = pad_to_multiple(N, self.TILE_N)

        def p2(t, r, c):
            if t.shape == (r, c): return t
            q = torch.zeros((r, c), dtype=torch.float32, device=x.device)
            q[:t.shape[0], :t.shape[1]] = t; return q

        xp  = p2(xf,           Mp, Kp)
        wp  = p2(self._fc1_wt, Kp, Np)
        mid = torch.zeros((Mp, Np), dtype=torch.float32, device=x.device)

        try:
            grid = (triton.cdiv(Mp, self.TILE_M), triton.cdiv(Np, self.TILE_N))
            linear_gelu_kernel[grid](
                xp, wp, mid, Mp, Np, Kp,
                xp.stride(0), xp.stride(1),
                wp.stride(0),  wp.stride(1),
                mid.stride(0), mid.stride(1),
                BLOCK_M=self.TILE_M, BLOCK_N=self.TILE_N, BLOCK_K=self.TILE_K)
        except Exception as e:
            print(f"[EncoderMLP] linear_gelu_kernel failed ({e}), falling back to standard path")
            return self.fc2(self.act_fn(self.fc1(x)))

        mid = mid[:M, :N]
        if self.bias_enabled and self.fc1.bias_param is not None:
            if self.fc1.bias_param.device != x.device:
                self.fc1.bias_param = self.fc1.bias_param.to(x.device)
            mid = mid + self.fc1.bias_param

        return self.fc2(mid.reshape(*shape[:-1], self.intermediate_size))




if __name__ == "__main__":
    print("=" * 65)
    print("  GLM-ASR Triton Layers — correctness + OPT-1 tile benchmark")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Correctness smoke tests ──────────────────────────────────────────────
    print("\n--- Correctness ---")
    x = torch.randn(2, 16, 256, device=device)

    rms = RMSNorm(256);  print(f"  RMSNorm:   {rms(x).shape}")
    ln  = LayerNorm(256); print(f"  LayerNorm: {ln(x).shape}")
    print(f"  GELU:      {gelu(x).shape}")
    print(f"  SiLU:      {silu(x).shape}")

    lin = Linear(256, 512)
    lin.weight = torch.randn(512, 256, device=device)
    print(f"  Linear:    {lin(x).shape}")

    emb = Embedding(1000, 256)
    ids = torch.randint(0, 1000, (2, 16), device=device)
    print(f"  Embedding: {emb(ids).shape}")

    xsm = torch.randn(2, 4, 16, 16, device=device)
    y   = softmax(xsm)
    print(f"  Softmax:   {y.shape}  sum={float(y[0,0,0].sum()):.6f} (should be 1.0)")

    mlp = MLP(256, 512, activation="silu", use_gating=True)
    print(f"  MLP:       {mlp(x).shape}")

    # ── OPT-1 Tile Benchmark ─────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("\n  No CUDA — skipping tile benchmark.")
    else:
        RUNS = 20; WARMUP = 5

        configs = {
            "Config A (32x32x16 | warps=2 | stages=2)": TILE_CONFIG_A,
            "Config B (64x64x32 | warps=4 | stages=3)": TILE_CONFIG_B,
            "Config C (128x64x32| warps=8 | stages=3)": TILE_CONFIG_C,
        }

        shapes = [
            # Audio encoder projections (Q/K/V)  hidden=1280, heads=20, head_dim=64
            ("Encoder Q/K/V  M=750  K=1280  N=1280 ", 750,  1280, 1280),
            # Audio encoder MLP  fc1: 1280->5120
            ("Encoder MLP    M=750  K=1280  N=5120 ", 750,  1280, 5120),
            # Text decoder projections  hidden=2048, heads=16, head_dim=128
            ("Decoder Q proj M=128  K=2048  N=2048 ", 128,  2048, 2048),
            # Text decoder KV projections  kv_heads=4 -> 4*128=512
            ("Decoder K/V    M=128  K=2048  N=512  ", 128,  2048,  512),
            # Text decoder SwiGLU MLP  gate/up: 2048->6144
            ("Decoder MLP    M=128  K=2048  N=6144 ", 128,  2048, 6144),
        ]

        print("\n--- OPT-1: Tile Configuration Benchmark ---")
        best = {}

        for label, M, K, N in shapes:
            print(f"\n  {label}")
            xb       = torch.randn(M, K, device=device, dtype=torch.float32)
            best_ms  = float('inf')
            best_cfg = ""

            for cname, cfg in configs.items():
                Linear.TILE_CONFIG = cfg
                Linear.BACKEND     = "triton"
                layer = Linear(K, N, bias=False)
                layer.weight = torch.randn(N, K, device=device)
                layer._weight_t_padded = None   # force recompute with new config

                try:
                    # Warmup
                    for _ in range(WARMUP): layer(xb)
                    torch.cuda.synchronize()

                    # Timed runs
                    t0 = time.perf_counter()
                    for _ in range(RUNS): layer(xb)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) / RUNS * 1000

                    marker = "  ← best" if ms < best_ms else ""
                    if ms < best_ms:
                        best_ms  = ms
                        best_cfg = cname
                    print(f"    {cname}: {ms:.3f} ms{marker}")

                except Exception as e:
                    print(f"    {cname}: FAILED ({e}) — skipped")

            if best_cfg:
                best[label.strip()] = best_cfg

        # Restore safe default
        Linear.TILE_CONFIG = TILE_CONFIG_B

        print("\n--- Summary: best config per shape ---")
        for lbl, cfg in best.items():
            print(f"  {lbl}: {cfg}")

    print("\nAll tests done.")

# ============================================================================
# OPT-1 Benchmark  —  python layers.py
# Tests all three tile configs on encoder + decoder matrix shapes.
# Config C is guarded with try/except as it may OOM on some GPUs.
# ============================================================================

# ============================================================================
# ── Active configuration — CHANGE THESE TO SWITCH TEST RUNS ─────────────────
# ============================================================================
#
#  OPT-1: choose tile config
#    TILE_CONFIG_A  ->  BLOCK_M=32,  BLOCK_N=32,  BLOCK_K=16, warps=2, stages=2
#    TILE_CONFIG_B  ->  BLOCK_M=64,  BLOCK_N=64,  BLOCK_K=32, warps=4, stages=3
#    TILE_CONFIG_C  ->  BLOCK_M=128, BLOCK_N=64,  BLOCK_K=32, warps=8, stages=3
#
Linear.TILE_CONFIG = TILE_CONFIG_C   # <-- change to A, B, or C
Linear.BACKEND     = "triton"        # "triton" or "cublas"

#  OPT-2: kernel fusion
MLP.FUSED        = False              # True = swiglu_fused_kernel
EncoderMLP.FUSED = False             # True = linear_gelu_kernel
# ─────────────────────────────────────────────────────────────────────────────