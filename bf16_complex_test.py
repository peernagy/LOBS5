#!/usr/bin/env python3
"""
Compare BF16 vs FP32 complex matrix multiplication (JAX).

Usage:
  python bf16_complex_test.py --batch 4 --m 2048 --k 2048 --n 2048 --runs 5 --seed 0

This script:
 - builds random batched pair-of-real inputs (shape: B, M, K, 2 and B, K, N, 2),
 - computes complex matmul using BF16 real matmuls (two or four real matmuls),
 - computes complex matmul using native complex64 matmul (reference),
 - times both (with JIT warmup), reports best and mean times and effective GFLOPS,
 - reports max absolute error between the two outputs.
"""
from __future__ import annotations
import argparse
import time
import sys
from functools import partial

import os
import jax
import jax.numpy as jnp

# ---------------------- helpers / matmul implementations ---------------------- #
def _pair_to_complex(pair: jnp.ndarray) -> jnp.ndarray:
    """Convert pair-of-reals (..., 2) to complex dtype (...,)."""
    return pair[..., 0] + 1j * pair[..., 1]

def _complex_to_pair(z: jnp.ndarray) -> jnp.ndarray:
    """Convert complex array to pair-of-reals (..., 2)."""
    return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)

@jax.jit
def complex_matmul_fp32(a_pair: jnp.ndarray, b_pair: jnp.ndarray) -> jnp.ndarray:
    """
    Reference complex matmul using complex64 arithmetic.

    a_pair: (B, M, K, 2) float32
    b_pair: (B, K, N, 2) float32
    returns: (B, M, N, 2) float32 (pair-of-reals)
    """
    a_c = _pair_to_complex(a_pair).astype(jnp.complex64)
    b_c = _pair_to_complex(b_pair).astype(jnp.complex64)
    # Batched matmul
    # Using einsum for clarity but jnp.matmul on complex dtype is also fine.
    out_c = jnp.einsum("bmk,bkn->bmn", a_c, b_c)
    return _complex_to_pair(out_c.astype(jnp.complex64)).astype(jnp.float32)


@jax.jit
def complex_matmul_bf16(a_pair: jnp.ndarray, b_pair: jnp.ndarray) -> jnp.ndarray:
    """
    BF16-accelerated complex matmul implemented via real BF16 matmuls.

    a_pair: (B, M, K, 2) float32
    b_pair: (B, K, N, 2) float32

    returns: (B, M, N, 2) float32 (pair-of-reals)
    """
    # Extract real / imag parts and cast to bfloat16
    a_re_bf = a_pair[..., 0].astype(jnp.bfloat16)   # (B, M, K)
    a_im_bf = a_pair[..., 1].astype(jnp.bfloat16)
    b_re_bf = b_pair[..., 0].astype(jnp.bfloat16)   # (B, K, N)
    b_im_bf = b_pair[..., 1].astype(jnp.bfloat16)

    # Compute four real matmuls in BF16
    # rr = a_re @ b_re
    # ii = a_im @ b_im
    # ri = a_re @ b_im
    # ir = a_im @ b_re
    rr = jnp.matmul(a_re_bf, b_re_bf)   # (B, M, N) in bfloat16
    ii = jnp.matmul(a_im_bf, b_im_bf)
    ri = jnp.matmul(a_re_bf, b_im_bf)
    ir = jnp.matmul(a_im_bf, b_re_bf)

    real_bf = rr - ii
    imag_bf = ri + ir

    # Cast back to float32 and pack as pair-of-reals
    real_f32 = real_bf.astype(jnp.float32)
    imag_f32 = imag_bf.astype(jnp.float32)
    return jnp.stack([real_f32, imag_f32], axis=-1)

# ------------------------------ benchmarking --------------------------------- #
def time_jit_fn(fn, *args, runs=5):
    """
    JIT-compile + warmup, then time 'runs' executions.
    Returns (times_list, last_output)
    """
    # Ensure function is jitted (fn might already be jitted)
    compiled = jax.jit(fn)
    # Warmup / compile
    out = compiled(*args)
    out.block_until_ready()

    times = []
    last_out = out
    for i in range(runs):
        t0 = time.perf_counter()
        last_out = compiled(*args)
        last_out.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times, last_out

def print_device_info():
    devices = jax.devices()
    print("JAX devices:")
    for d in devices:
        kind = getattr(d, "device_kind", None)
        print(f" - id={d.id} platform={d.platform} kind={kind} name={d}")

def compute_gflops(batch, m, k, n, best_time):
    """
    Effective complex-equivalent GFLOPS estimate.
    Common approximation: 8 * M*K*N real FLOPs per complex GEMM.
    """
    ops_per_mat = 8 * m * k * n
    total_ops = ops_per_mat * batch
    return total_ops / best_time / 1e9

# ---------------------------------- main ------------------------------------ #
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Use only first GPU if multiple are present
    p = argparse.ArgumentParser(description="Compare BF16 vs FP32 complex matmul (JAX)")
    p.add_argument("--batch", type=int, default=4, help="Batch size (B)")
    p.add_argument("--m", type=int, default=1024, help="M (rows of A output)")
    p.add_argument("--k", type=int, default=1024, help="K (inner dimension)")
    p.add_argument("--n", type=int, default=1024, help="N (cols of B output)")
    p.add_argument("--runs", type=int, default=5, help="Number of timed runs (after warmup)")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = p.parse_args()

    print_device_info()
    print()
    print(f"Benchmark configuration: batch={args.batch}, M={args.m}, K={args.k}, N={args.n}, runs={args.runs}")
    print()

    # Build random inputs in pair-of-reals representation (float32)
    key = jax.random.PRNGKey(args.seed)
    key_a, key_b = jax.random.split(key, 2)
    a_pair = jax.random.normal(key_a, (args.batch, args.m, args.k, 2), dtype=jnp.float32)
    b_pair = jax.random.normal(key_b, (args.batch, args.k, args.n, 2), dtype=jnp.float32)

    # BF16 path timing
    print("Running BF16 path (heavy work in bfloat16)...")
    bf_times, bf_out_pair = time_jit_fn(lambda x, y: complex_matmul_bf16(x, y), a_pair, b_pair, runs=args.runs)
    bf_best = min(bf_times)
    bf_mean = sum(bf_times) / len(bf_times)
    bf_gflops = compute_gflops(args.batch, args.m, args.k, args.n, bf_best)
    print(f" BF16 best: {bf_best:.6f} s   mean: {bf_mean:.6f} s   effective GFLOPS: {bf_gflops:.2f}")

    # FP32/complex64 reference timing
    print("\nRunning FP32/complex64 reference path...")
    fp_times, fp_out_pair = time_jit_fn(lambda x, y: complex_matmul_fp32(x, y), a_pair, b_pair, runs=args.runs)
    fp_best = min(fp_times)
    fp_mean = sum(fp_times) / len(fp_times)
    fp_gflops = compute_gflops(args.batch, args.m, args.k, args.n, fp_best)
    print(f" FP32 best: {fp_best:.6f} s   mean: {fp_mean:.6f} s   effective GFLOPS: {fp_gflops:.2f}")

    # Accuracy comparisons
    print("\nValidating accuracy (max absolute error) between BF16 and FP32 outputs...")
    # convert pair-of-reals to complex64 for both outputs
    bf_out_c = _pair_to_complex(bf_out_pair).astype(jnp.complex64)
    fp_out_c = _pair_to_complex(fp_out_pair).astype(jnp.complex64)

    max_abs_err = jnp.max(jnp.abs(fp_out_c - bf_out_c))
    print(f" Max absolute error (FP32 - BF16): {float(max_abs_err):.6e}")

    # Also report relative error metric (max abs error / max abs of reference)
    denom = jnp.max(jnp.abs(fp_out_c))
    rel_err = float(max_abs_err / (denom + 1e-12))
    print(f" Relative max error: {rel_err:.6e}")

    # Sample values
    print("\nSample outputs (batch=0,row=0,col=0):")
    print(" Reference (complex64):", fp_out_c[0, 0, 0])
    print(" BF16->f32 result     :", bf_out_c[0, 0, 0])

    # Compare speed ratio
    if bf_best > 0 and fp_best > 0:
        speedup = fp_best / bf_best
        print(f"\nSpeedup (FP32_best / BF16_best): {speedup:.3f}x")
    print("\nFinished.")

if __name__ == "__main__":
    main()
