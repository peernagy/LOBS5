#!/usr/bin/env python3
"""
check_fp8_compat.py

Small script to:
 - let you pick visible CUDA devices (sets CUDA_VISIBLE_DEVICES before JAX import)
 - report JAX / Flax versions and visible devices
 - check CUDA GPU presence and compute capability (Hopper GPUs require ~9.0)
 - compile a tiny FP8 dot and scan the HLO for an FP8 custom-call (f8e4m3fn)
The script attempts to be robust and prints clear guidance.
"""

import os
import re
import sys
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Check JAX/Flax FP8 compatibility and pick visible CUDA devices.")
    p.add_argument("--devices", type=str, default=None,
                   help="Comma-separated list of CUDA device indices to expose (e.g. '0' or '0,1'). If omitted, do not modify CUDA_VISIBLE_DEVICES.")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip the compile+HLO check (useful if you only want to inspect devices/versions).")
    p.add_argument("--verbose", "-v", action="store_true", help="Print extra debug info.")
    return p.parse_args()

def set_visible_devices(devices_str):
    if devices_str is None:
        return
    # Sanity-strip spaces
    devices_str = devices_str.replace(" ", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
    print(f"Set CUDA_VISIBLE_DEVICES={devices_str}")

def main():
    args = parse_args()
    set_visible_devices(args.devices)

    # Import JAX and Flax after potentially setting CUDA_VISIBLE_DEVICES
    try:
        import jax
        import jax.numpy as jnp
        import flax
    except Exception as e:
        print("Failed to import JAX/Flax. Ensure they are installed in your environment.")
        print("Import error:", e)
        sys.exit(1)

    print(f"JAX version: {jax.__version__}")
    print(f"Flax version: {flax.__version__}")

    # Backend/platform
    try:
        backend = jax.lib.xla_bridge.get_backend()
        platform = backend.platform
    except Exception:
        # Fall back to device inspection
        platform = "unknown"
    print(f"XLA platform reported: {platform}")

    # List devices
    devices = jax.devices()
    if args.verbose:
        print("Detailed devices:")
        for d in devices:
            print(" ", d)
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    print(f"Total devices visible to JAX: {len(devices)}; GPU devices: {len(gpu_devices)}")

    if len(gpu_devices) == 0:
        print("\nNo GPUs detected by JAX. FP8 via XLA requires NVIDIA GPUs (Hopper or later).")
        print("If you expect GPUs, ensure CUDA, cuDNN and a JAX build with GPU support are installed.")
    else:
        print("\nDetected GPU device(s):")
        for i, d in enumerate(gpu_devices):
            # device_kind may contain compute capability info in its string repr
            print(f" [{i}] {d.device_kind} (id: {d.id}, platform: {d.platform})")

    # Check compute capability using the test utility if available
    compute_ok = False
    try:
        # private test util used in Flax docs; may or may not exist depending on JAX version
        from jax._src import test_util as jtu
        compute_ok = jtu.is_cuda_compute_capability_at_least("9.0")
        print(f"Compute capability >= 9.0 (Hopper)?: {compute_ok}")
    except Exception as e:
        print("Could not query compute capability via jax._src.test_util.is_cuda_compute_capability_at_least.")
        if args.verbose:
            print("  (error was: {})".format(e))
        print("This function is not guaranteed to be present for all JAX builds.")
        # Try to infer from device_kind string (best-effort)
        try:
            inferred = any(("compute" in str(d.device_kind).lower()) or ("h100" in str(d.device_kind).lower()) for d in gpu_devices)
            if inferred:
                print("Best-effort inference: device description suggests a recent GPU (may be compatible).")
            else:
                print("Best-effort inference: could not determine compute capability from device descriptions.")
        except Exception:
            pass

    # Attempt to run a tiny FP8 compile test unless skipped
    if args.no_compile:
        print("\nSkipping compile/HLO check (user requested --no-compile).")
        return

    print("\nRunning a small FP8 compile + HLO inspection test. This will JIT-compile a trivial jnp.dot with float8 inputs.")
    try:
        # Use FP8 dtype aliases if available
        try:
            e4m3 = jnp.float8_e4m3fn
        except Exception:
            e4m3 = None

        f32 = jnp.float32

        if e4m3 is None:
            print("This JAX build does not expose jnp.float8_e4m3fn. FP8 dtype is not available.")
            print("You will not get XLA-FP8 custom-calls without a JAX build that recognises float8 dtypes.")
            return

        key = jax.random.PRNGKey(0)
        a = jax.random.uniform(key, (8, 16), dtype=f32)
        b = jax.random.uniform(key, (16, 8), dtype=f32)

        @jax.jit
        def dot_fp8(x, y):
            # cast inputs to FP8 and request accumulation in f32
            return jnp.dot(x.astype(e4m3), y.astype(e4m3), preferred_element_type=f32)

        # Lower the function and compile to HLO
        lowered = dot_fp8.lower(a, b)
        compiled = lowered.compile()
        hlo_text = compiled.as_text()

        # Look for the FP8 custom-call signature used in Flax docs: "custom-call(f8e4m3fn"
        if re.search(r"custom-call\([^)]*f8e4m3fn", hlo_text, flags=re.IGNORECASE):
            print("\nSUCCESS: HLO contains an FP8 custom-call (f8e4m3fn). XLA-FP8 appears to be reachable.")
            print("Note: This does not guarantee full runtime support, but it is a very good sign.")
        else:
            # Some JAX/XLA variants may use different naming (be somewhat tolerant)
            if re.search(r"f8e4m3fn", hlo_text, flags=re.IGNORECASE) or re.search(r"float8", hlo_text, flags=re.IGNORECASE):
                print("\nFP8-like tokens found in HLO, but no explicit custom-call pattern matched.")
                print("HLO excerpt (short):\n", "\n".join(hlo_text.splitlines()[:30]))
            else:
                print("\nNo FP8 custom-call detected in the compiled HLO.")
                print("HLO excerpt (first 30 lines):\n", "\n".join(hlo_text.splitlines()[:30]))
                print("\nPossible reasons:")
                print(" - JAX/XLA in this environment is not built with XLA-FP8 support.")
                print(" - Although FP8 dtypes exist at Python level, the backend may fall back to non-FP8 paths.")
                print(" - The GPU present may not support the FP8 custom-call (Hopper or later GPUs are typically required).")

    except Exception as e:
        print("\nFailed while attempting to compile/inspect HLO.")
        print("Exception:", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("If you are on a remote server, ensure your environment's CUDA/cuDNN drivers and JAX build match and that GPUs are accessible.")
        return

if __name__ == "__main__":
    main()
