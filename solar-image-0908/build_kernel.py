"""Compile module_fused_moe_router at image-build time, with no GPU attached.

Without this the kernel JITs on the first decode batch (~34 s) inside the
first request, and because that write lands in the *container's* writable
layer rather than the image, every fresh container pays it again.

Two things on the `import aiter` path want a device. Both are worked around
here rather than patched into aiter:

  * chip_info._detect_native() shells out to rocminfo    -> GPU_ARCHS pins it
  * triton's arch_info probes the active driver and, on failure, falls back to
    jax, which is not installed                          -> stub that module

Only compilation happens here; the kernel is never executed, which is what
makes it safe without a GPU. Run under the arch the image targets.
"""

import inspect
import os
import sys
import time
import types

ARCH = os.environ.setdefault("GPU_ARCHS", "gfx950")

# Minimal stand-in for jax._src.lib.gpu_triton. get_arch_details is the only
# thing called, and only to produce an arch string we already know.
_jax = types.ModuleType("jax")
_src = types.ModuleType("jax._src")
_lib = types.ModuleType("jax._src.lib")
_gt = types.ModuleType("jax._src.lib.gpu_triton")
_gt.get_arch_details = lambda _dev: ARCH.split(";")[-1]
_lib.gpu_triton = _gt
_src.lib = _lib
_jax._src = _src
sys.modules.update(
    {
        "jax": _jax,
        "jax._src": _src,
        "jax._src.lib": _lib,
        "jax._src.lib.gpu_triton": _gt,
    }
)

# aiter is installed in the image, not in this repo's environment.
from aiter.jit import core  # type: ignore[import-not-found]  # noqa: E402

MD_NAME = "module_fused_moe_router"

# get_args_of_build returns a superset of build_module's signature
# (hip_clang_path, ...), so filter rather than splat blindly.
args = core.get_args_of_build(MD_NAME)
accepted = set(inspect.signature(core.build_module).parameters)
kwargs = {k: v for k, v in args.items() if k in accepted}
kwargs["md_name"] = MD_NAME
kwargs.setdefault("verbose", False)

dropped = sorted(set(args) - accepted)
if dropped:
    print(f"[note] not passed to build_module: {dropped}", flush=True)

t0 = time.perf_counter()
core.build_module(**kwargs)
elapsed = time.perf_counter() - t0

so = os.path.join(os.path.dirname(core.__file__), f"{MD_NAME}.so")
if not os.path.isfile(so):
    raise SystemExit(f"build reported success but {so} is missing")
print(f"built {MD_NAME} for {ARCH} in {elapsed:.1f}s ({os.path.getsize(so)} bytes)")
