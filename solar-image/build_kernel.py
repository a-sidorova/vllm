"""Compile AITER JIT modules at image-build time, with no GPU attached.

    python3 build_kernel.py module_fused_moe_router module_gemm_a8w8

module_fused_moe_router: without this the kernel JITs on the first decode
batch (~35 s) inside the first request, and because that write lands in the
*container's* writable layer rather than the image, every fresh container pays
it again. One .so holds both the MXFP4 and the FP8 per-token kernels.

module_gemm_a8w8: the tuned a8w8 table is compiled into this .so (the codegen
reads AITER_CONFIG_GEMM_A8W8), and AITER never rebuilds an existing .so, so a
changed table only takes effect by rebuilding it here. The codegen keeps the
rows for the build target alone (GPU_ARCHS, and CU_NUM if set).

Two things on the `import aiter` path want a device. Both are worked around
here rather than patched into aiter:

  * chip_info shells out to rocminfo. GPU_ARCHS covers get_gfx(), but
    get_gfx_runtime() - which aiter.utility.dtypes calls at import - ignores
    it by design                                         -> fake rocminfo
  * triton's arch_info probes the active driver and, on failure, falls back to
    jax, which is not installed                          -> stub that module

Only compilation happens here; no kernel is executed, which is what makes it
safe without a GPU. Run under the arch the image targets.
"""

import inspect
import os
import stat
import sys
import tempfile
import time
import types

ARCH = os.environ.setdefault("GPU_ARCHS", "gfx950")
CU_NUM = os.environ.get("SOLAR_CU_NUM", "256")
MD_NAMES = sys.argv[1:] or ["module_fused_moe_router"]

# One GPU agent in rocminfo's layout: chip_info parses the 2-space-indented
# Name / Device Type / ASIC Revision / Compute Unit keys. AITER finds rocminfo
# with shutil.which, so it only has to come first on PATH.
_bin = tempfile.mkdtemp(prefix="fake-rocminfo-")
with open(os.path.join(_bin, "rocminfo"), "w") as f:
    f.write(
        "#!/bin/sh\ncat <<'EOF'\n"
        "Agent 1\n"
        f"  Name:                    {ARCH.split(';')[-1]}\n"
        "  Device Type:             GPU\n"
        "  ASIC Revision:           0(0x0)\n"
        f"  Compute Unit:            {CU_NUM}\n"
        "EOF\n"
    )
os.chmod(os.path.join(_bin, "rocminfo"), stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]

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

accepted = set(inspect.signature(core.build_module).parameters)

for md_name in MD_NAMES:
    # get_args_of_build returns a superset of build_module's signature
    # (hip_clang_path, ...), so filter rather than splat blindly.
    args = core.get_args_of_build(md_name)
    kwargs = {k: v for k, v in args.items() if k in accepted}
    kwargs["md_name"] = md_name
    kwargs.setdefault("verbose", False)

    dropped = sorted(set(args) - accepted)
    if dropped:
        print(f"[note] {md_name}: not passed to build_module: {dropped}", flush=True)

    t0 = time.perf_counter()
    core.build_module(**kwargs)
    elapsed = time.perf_counter() - t0

    so = os.path.join(core.get_user_jit_dir(), f"{md_name}.so")
    if not os.path.isfile(so):
        raise SystemExit(f"build reported success but {so} is missing")
    print(
        f"built {md_name} for {ARCH} in {elapsed:.1f}s ({os.path.getsize(so)} bytes)",
        flush=True,
    )
