#!/usr/bin/env bash
# Runtime verification for the Solar 1-Oct (MXFP4 + FP8 router) image. Needs a
# GPU: importing aiter.fused_moe calls get_gfx(), which shells out to rocminfo,
# so none of this can run during `docker build`.
#
#   docker run --rm --entrypoint verify-fused-router \
#     --device=/dev/kfd --device=/dev/dri --group-add video \
#     -e HIP_VISIBLE_DEVICES=0 vllm-solar:1001-fp8
set -euo pipefail

python3 - <<'PY'
import inspect, importlib.metadata as md
import torch

print(f"torch            {torch.__version__}")
print(f"amd-aiter        {md.version('amd-aiter')}")
print(f"flydsl           {md.version('flydsl')}")

import vllm
print(f"vllm             {vllm.__version__}")

from aiter.jit.utils.chip_info import get_gfx
print(f"gfx              {get_gfx()}")

from aiter.fused_moe import (
    fused_moe_router,
    fused_moe_router_arch_supported,
    fused_moe_router_config_supported,
    fused_moe_router_supported,
)

# Every keyword vllm/_aiter_ops.py forwards must exist, or the fused path
# raises TypeError only once a real decode batch reaches it.
params = inspect.signature(fused_moe_router).parameters
required = (
    "gate_mode", "num_fused_shared_experts", "shared_expert_weight",
    "ep_rank", "ep_size", "hidden_pad", "intermediate_pad",
    "num_local_tokens", "expert_mask",
)
missing = [k for k in required if k not in params]
assert not missing, f"fused_moe_router missing kwargs: {missing}"
print(f"router kwargs    all {len(required)} present")

print(f"arch_supported   {fused_moe_router_arch_supported()}")

from vllm._aiter_ops import rocm_aiter_ops
print(f"has_fused_moe_router()     {rocm_aiter_ops.has_fused_moe_router()}")
print(f"is_fused_router_enabled()  {rocm_aiter_ops.is_fused_router_enabled()}")

# The Solar-Open2 decode shapes: bf16 in, MXFP4 or FP8 per-token experts.
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

MXFP4 = (dtypes.fp4x2, QuantType.per_1x32)
FP8 = (dtypes.fp8, QuantType.per_Token)

# 4096 is Solar-250B, 2048 is the 35B. 1024 must stay refused - phase 1's quant
# geometry needs cols == 256 * TD with a 16-byte load per thread. FP8 weights
# with the MXFP4 quant type (MXFP8) have no router kernel and must be refused.
CASES = (
    (4096, MXFP4, 0, True,  "MXFP4 250B, 80 routed"),
    (4096, MXFP4, 1, True,  "MXFP4 250B, 80 routed + 1 shared"),
    (2048, MXFP4, 0, True,  "MXFP4 35B,  128 routed"),
    (2048, MXFP4, 1, True,  "MXFP4 35B,  128 routed + 1 shared"),
    (1024, MXFP4, 0, False, "MXFP4 unsupported hidden dim"),
    (2048, FP8,   0, True,  "FP8   35B,  128 routed"),
    (2048, FP8,   1, True,  "FP8   35B,  128 routed + 1 shared"),
    (1024, FP8,   0, False, "FP8   unsupported hidden dim"),
    (2048, (dtypes.fp8, QuantType.per_1x32), 0, False, "MXFP8 (no kernel)"),
)
for hidden, (w_dtype, qt), n_shared, want, label in CASES:
    ok = fused_moe_router_config_supported(
        hidden, dtypes.bf16, w_dtype,
        quant_type=qt.value,
        activation=ActivationType.Silu.value,
        gate_mode=GateMode.SEPARATED.value,
        num_fused_shared_experts=n_shared,
    )
    print(f"config_supported ({label:<34}) {ok}")
    assert ok is want, f"{label}: want {want}, got {ok}"

from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterFusedRouterExperts,
    _rocm_aiter_fused_router_quant_method,
)
print(f"AiterFusedRouterExperts    imported ({AiterFusedRouterExperts.__name__})")

# The FP8 oracle must offer the router first while the flag is on, or FP8
# checkpoints silently keep the four-kernel preamble.
import inspect as _inspect
import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle
assert "AiterFusedRouterExperts" in _inspect.getsource(fp8_oracle), \
    "oracle/fp8.py does not offer AiterFusedRouterExperts"
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW8A8Fp8MoEMethod,
)
assert "apply_monolithic" in QuarkW8A8Fp8MoEMethod.__dict__, \
    "QuarkW8A8Fp8MoEMethod.apply_monolithic missing"
print(f"FP8 engine path            oracle + {_rocm_aiter_fused_router_quant_method.__name__} + Quark apply_monolithic")

# The tuned a8w8 table only takes effect if module_gemm_a8w8.so was rebuilt
# from it at image build; a stale prebuilt .so would load without complaint.
import os
from aiter.jit import core
so = os.path.join(core.get_user_jit_dir(), "module_gemm_a8w8.so")
print(f"AITER_CONFIG_GEMM_A8W8     {os.environ.get('AITER_CONFIG_GEMM_A8W8')}")
print(f"module_gemm_a8w8.so        {os.path.getsize(so)} bytes")
assert os.path.getmtime(so) >= os.path.getmtime(os.environ["AITER_CONFIG_GEMM_A8W8"]), \
    "module_gemm_a8w8.so is older than the a8w8 table it should be built from"
print("\nOK")
PY
