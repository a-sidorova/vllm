#!/usr/bin/env bash
# Runtime verification for the Solar 8-Sep combined image. Needs a GPU:
# importing aiter.fused_moe calls get_gfx(), which shells out to rocminfo, so
# none of this can run during `docker build`.
#
#   docker run --rm --entrypoint verify-fused-router \
#     --device=/dev/kfd --device=/dev/dri --group-add video \
#     -e HIP_VISIBLE_DEVICES=0 vllm-solar:0908-combined
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

# The Solar-Open2 decode shape: bf16 in, mxfp4 weights, hidden 4096.
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

# 4096 is Solar-250B, 2048 is the 35B. 1024 must stay refused - phase 1's quant
# geometry needs cols == 256 * TD with a 16-byte load per thread.
CASES = (
    (4096, 0, True,  "250B, 80 routed"),
    (4096, 1, True,  "250B, 80 routed + 1 shared"),
    (2048, 0, True,  "35B,  128 routed"),
    (2048, 1, True,  "35B,  128 routed + 1 shared"),
    (1024, 0, False, "unsupported hidden dim"),
)
for hidden, n_shared, want, label in CASES:
    ok = fused_moe_router_config_supported(
        hidden, dtypes.bf16, dtypes.fp4x2,
        quant_type=QuantType.per_1x32.value,
        activation=ActivationType.Silu.value,
        gate_mode=GateMode.SEPARATED.value,
        num_fused_shared_experts=n_shared,
    )
    print(f"config_supported ({label:<27}) {ok}")
    assert ok is want, f"hidden={hidden} n_shared={n_shared}: want {want}, got {ok}"

from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterFusedRouterExperts,
)
print(f"AiterFusedRouterExperts    imported ({AiterFusedRouterExperts.__name__})")
print("\nOK")
PY
