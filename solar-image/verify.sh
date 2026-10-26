#!/usr/bin/env bash
# Runtime verification for the Solar v0.31.0 (MXFP4 + FP8 router) image. Needs
# a GPU: importing aiter.fused_moe calls get_gfx(), which shells out to
# rocminfo, so none of this can run during `docker build`.
#
#   docker run --rm --entrypoint verify-fused-router \
#     --device=/dev/kfd --device=/dev/dri --group-add video \
#     -e HIP_VISIBLE_DEVICES=0 vllm-solar:v031-fp8
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

# The image leaves the runner to vLLM, which picks Model Runner V2. Only a
# launch with the V1-interface SolarOpen2TemplateLogitsProcessor sets it to 0.
import vllm.envs as envs
print(f"VLLM_USE_V2_MODEL_RUNNER   {envs.VLLM_USE_V2_MODEL_RUNNER}")
assert envs.VLLM_USE_V2_MODEL_RUNNER is not False, \
    "VLLM_USE_V2_MODEL_RUNNER=0 is set: the image must not force Model Runner V1"

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

# The router must be compiled into the image, or every fresh container JITs it
# inside its first decode batch.
import os
from aiter.jit import core
so = os.path.join(core.get_user_jit_dir(), "module_fused_moe_router.so")
assert os.path.isfile(so), f"{so} missing: the router was not compiled in"
print(f"module_fused_moe_router.so {os.path.getsize(so)} bytes")

# QuickReduce: the QR+RMSNorm fusion exchanges IPC handles through AITER's
# qr_exchange_handles (else the fusion is silently disabled), and
# VLLM_ROCM_QUICK_REDUCE_FLYDSL=1 dispatches bf16 INT4 to QuickAllReduceInt4.
from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles
from aiter.ops.flydsl import QuickAllReduceInt4
import vllm._aiter_ops as vllm_aiter_ops
from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce
assert hasattr(vllm_aiter_ops, "_aiter_qr_exchange_handles"), \
    "vllm/_aiter_ops.py predates the QR+RMSNorm handle-exchange fix"
assert hasattr(QuickAllReduce, "use_flydsl"), \
    "vllm QuickAllReduce has no FlyDSL INT4 backend"
print(f"QuickReduce                qr_exchange_handles + {QuickAllReduceInt4.__name__} "
      f"(VLLM_ROCM_QUICK_REDUCE_FLYDSL={envs.VLLM_ROCM_QUICK_REDUCE_FLYDSL})")

# The module vLLM's ROCM_AITER_UNIFIED_ATTN backend imports must carry
# aiter#6336 (gfx950 Gluon split grid for spec-decode verify batches).
import aiter.ops.triton.unified_attention as aiter_ua
assert hasattr(aiter_ua, "_gfx950_split_grid"), \
    f"{aiter_ua.__file__} predates aiter#6336"
print(f"unified attention          aiter#6336 ({aiter_ua.__file__})")

# AITER resolves the bf16 table once, at import.
from aiter.jit.core import AITER_CONFIGS
bf16 = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
print(f"AITER_CONFIG_GEMM_BF16     {bf16}")
assert os.environ["AITER_CONFIG_GEMM_BF16"] in bf16, \
    f"AITER does not read the tuned bf16 table: {bf16}"

# The a8w8 table is compiled into module_gemm_a8w8.so; one built before the
# table landed would be the base image's stock build.
a8w8 = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE
print(f"AITER_CONFIG_GEMM_A8W8     {a8w8}")
assert os.environ["AITER_CONFIG_GEMM_A8W8"] in a8w8, \
    f"AITER does not read the tuned a8w8 table: {a8w8}"
so = os.path.join(core.get_user_jit_dir(), "module_gemm_a8w8.so")
assert os.path.isfile(so), f"{so} missing: the a8w8 GEMM was not compiled in"
print(f"module_gemm_a8w8.so        {os.path.getsize(so)} bytes")
assert os.path.getmtime(so) >= os.path.getmtime(os.environ["AITER_CONFIG_GEMM_A8W8"]), \
    "module_gemm_a8w8.so is older than the a8w8 table it should be built from"

# The 250B MoE table reaches AITER only through its default merge of
# model_configs/, which AITER_CONFIG_FMOE replaces. Every row must then resolve
# to its own kernels, i.e. no other table shadows it.
import pandas as pd
import aiter.fused_moe as fm
fmoe_tbl = "/tuning/a4w4_tuned_fmoe_250b_tp4.csv"
assert not os.environ.get("AITER_CONFIG_FMOE"), \
    f"AITER_CONFIG_FMOE={os.environ['AITER_CONFIG_FMOE']} replaces the baked-in MoE tables"
link = os.path.join(os.path.dirname(fm.__file__), "configs", "model_configs",
                    os.path.basename(fmoe_tbl))
assert os.path.realpath(link) == fmoe_tbl, f"{link} does not link to {fmoe_tbl}"
print(f"AITER_CONFIG_FMOE          {AITER_CONFIGS.AITER_CONFIG_FMOE_FILE} (merged)")
t = pd.read_csv(fmoe_tbl)
for r in t.itertuples():
    md = str(fm.get_2stage_cfgs(
        r.token, r.model_dim, r.inter_dim, r.expert, r.topk,
        dtypes.bf16, dtypes.fp4x2, dtypes.fp4x2, QuantType.per_1x32,
        bool(r.use_g1u1), ActivationType.Silu, bool(r.doweight_stage1), 0, 0, True,
    ))
    assert r.kernelName1 in md and r.kernelName2 in md, \
        f"fmoe token={r.token} E={r.expert} topk={r.topk} does not use its tuned kernels: {md[:200]}"
print(f"fmoe 250B MXFP4             all {len(t)} rows resolve to their tuned kernels")
print("\nOK")
PY
