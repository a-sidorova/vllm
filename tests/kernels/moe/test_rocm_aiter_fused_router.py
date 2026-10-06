# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused-router experts (`AiterFusedRouterExperts`).

The selection tests are GPU-free. The end-to-end test compares the fused
router with the unfused path it falls back to (`rocm_aiter_grouped_topk` +
`rocm_aiter_fused_experts`) at the Solar-Open2 35B FP8 shape, and skips unless
the installed AITER fuses FP8 per-token on this GPU.
"""

import importlib.util
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)
if importlib.util.find_spec("aiter") is None:
    pytest.skip("These tests require AITER to run.", allow_module_level=True)

from aiter.utility import dtypes  # noqa: E402

from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (  # noqa: E402
    ActivationMethod,
    AiterExperts,
    AiterFusedRouterExperts,
    QuantMethod,
    _rocm_aiter_fused_router_quant_method,
    rocm_aiter_fused_experts,
    rocm_aiter_fused_router_experts,
    rocm_aiter_grouped_topk,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (  # noqa: E402
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (  # noqa: E402
    Fp8MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (  # noqa: E402
    QuarkW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)

E, TOPK, HIDDEN, INTER = 128, 4, 2048, 1024
# FUSED_MOE_ROUTER_MAX_TOKENS in AITER.
MAX_FUSED_TOKENS = 128


@pytest.mark.parametrize(
    "weight_key,activation_key,expected",
    [
        (kMxfp4Static, kMxfp4Dynamic, (dtypes.fp4x2, QuantMethod.BLOCK_1X32.value)),
        (
            kFp8StaticChannelSym,
            kFp8DynamicTokenSym,
            (dtypes.fp8, QuantMethod.PER_TOKEN.value),
        ),
        (kFp8StaticTensorSym, kFp8StaticTensorSym, (None, None)),
        (kFp8StaticTensorSym, kFp8DynamicTensorSym, (None, None)),
        (kFp8Static128BlockSym, kFp8Dynamic128Sym, (None, None)),
        (None, None, (None, None)),
    ],
)
def test_quant_params(weight_key, activation_key, expected):
    assert (
        AiterFusedRouterExperts._aiter_quant_params(weight_key, activation_key)
        == expected
    )
    assert AiterFusedRouterExperts._supports_quant_scheme(
        weight_key, activation_key
    ) == (expected != (None, None))


def _fp8_config(**kw) -> FusedMoEQuantConfig:
    one = torch.ones(1)
    return fp8_w8a8_moe_quant_config(w1_scale=one, w2_scale=one, **kw)


_MXFP4_CONFIG = FusedMoEQuantConfig(
    _a1=FusedMoEQuantDesc("mxfp4"),
    _a2=FusedMoEQuantDesc("mxfp4"),
    _w1=FusedMoEQuantDesc("mxfp4"),
    _w2=FusedMoEQuantDesc("mxfp4"),
)


@pytest.mark.parametrize(
    "quant_config,expected",
    [
        (_MXFP4_CONFIG, QuantMethod.BLOCK_1X32.value),
        (
            _fp8_config(per_act_token_quant=True, per_out_ch_quant=True),
            QuantMethod.PER_TOKEN.value,
        ),
        # Per-tensor weights, or activations that are not per-token.
        (_fp8_config(per_act_token_quant=True), None),
        (_fp8_config(per_out_ch_quant=True), None),
        (_fp8_config(), None),
        # A static scale the router would silently replace with a dynamic one.
        (
            _fp8_config(
                per_act_token_quant=True,
                per_out_ch_quant=True,
                a1_scale=torch.ones(1),
            ),
            None,
        ),
        (_fp8_config(block_shape=[128, 128]), None),
        (FUSED_MOE_UNQUANTIZED_CONFIG, None),
    ],
)
def test_runtime_quant_method(quant_config, expected):
    assert _rocm_aiter_fused_router_quant_method(quant_config) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_fp8_oracle_offers_fused_router_only_when_enabled(enabled):
    with patch.object(
        rocm_aiter_ops, "is_fused_router_enabled", return_value=enabled
    ):
        classes = backend_to_kernel_cls(Fp8MoeBackend.AITER)
    expected = [AiterFusedRouterExperts, AiterExperts] if enabled else [AiterExperts]
    assert classes == expected


def test_quark_fp8_has_apply_monolithic():
    # A monolithic kernel dispatches through apply_monolithic; the base one
    # raises NotImplementedError.
    assert (
        QuarkW8A8Fp8MoEMethod.apply_monolithic
        is not FusedMoEMethodBase.apply_monolithic
    )


def _fp8_router_available() -> bool:
    return rocm_aiter_ops.fused_moe_router_config_supported(
        hidden_dim=HIDDEN,
        hidden_dtype=torch.bfloat16,
        w1_dtype=dtypes.fp8,
        quant_method=QuantMethod.PER_TOKEN.value,
        activation_method=ActivationMethod.SILU.value,
        gate_mode="",
    )


def _per_channel_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = current_platform.fp8_dtype()
    scale = w.float().abs().amax(dim=-1, keepdim=True) / torch.finfo(fp8).max
    return (w.float() / scale).to(fp8), scale


@pytest.fixture(scope="module")
def fp8_experts():
    torch.manual_seed(0)
    w13, w13_scale = _per_channel_fp8(
        torch.randn(E, 2 * INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) / 16
    )
    w2, w2_scale = _per_channel_fp8(
        torch.randn(E, HIDDEN, INTER, device="cuda", dtype=torch.bfloat16) / 16
    )
    w13, w2 = rocm_aiter_ops.shuffle_weights(w13, w2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    moe_config = make_dummy_moe_config(
        num_experts=E,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
    )
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )
    return w13, w2, moe_config, quant_config


def _unfused(h, logits, bias, w13, w2, moe_config, quant_config):
    topk_weights, topk_ids = rocm_aiter_grouped_topk(
        hidden_states=h,
        gating_output=logits,
        topk=TOPK,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )
    return rocm_aiter_fused_experts(
        hidden_states=h,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=moe_config,
        quant_config=quant_config,
        output_dtype=h.dtype,
    )


# Spans 2-stage (M <= 16) and the 1-stage fmoe_g1u1 path under AITER's default
# heuristic, the fused/split routing boundary (104), and the token cap.
@pytest.mark.parametrize("m", [1, 16, 17, 64, 104, 128, MAX_FUSED_TOKENS + 1])
def test_fp8_fused_router_matches_unfused(m, fp8_experts):
    if not _fp8_router_available():
        pytest.skip("installed AITER does not fuse FP8 per-token routing here")
    w13, w2, moe_config, quant_config = fp8_experts
    torch.manual_seed(m)
    h = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(m, E, device="cuda", dtype=torch.bfloat16)
    # bf16-representable, so the unfused path's bf16 cast of the bias is exact
    # and both paths select the same experts.
    bias = torch.randn(E, device="cuda", dtype=torch.bfloat16).float()

    fused = rocm_aiter_fused_router_experts(
        hidden_states=h,
        w1=w13,
        w2=w2,
        router_logits=logits,
        e_score_correction_bias=bias,
        topk=TOPK,
        moe_config=moe_config,
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
        routed_scaling_factor=1.0,
        quant_config=quant_config,
        output_dtype=h.dtype,
    )
    assert (fused is not None) == (m <= MAX_FUSED_TOKENS)

    experts = AiterFusedRouterExperts(moe_config, quant_config)
    out = experts.apply(
        hidden_states=h,
        w1=w13,
        w2=w2,
        router_logits=logits,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        a1q_scale=None,
        apply_router_weight_on_input=False,
        e_score_correction_bias=bias,
    )

    # The expert kernels accumulate with atomics, so compare against the
    # unfused path's own run-to-run noise rather than bit-exactly.
    ref0 = _unfused(h, logits, bias, w13, w2, moe_config, quant_config).float()
    ref1 = _unfused(h, logits, bias, w13, w2, moe_config, quant_config).float()
    noise = (ref0 - ref1).abs().max().item()
    scale = ref0.abs().max().item()
    assert scale > 0
    for name, got in (("fused", fused), ("apply", out)):
        if got is None:
            continue
        assert torch.isfinite(got).all(), name
        diff = (got.float() - ref0).abs().max().item()
        assert diff <= max(2 * noise, 1e-2 * scale), (
            f"{name}: |fused - unfused| = {diff}, unfused noise {noise}, "
            f"|max| {scale}"
        )
