# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER FP8 blockscale (128x128) bpreshuffle linear path."""

import pytest
import torch

from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and IS_AITER_FOUND)
    or current_platform.is_fp8_fnuz(),
    reason="Requires ROCm OCP-FP8 GPU (gfx950) with aiter",
)

FP8_DTYPE = current_platform.fp8_dtype()
GROUP = 128

# (N, K) of the FP8 linears of Qwen3.8-27B (TP=1); present in the tuned table.
QWEN38_SHAPES = [
    (16384, 5120),
    (14336, 5120),
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
]


@pytest.fixture
def aiter_bpreshuffle_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FP8_BLOCKSCALE_BPRESHUFFLE", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _transpose_scale(scale: torch.Tensor, strided: bool = False) -> torch.Tensor:
    """(M, G) row-major scale -> (M, G) tensor holding the (G, M) layout: either
    aiter's contiguous form or the strided view QuantFP8 native/CUDA return."""
    if strided:
        return scale.t().contiguous().t()
    return scale.t().contiguous().view(*scale.shape)


def _untranspose_scale(scale_t: torch.Tensor) -> torch.Tensor:
    m, g = scale_t.shape
    return scale_t.view(g, m).t()


def _make_weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = (torch.randn(n, k, device="cuda") / 4).to(FP8_DTYPE)
    ws = torch.rand(n // GROUP, k // GROUP, device="cuda") + 0.5
    return w, ws


def _ref_gemm(xq, xs, w, ws) -> torch.Tensor:
    x_dq = xq.float() * xs.repeat_interleave(GROUP, dim=1)
    w_dq = w.float() * ws.repeat_interleave(GROUP, 0).repeat_interleave(GROUP, 1)
    return x_dq @ w_dq.t()


def _assert_gemm_close(out: torch.Tensor, ref: torch.Tensor, tol: float = 2e-2):
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < tol, f"max rel err {err.item():.3e}"


@pytest.mark.parametrize("m", [1, 7, 128, 300])
@pytest.mark.parametrize("k", [512, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_group_fp8_quant_transposed(m: int, k: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    q, s = rocm_aiter_ops.group_fp8_quant(x, GROUP)
    q_t, s_t = rocm_aiter_ops.group_fp8_quant(x, GROUP, transpose_scale=True)
    assert s_t.shape == s.shape and s_t.is_contiguous()
    torch.testing.assert_close(q_t.float(), q.float(), atol=0, rtol=0)
    torch.testing.assert_close(_untranspose_scale(s_t), s, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 7, 300])
@pytest.mark.parametrize("use_aiter", [True, False])
def test_quant_fp8_column_major_scales(m: int, use_aiter: bool):
    """QuantFP8(column_major_scales=True) returns the same values with the same
    column-major strides on the aiter and fallback paths."""
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8

    torch.manual_seed(0)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    q_ref, s_ref = rocm_aiter_ops.group_fp8_quant(x, GROUP)
    with set_current_vllm_config(VllmConfig()):
        quant = QuantFP8(
            static=False, group_shape=GroupShape(1, GROUP), column_major_scales=True
        )
    quant.use_aiter = use_aiter
    q, s = quant.forward_hip(x)
    assert s.shape == s_ref.shape
    assert s.t().is_contiguous()
    if use_aiter:
        torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(s, s_ref, atol=0, rtol=0)
    else:
        # The fallback quantizer may round a few values to the adjacent code,
        # so bound the e4m3 quantization error against x instead.
        torch.testing.assert_close(s, s_ref, atol=0, rtol=1e-5)
        s_full = s.repeat_interleave(GROUP, dim=1)
        err = (q.float() * s_full - x.float()).abs()
        assert (err <= 2**-4 * x.float().abs() + 2**-8 * s_full).all()


@pytest.mark.parametrize("m", [1, 7, 128, 300])
@pytest.mark.parametrize("k", [5120])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_fp8_group_quant_transposed(m: int, k: int, with_residual: bool):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    if with_residual:
        res = torch.randn_like(x)
        op = torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant
        q, r, s = op(x, res, weight, eps, GROUP)
        q_t, r_t, s_t = op(x, res, weight, eps, GROUP, True)
        torch.testing.assert_close(r_t, r, atol=0, rtol=0)
    else:
        op = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant
        q, s = op(x, weight, eps, GROUP)
        q_t, s_t = op(x, weight, eps, GROUP, True)
    assert s_t.shape == s.shape and s_t.is_contiguous()
    torch.testing.assert_close(q_t.float(), q.float(), atol=0, rtol=0)
    torch.testing.assert_close(_untranspose_scale(s_t), s, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 7, 128, 300])
@pytest.mark.parametrize("n", [256, 17408])
def test_act_mul_fp8_group_quant_transposed(m: int, n: int):
    torch.manual_seed(0)
    x = torch.randn(m, 2 * n, device="cuda", dtype=torch.bfloat16)
    op = torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant
    q, s = op(x, GROUP)
    q_t, s_t = op(x, GROUP, True)
    assert q_t.shape == (m, n) and s_t.shape == s.shape and s_t.is_contiguous()
    torch.testing.assert_close(q_t.float(), q.float(), atol=0, rtol=0)
    torch.testing.assert_close(_untranspose_scale(s_t), s, atol=0, rtol=0)


@pytest.mark.parametrize("n,k", QWEN38_SHAPES)
@pytest.mark.parametrize("m", [1, 4, 64, 136, 1000, 4096])
@pytest.mark.parametrize("strided_scale", [False, True])
def test_bpreshuffle_gemm(n: int, k: int, m: int, strided_scale: bool):
    torch.manual_seed(0)
    w, ws = _make_weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    xq, xs = rocm_aiter_ops.group_fp8_quant(x, GROUP)
    out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
        xq,
        rocm_aiter_ops.shuffle_weight(w, layout=(16, 16)),
        _transpose_scale(xs, strided_scale),
        ws,
        output_dtype=torch.bfloat16,
    )
    assert out.dtype == torch.bfloat16
    _assert_gemm_close(out, _ref_gemm(xq, xs, w, ws))


@pytest.mark.parametrize("m", [30848, 32768, 33001])
@pytest.mark.parametrize("strided_scale", [False, True])
def test_bpreshuffle_gemm_chunks_large_output(m: int, strided_scale: bool):
    """gate_up output at M>=30848 exceeds the 2 GiB kernel limit and is
    computed in M-chunks."""
    n, k = 34816, 5120
    assert m * n * 2 >= 2**31
    torch.manual_seed(0)
    w, ws = _make_weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    xq, xs = rocm_aiter_ops.group_fp8_quant(x, GROUP)
    out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
        xq,
        rocm_aiter_ops.shuffle_weight(w, layout=(16, 16)),
        _transpose_scale(xs, strided_scale),
        ws,
        output_dtype=torch.bfloat16,
    )
    # Check rows on both sides of every chunk boundary as well as the tail.
    rows = torch.cat(
        [torch.arange(0, m, 997, device="cuda"), torch.arange(m - 64, m, device="cuda")]
    )
    _assert_gemm_close(out[rows], _ref_gemm(xq[rows], xs[rows], w, ws))


class _Fp8BlockLinear(torch.nn.Module):
    def __init__(self, n: int, k: int, force_kernel=None):
        super().__init__()
        from vllm.model_executor.kernels.linear import init_fp8_linear_kernel

        w, ws = _make_weight(n, k)
        self.ref_weight, self.ref_weight_scale = w.clone(), ws.clone()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(ws, requires_grad=False)
        self.input_scale = None
        self.input_scale_ub = None
        self.weight_block_size = [GROUP, GROUP]
        self.kernel = init_fp8_linear_kernel(
            activation_quant_key=create_fp8_quant_key(
                static=False, group_shape=GroupShape(1, GROUP)
            ),
            weight_quant_key=create_fp8_quant_key(
                static=True, group_shape=GroupShape(GROUP, GROUP)
            ),
            weight_shape=(n, k),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            force_kernel=force_kernel,
        )
        self.kernel.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_weights(self, x)


def test_kernel_selection(aiter_bpreshuffle_env, monkeypatch: pytest.MonkeyPatch):
    from vllm.model_executor.kernels.linear import (
        AiterFp8BlockScaledBPreshuffleMMKernel,
        AiterFp8BlockScaledMMKernel,
    )

    with set_current_vllm_config(VllmConfig()):
        for n, k in QWEN38_SHAPES:
            assert rocm_aiter_ops.is_gemm_a8w8_blockscale_bpreshuffle_tuned(n, k)
            layer = _Fp8BlockLinear(n, k)
            assert type(layer.kernel) is AiterFp8BlockScaledBPreshuffleMMKernel

        # Shapes without tuned rows keep the existing AITER blockscale kernel.
        assert not rocm_aiter_ops.is_gemm_a8w8_blockscale_bpreshuffle_tuned(256, 256)
        layer = _Fp8BlockLinear(256, 256)
        assert type(layer.kernel) is AiterFp8BlockScaledMMKernel

        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FP8_BLOCKSCALE_BPRESHUFFLE", "0")
        rocm_aiter_ops.refresh_env_variables()
        layer = _Fp8BlockLinear(*QWEN38_SHAPES[0])
        assert type(layer.kernel) is AiterFp8BlockScaledMMKernel


@pytest.mark.parametrize("n,k", QWEN38_SHAPES)
@pytest.mark.parametrize("m", [1, 4, 33, 512, 2048])
@pytest.mark.parametrize("use_aiter_quant", [True, False])
def test_bpreshuffle_linear_kernel(
    aiter_bpreshuffle_env, n: int, k: int, m: int, use_aiter_quant: bool
):
    from vllm.model_executor.kernels.linear import (
        AiterFp8BlockScaledBPreshuffleMMKernel,
        AiterFp8BlockScaledMMKernel,
    )

    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        layer = _Fp8BlockLinear(n, k)
        assert type(layer.kernel) is AiterFp8BlockScaledBPreshuffleMMKernel
        # False: QuantFP8 falls back to the CUDA path's strided column-major scale.
        layer.kernel.quant_fp8.use_aiter = use_aiter_quant
        assert layer.weight.shape == (n, k)
        assert not torch.equal(
            layer.weight.view(torch.uint8), layer.ref_weight.view(torch.uint8)
        )

        baseline = _Fp8BlockLinear(n, k, force_kernel=AiterFp8BlockScaledMMKernel)
        assert type(baseline.kernel) is AiterFp8BlockScaledMMKernel
        baseline.weight.data.copy_(layer.ref_weight)
        baseline.weight_scale_inv.data.copy_(layer.ref_weight_scale)

        x = torch.randn(3, m, k, device="cuda", dtype=torch.bfloat16)
        out = layer(x)
        out_base = baseline(x)
        assert out.shape == (3, m, n)

        xq, xs = rocm_aiter_ops.group_fp8_quant(x.view(-1, k), GROUP)
        ref = _ref_gemm(xq, xs, layer.ref_weight, layer.ref_weight_scale)
        _assert_gemm_close(out.view(-1, n), ref)
        _assert_gemm_close(out.view(-1, n), out_base.view(-1, n).float())
