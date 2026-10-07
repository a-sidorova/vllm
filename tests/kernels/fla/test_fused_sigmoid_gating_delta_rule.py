# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.kda import (
    fused_kda_gate,
    fused_recurrent_kda,
)
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_non_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    seq_len = 1  # seq_len is 1 for decode
    num_tokens = num_reqs * seq_len
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    # Entry 0 is reserved as NULL_BLOCK_ID (CUDA graph padding), so valid
    # state indices start at 1.
    ssm_state = torch.rand(
        total_entries + 1, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries, dtype=torch.int32) + 1)[:num_tokens]
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_speculative_tokens: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * (num_speculative_tokens + 1)
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    # Entry 0 is reserved as NULL_BLOCK_ID (CUDA graph padding), so valid
    # state indices start at 1.
    ssm_state = torch.rand(
        total_entries + 1, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries, dtype=torch.int32) + 1)[
        :num_tokens
    ].view(num_reqs, num_speculative_tokens + 1)
    num_accepted_tokens = torch.randint(
        1, num_speculative_tokens + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(
        0, num_tokens + 1, num_speculative_tokens + 1, dtype=torch.int32
    )

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


def _make_kda_inputs(
    seq_lens: list[int],
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
):
    num_tokens = sum(seq_lens)
    key_dim = head_k_dim * num_heads
    value_dim = head_v_dim * num_heads
    mixed_qkv = torch.randn(num_tokens, key_dim * 2 + value_dim, dtype=dtype)
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.view(1, num_tokens, num_heads, head_k_dim)
    key = key.view(1, num_tokens, num_heads, head_k_dim)
    value = value.view(1, num_tokens, num_heads, head_v_dim)

    # Parameter shapes/dtypes as in KimiDeltaAttention; `a` is the per-key-dim
    # raw gate and `b` the raw beta logits as the Solar-Open2 decode path
    # passes them.
    A_log = torch.randn(1, 1, num_heads, 1, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(num_heads * head_k_dim, dtype=torch.float32) * 0.1
    a = torch.randn(num_tokens, num_heads, head_k_dim, dtype=dtype)
    b = torch.randn(1, num_tokens, num_heads, dtype=dtype)
    cu_seqlens = torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)
    return query, key, value, A_log, dt_bias, a, b, cu_seqlens


def _run_kda_ref_and_fused(
    query,
    key,
    value,
    A_log,
    dt_bias,
    a,
    b,
    cu_seqlens,
    ssm_state,
    state_indices,
    num_accepted_tokens,
    neg_eigval,
):
    num_tokens, num_heads, head_k_dim = a.shape
    g = fused_kda_gate(
        a.reshape(num_tokens, num_heads * head_k_dim),
        A_log,
        head_k_dim,
        g_bias=dt_bias,
    ).unsqueeze(0)
    beta = b.float().sigmoid()
    if neg_eigval:
        beta = beta * 2.0
    ref_state = ssm_state.clone()
    core_attn_out_ref, _ = fused_recurrent_kda(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=ref_state,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )

    core_attn_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        neg_eigval=neg_eigval,
    )
    return core_attn_out, core_attn_out_ref, ssm_state, ref_state


def _kda_tols(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return dict(atol=1e-4, rtol=1e-4)
    return dict(atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_reqs", [1, 4, 16])
@pytest.mark.parametrize(
    "num_heads,head_k_dim,head_v_dim", [(16, 128, 128), (4, 64, 64)]
)
@pytest.mark.parametrize("neg_eigval", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_sigmoid_gating_delta_rule_update_kda_non_spec(
    num_reqs: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    neg_eigval: bool,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    query, key, value, A_log, dt_bias, a, b, cu_seqlens = _make_kda_inputs(
        [1] * num_reqs, num_heads, head_k_dim, head_v_dim, dtype
    )
    # Slot 0 is NULL_BLOCK_ID (skipped by the kernels); spare slots catch
    # stray writes.
    total_entries = num_reqs * 2 + 1
    ssm_state = torch.randn(
        total_entries, num_heads, head_v_dim, head_k_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries - 1)[:num_reqs] + 1).to(torch.int32)

    out, out_ref, state, state_ref = _run_kda_ref_and_fused(
        query,
        key,
        value,
        A_log,
        dt_bias,
        a,
        b,
        cu_seqlens,
        ssm_state,
        state_indices,
        None,
        neg_eigval,
    )

    torch.testing.assert_close(out, out_ref, **_kda_tols(dtype))
    torch.testing.assert_close(state, state_ref, **_kda_tols(dtype))


@pytest.mark.parametrize("num_reqs", [1, 5])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 7])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize(
    "num_heads,head_k_dim,head_v_dim", [(16, 128, 128), (4, 64, 64)]
)
@pytest.mark.parametrize("neg_eigval", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_sigmoid_gating_delta_rule_update_kda_spec(
    num_reqs: int,
    num_speculative_tokens: int,
    ragged: bool,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    neg_eigval: bool,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    max_len = num_speculative_tokens + 1
    if ragged:
        seq_lens = torch.randint(1, max_len + 1, (num_reqs,)).tolist()
        seq_lens[0] = max_len
    else:
        seq_lens = [max_len] * num_reqs
    query, key, value, A_log, dt_bias, a, b, cu_seqlens = _make_kda_inputs(
        seq_lens, num_heads, head_k_dim, head_v_dim, dtype
    )
    total_entries = num_reqs * max_len * 2 + 1
    ssm_state = torch.randn(
        total_entries, num_heads, head_v_dim, head_k_dim, dtype=dtype
    )
    state_indices = (
        (torch.randperm(total_entries - 1)[: num_reqs * max_len] + 1)
        .to(torch.int32)
        .view(num_reqs, max_len)
    )
    num_accepted_tokens = torch.randint(1, max_len + 1, (num_reqs,), dtype=torch.int32)

    out, out_ref, state, state_ref = _run_kda_ref_and_fused(
        query,
        key,
        value,
        A_log,
        dt_bias,
        a,
        b,
        cu_seqlens,
        ssm_state,
        state_indices,
        num_accepted_tokens,
        neg_eigval,
    )

    torch.testing.assert_close(out, out_ref, **_kda_tols(dtype))
    torch.testing.assert_close(state, state_ref, **_kda_tols(dtype))


@torch.inference_mode()
def test_fused_sigmoid_gating_delta_rule_update_rejects_bad_state_indices() -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    num_reqs, max_len, num_heads, head_dim = 2, 4, 4, 64
    query, key, value, A_log, dt_bias, a, b, cu_seqlens = _make_kda_inputs(
        [max_len] * num_reqs, num_heads, head_dim, head_dim, torch.float32
    )
    ssm_state = torch.randn(num_reqs * max_len * 2 + 1, num_heads, head_dim, head_dim)
    num_accepted_tokens = torch.ones(num_reqs, dtype=torch.int32)
    kwargs = dict(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
    )

    strided_indices = torch.arange(
        1, num_reqs * max_len * 2 + 1, dtype=torch.int32
    ).view(num_reqs, max_len * 2)[:, ::2]
    with pytest.raises(AssertionError, match="contiguous"):
        fused_sigmoid_gating_delta_rule_update(
            **kwargs,
            ssm_state_indices=strided_indices,
            num_accepted_tokens=num_accepted_tokens,
        )

    flat_indices = torch.arange(1, num_reqs + 1, dtype=torch.int32)
    with pytest.raises(AssertionError, match="2-D index tensor"):
        fused_sigmoid_gating_delta_rule_update(
            **kwargs,
            ssm_state_indices=flat_indices,
            num_accepted_tokens=num_accepted_tokens,
        )
