# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused KDA branch of ``SolarOpen2KimiDeltaAttention._forward``.

The fused branch differs from the generic (``fused_path=False``) branch of the
same ``_forward`` -- called the base layer below -- in three ways: one conv
over concat[q, k, v] against the whole conv_state,
``beta`` arriving as raw b logits, and decode and speculative decode running a
single ``fused_sigmoid_gating_delta_rule_update`` on the raw gate. These tests
drive the real ``_forward`` on a stub layer:

- CPU, kernel spies: which indices, sequence boundaries and output slice reach
  each kernel, for multi-token non-speculative decode, all-speculative batches
  (with and without full-cudagraph request padding) and mixed speculative +
  prefill batches.
- GPU, real builder: the committed-state fold is bit-identical to the base
  layer's, and decode / speculative output and state match the base layer run
  with the merged conv weight sliced into q/k/v and a gated beta.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

LAYER = "layer"
CONV_WIDTH = 4


class _Spy:
    """Records calls and returns a correctly-shaped stand-in."""

    def __init__(self, ret_fn):
        self.calls = []
        self._ret_fn = ret_fn

    def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self._ret_fn(*args, **kwargs)

    @property
    def call(self):
        assert len(self.calls) == 1, f"expected exactly 1 call, got {len(self.calls)}"
        return self.calls[0]


def _bind(layer, *names):
    from vllm.model_executor.models.solar_open2 import SolarOpen2KimiDeltaAttention

    for name in names:
        setattr(
            layer, name, MethodType(getattr(SolarOpen2KimiDeltaAttention, name), layer)
        )
    return layer


def _fused_layer(kv_cache, conv_weight, A_log, dt_bias, num_spec, head_dim):
    layer = SimpleNamespace(
        fused_path=True,
        prefix=LAYER,
        kv_cache=tuple(kv_cache),
        head_dim=head_dim,
        local_num_heads=A_log.size(2),
        # GatedDeltaNetAttention.__init__ derives this from speculative_config.
        num_spec=num_spec,
        A_log=A_log,
        dt_bias=dt_bias,
        conv1d=SimpleNamespace(weight=conv_weight, bias=None),
        allow_neg_eigval=True,
    )
    return _bind(layer, "_spec_qkv_conv", "_fused_qkv_conv")


def _fused_inputs(g1, b, qkv):
    """Fused-path op inputs: q/k/v as adjacent column slices of one buffer (as
    the fused in_proj produces them) and beta as the raw b logits."""
    q, k, v = qkv.chunk(3, dim=-1)
    return {
        "q_proj_states": q,
        "k_proj_states": k,
        "v_proj_states": v,
        "g1": g1,
        "beta": b.unsqueeze(0),
    }


def _fused_forward(layer, inputs, core_attn_out):
    from vllm.model_executor.models.solar_open2 import SolarOpen2KimiDeltaAttention

    SolarOpen2KimiDeltaAttention._forward(layer, **inputs, core_attn_out=core_attn_out)


# ---------------------------------------------------------------------------
# Decode wiring (CPU)
# ---------------------------------------------------------------------------

S_HEAD_DIM = 4
S_NUM_HEADS = 2
S_PROJ = S_NUM_HEADS * S_HEAD_DIM
S_NUM_BLOCKS = 16


def _install_spies(monkeypatch, meta):
    from vllm.model_executor.models import solar_open2 as mod

    spies = SimpleNamespace(
        conv_update=_Spy(lambda x, *a, **k: torch.zeros_like(x)),
        conv_fn=_Spy(lambda x, *a, **k: torch.zeros_like(x)),
        chunk=_Spy(
            lambda **k: (
                torch.zeros_like(k["v"]),
                torch.zeros_like(k["initial_state"]),
            )
        ),
        fused=_Spy(
            lambda **k: (
                torch.zeros_like(k["v"]) if k["out"] is None else k["out"],
                k["initial_state"],
            )
        ),
    )
    monkeypatch.setattr(mod, "causal_conv1d_update", spies.conv_update)
    monkeypatch.setattr(mod, "causal_conv1d_fn", spies.conv_fn)
    monkeypatch.setattr(mod, "chunk_kda_with_fused_gate", spies.chunk)
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update", spies.fused)
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={LAYER: meta}),
    )
    return spies


def _spy_layer(num_spec):
    return _fused_layer(
        (
            torch.zeros(S_NUM_BLOCKS, CONV_WIDTH - 1 + num_spec, 3 * S_PROJ),
            torch.zeros(S_NUM_BLOCKS, S_NUM_HEADS, S_HEAD_DIM, S_HEAD_DIM),
        ),
        conv_weight=torch.zeros(3 * S_PROJ, 1, CONV_WIDTH),
        A_log=torch.zeros(1, 1, S_NUM_HEADS, 1),
        dt_bias=torch.zeros(S_PROJ),
        num_spec=num_spec,
        head_dim=S_HEAD_DIM,
    )


def _run_spied_decode(monkeypatch, meta, num_spec):
    spies = _install_spies(monkeypatch, meta)
    n = meta.num_actual_tokens
    layer = _spy_layer(num_spec)
    inputs = _fused_inputs(
        torch.zeros(1, n, S_NUM_HEADS, S_HEAD_DIM),
        torch.zeros(n, S_NUM_HEADS),
        torch.zeros(n, 3 * S_PROJ),
    )
    core_attn_out = torch.empty(1, n, S_NUM_HEADS, S_HEAD_DIM)
    _fused_forward(layer, inputs, core_attn_out)
    return spies, core_attn_out


def _decode_metadata(num_seqs, tokens_per_seq):
    total = num_seqs * tokens_per_seq
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_seqs,
        num_decode_tokens=total,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=total,
        non_spec_query_start_loc=torch.arange(
            0, total + 1, tokens_per_seq, dtype=torch.int32
        ),
        non_spec_state_indices_tensor=torch.arange(1, num_seqs + 1, dtype=torch.int32),
    )


def _assert_fused_kernel_covers_all_tokens(spies, meta, core_attn_out):
    assert len(spies.conv_fn.calls) == 0
    assert len(spies.chunk.calls) == 0
    fused = spies.fused.call.kwargs
    n = meta.num_actual_tokens
    assert fused["is_kda"] is True
    assert fused.get("num_accepted_tokens") is None
    assert fused["ssm_state_indices"] is meta.non_spec_state_indices_tensor
    assert torch.equal(
        fused["cu_seqlens"], meta.non_spec_query_start_loc[: meta.num_decodes + 1]
    )
    for name in ("q", "k", "v"):
        assert fused[name].shape[1] == n, name
    assert fused["a"].shape[0] == n
    assert fused["b"].shape[1] == n
    out = fused["out"]
    assert out.shape == (1, n, S_NUM_HEADS, S_HEAD_DIM), (
        "the fused kernel must write every decode token, not one per sequence"
    )
    assert out.data_ptr() == core_attn_out.data_ptr()


@pytest.mark.skip_global_cleanup
def test_single_token_decode_wiring(monkeypatch):
    meta = _decode_metadata(num_seqs=3, tokens_per_seq=1)
    spies, core_attn_out = _run_spied_decode(monkeypatch, meta, num_spec=0)

    conv = spies.conv_update.call.kwargs
    assert torch.equal(
        conv["conv_state_indices"], meta.non_spec_state_indices_tensor[:3]
    )
    assert conv.get("query_start_loc") is None
    assert conv.get("max_query_len", -1) == -1
    assert conv.get("num_accepted_tokens") is None
    assert conv["validate_data"] is True
    _assert_fused_kernel_covers_all_tokens(spies, meta, core_attn_out)


@pytest.mark.skip_global_cleanup
def test_uniform_multi_token_decode_wiring(monkeypatch):
    """Full-cudagraph capture with speculative decoding enabled builds a
    non-speculative decode batch of 1 + K tokens per sequence. The conv must
    index conv_state per sequence and get the sequence boundaries, and the
    fused kernel must write all tokens."""
    num_seqs, tokens_per_seq = 4, 4
    meta = _decode_metadata(num_seqs, tokens_per_seq)
    spies, core_attn_out = _run_spied_decode(
        monkeypatch, meta, num_spec=tokens_per_seq - 1
    )

    conv = spies.conv_update.call.kwargs
    assert conv["conv_state_indices"].numel() == num_seqs
    assert conv.get("query_start_loc") is not None, (
        "with several tokens per sequence the conv needs the sequence boundaries"
    )
    assert torch.equal(
        conv["query_start_loc"], meta.non_spec_query_start_loc[: num_seqs + 1]
    )
    assert conv.get("max_query_len") == tokens_per_seq
    assert conv.get("num_accepted_tokens") is None
    _assert_fused_kernel_covers_all_tokens(spies, meta, core_attn_out)


# ---------------------------------------------------------------------------
# Speculative wiring (CPU)
# ---------------------------------------------------------------------------


def _cu_seqlens(lens: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).int()])


def _spec_metadata(query_lens, is_spec, num_spec, num_reqs_padded=None):
    """Metadata laid out as GDNAttentionMetadataBuilder emits it for a batch
    with speculative requests: every non-drafting request is a prefill, and
    full-cudagraph request padding (all-speculative batches only) pads the
    per-request tensors with null blocks, empty sequences and one accepted
    token."""
    width = num_spec + 1
    num_reqs = len(query_lens)
    blocks = (1 + torch.arange(num_reqs * width, dtype=torch.int32)).view(
        num_reqs, width
    )
    spec_mask = torch.tensor(is_spec)
    lens = torch.tensor(query_lens, dtype=torch.int32)
    total = int(lens.sum())
    num_spec_decodes = int(spec_mask.sum())
    num_accepted = 1 + torch.arange(num_spec_decodes, dtype=torch.int32) % width
    spec_query_start_loc = _cu_seqlens(lens[spec_mask])

    if bool(spec_mask.all()):
        num_pad = (num_reqs_padded or num_reqs) - num_reqs
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=total,
            num_actual_tokens=total + num_pad * width,
            spec_query_start_loc=torch.cat(
                [spec_query_start_loc, torch.full((num_pad,), total, dtype=torch.int32)]
            ),
            spec_state_indices_tensor=torch.cat(
                [blocks, torch.zeros(num_pad, width, dtype=torch.int32)]
            ),
            spec_sequence_masks=torch.cat(
                [spec_mask, torch.zeros(num_pad, dtype=torch.bool)]
            ),
            spec_token_indx=torch.arange(total, dtype=torch.int32),
            non_spec_token_indx=torch.empty(0, dtype=torch.int32),
            num_accepted_tokens=torch.cat(
                [num_accepted, torch.ones(num_pad, dtype=torch.int32)]
            ),
        )

    assert num_reqs_padded is None
    token_is_spec = spec_mask.repeat_interleave(lens.long())
    order = torch.argsort(token_is_spec.int(), stable=True)
    num_ns_tokens = int((~token_is_spec).sum())
    return GDNAttentionMetadata(
        num_prefills=num_reqs - num_spec_decodes,
        num_prefill_tokens=num_ns_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=total - num_ns_tokens,
        num_actual_tokens=total,
        has_initial_state=torch.ones(num_reqs - num_spec_decodes, dtype=torch.bool),
        spec_query_start_loc=spec_query_start_loc,
        non_spec_query_start_loc=_cu_seqlens(lens[~spec_mask]),
        spec_state_indices_tensor=blocks[spec_mask],
        non_spec_state_indices_tensor=blocks[~spec_mask, 0],
        spec_sequence_masks=spec_mask,
        spec_token_indx=order[num_ns_tokens:],
        non_spec_token_indx=order[:num_ns_tokens],
        num_accepted_tokens=num_accepted,
    )


def _run_spied_spec(monkeypatch, meta, num_spec, spies=None):
    """Drive the fused _forward with q/k/v/b as column slices of one in_proj
    buffer that also carries the trailing f_a/g_a columns, as the fused
    projection lays them out."""
    spies = spies or _install_spies(monkeypatch, meta)
    n = meta.num_actual_tokens
    in_proj = torch.randn(n, 3 * S_PROJ + S_NUM_HEADS + 2 * S_HEAD_DIM)
    q, k, v, b, _ = in_proj.split(
        [S_PROJ, S_PROJ, S_PROJ, S_NUM_HEADS, 2 * S_HEAD_DIM], dim=-1
    )
    inputs = {
        "q_proj_states": q,
        "k_proj_states": k,
        "v_proj_states": v,
        "g1": torch.randn(1, n, S_NUM_HEADS, S_HEAD_DIM),
        "beta": b.unsqueeze(0),
    }
    core_attn_out = torch.full((1, n, S_NUM_HEADS, S_HEAD_DIM), float("nan"))
    _fused_forward(_spy_layer(num_spec), inputs, core_attn_out)
    return spies, SimpleNamespace(in_proj=in_proj, out=core_attn_out, **inputs)


def _assert_spec_kernels_wired(spies, meta, num_spec):
    conv = spies.conv_update.call.kwargs
    num_seqs = meta.num_spec_decodes
    assert torch.equal(
        conv["conv_state_indices"], meta.spec_state_indices_tensor[:, 0][:num_seqs]
    ), "conv state lives in ladder column 0"
    assert conv["num_accepted_tokens"] is meta.num_accepted_tokens
    assert conv["query_start_loc"] is meta.spec_query_start_loc
    assert conv["max_query_len"] == num_spec + 1
    assert conv["validate_data"] is False

    fused = spies.fused.call.kwargs
    assert fused["is_kda"] is True and fused["neg_eigval"] is True
    assert fused["inplace_final_state"] is True
    assert fused["ssm_state_indices"] is meta.spec_state_indices_tensor, (
        "the recurrent kernel addresses the whole draft ladder"
    )
    assert fused["num_accepted_tokens"] is meta.num_accepted_tokens
    assert fused["cu_seqlens"].numel() == num_seqs + 1
    assert torch.equal(
        fused["cu_seqlens"], meta.spec_query_start_loc[: num_seqs + 1]
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("padded", [False, True], ids=["exact", "cudagraph_padded"])
def test_all_spec_wiring(monkeypatch, padded):
    """All-speculative batch: one conv on the zero-copy qkv view, one fused
    kernel writing straight into core_attn_out, no prefill kernels. Under
    full-cudagraph padding the per-request tensors are longer than the real
    batch, so every consumer must slice by num_spec_decodes."""
    num_spec, num_reqs = 3, 3
    meta = _spec_metadata(
        [num_spec + 1] * num_reqs,
        [True] * num_reqs,
        num_spec,
        num_reqs_padded=5 if padded else None,
    )
    spies, run = _run_spied_spec(monkeypatch, meta, num_spec)
    n = meta.num_actual_tokens

    assert len(spies.conv_fn.calls) == 0
    assert len(spies.chunk.calls) == 0
    _assert_spec_kernels_wired(spies, meta, num_spec)

    x = spies.conv_update.call.args[0]
    assert x.shape == (n, 3 * S_PROJ)
    assert x.data_ptr() == run.in_proj.data_ptr()
    assert x.stride() == (run.in_proj.stride(0), 1), (
        "the conv input must be the zero-copy view over the in_proj output"
    )

    fused = spies.fused.call.kwargs
    assert fused["a"].shape == (n, S_NUM_HEADS, S_HEAD_DIM)
    assert fused["b"].data_ptr() == run.beta.data_ptr(), "raw b read in place"
    for name in ("q", "k", "v"):
        assert fused[name].shape == (1, n, S_NUM_HEADS, S_HEAD_DIM), name
    out = fused["out"]
    assert out.shape == (1, n, S_NUM_HEADS, S_HEAD_DIM)
    assert out.data_ptr() == run.out.data_ptr()


def _tagged_rows(like: torch.Tensor, base: float) -> torch.Tensor:
    """Row i of the token axis holds base + i."""
    n = like.size(1)
    rows = base + torch.arange(n, dtype=torch.float32)
    return rows.view(1, n, 1, 1).expand(like.shape).contiguous()


@pytest.mark.skip_global_cleanup
def test_mixed_spec_prefill_wiring(monkeypatch):
    """Spec and prefill requests interleaved: the batch is split by one gather
    per half on the fused [n, 3P] view, each half reaches its own kernels, and
    the two outputs are scattered back to batch token order."""
    num_spec = 3
    is_spec = [False, True, False, True]
    meta = _spec_metadata([3, num_spec + 1, 2, num_spec + 1], is_spec, num_spec)
    spec_idx = meta.spec_token_indx.long()
    ns_idx = meta.non_spec_token_indx.long()
    assert not torch.equal(spec_idx, torch.arange(spec_idx.numel()))

    gathers = []
    index_select = torch.Tensor.index_select

    def recording_index_select(self, dim, index):
        gathers.append((tuple(self.shape), dim))
        return index_select(self, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", recording_index_select)
    spies = _install_spies(monkeypatch, meta)
    spies.fused._ret_fn = lambda **k: (
        _tagged_rows(k["v"], 1000),
        k["initial_state"],
    )
    spies.chunk._ret_fn = lambda **k: (
        _tagged_rows(k["v"], 2000),
        k["initial_state"],
    )
    spies, run = _run_spied_spec(monkeypatch, meta, num_spec, spies)
    n = meta.num_actual_tokens

    qkv_gathers = [g for g in gathers if len(g[0]) == 2]
    assert qkv_gathers == [((n, 3 * S_PROJ), 0)] * 2, (
        "split the fused qkv view once per half, not q/k/v separately"
    )
    qkv = run.in_proj[:, : 3 * S_PROJ]

    _assert_spec_kernels_wired(spies, meta, num_spec)
    assert torch.equal(spies.conv_update.call.args[0], qkv[spec_idx])
    assert torch.equal(spies.conv_fn.call.args[0], qkv[ns_idx].T)
    fused = spies.fused.call.kwargs
    assert fused["out"] is None
    assert torch.equal(fused["a"], run.g1[0, spec_idx])
    assert torch.equal(fused["b"], run.beta[:, spec_idx]), "raw b logits"
    chunk = spies.chunk.call.kwargs
    assert torch.equal(chunk["raw_g"], run.g1[:, ns_idx])
    torch.testing.assert_close(
        chunk["beta"], 2.0 * run.beta[:, ns_idx].float().sigmoid()
    )

    out = run.out[0, :, 0, 0]
    assert torch.equal(out[spec_idx], 1000 + torch.arange(spec_idx.numel()).float())
    assert torch.equal(out[ns_idx], 2000 + torch.arange(ns_idx.numel()).float())


# ---------------------------------------------------------------------------
# GPU: real builder, Solar-Open2 TP=4 sizes
# ---------------------------------------------------------------------------

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU for the Triton kernels"
)

NUM_HEADS = 16
HEAD_DIM = 128
PROJ = NUM_HEADS * HEAD_DIM
BLOCK_SIZE = 16
MAX_NUM_SEQS = 64
# 8 request ladders of 1 + K blocks at K = 7, plus blocks no request owns.
NUM_BLOCKS = 96


def _make_builder(num_spec: int, full_cuda_graph: bool = False):
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import SpeculativeConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
    )
    vllm_config.cache_config.enable_prefix_caching = False
    vllm_config.cache_config.mamba_cache_mode = "none"
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY if full_cuda_graph else CUDAGraphMode.PIECEWISE
    )
    if num_spec:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_spec
        )
    shapes = MambaStateShapeCalculator.kda_state_shape(
        1, NUM_HEADS, HEAD_DIM, conv_kernel_size=CONV_WIDTH, num_spec=num_spec
    )
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=shapes,
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="none",
        num_speculative_blocks=num_spec,
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=[LAYER],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )


def _build(
    builder,
    query_lens: list[int],
    num_computed: list[int],
    block_rows: torch.Tensor,
    num_accepted: list[int] | None = None,
    num_reqs_padded: int | None = None,
    num_drafts: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Builder metadata for one batch. ``num_accepted`` is what the previous
    (speculative) step of each request accepted; with it the builder emits the
    committed-state fold metadata.

    ``num_drafts`` makes it a speculative batch: entry i is request i's draft
    count (query length 1 + drafts), or -1 for a request that is not drafting.
    Padded requests then carry no tokens and 1 + K padded token slots each, as
    in a captured speculative graph; without it they carry one dummy token."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    num_reqs = len(query_lens)
    num_reqs_padded = num_reqs_padded or num_reqs
    num_pad = num_reqs_padded - num_reqs
    num_tokens = sum(query_lens)
    qsl = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
    qsl[0] = 0
    np.cumsum(query_lens, out=qsl[1 : num_reqs + 1])
    if num_drafts is None:
        num_tokens_padded = num_tokens + num_pad
        pad_seq_len = 1
        if num_pad:
            qsl[num_reqs + 1 :] = num_tokens + np.arange(1, num_pad + 1)
    else:
        assert num_accepted is not None
        assert all(d < 0 or q == 1 + d for q, d in zip(query_lens, num_drafts))
        num_tokens_padded = num_tokens + num_pad * (builder.num_spec + 1)
        pad_seq_len = 0
    qsl_cpu = torch.from_numpy(qsl)
    seq_lens_cpu = torch.tensor(
        [c + q for c, q in zip(num_computed, query_lens)] + [pad_seq_len] * num_pad,
        dtype=torch.int32,
    )
    block_rows = torch.cat(
        [block_rows, torch.zeros(num_pad, block_rows.size(1), dtype=torch.int32)]
    )
    common = CommonAttentionMetadata(
        query_start_loc=qsl_cpu.cuda(),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq_lens_cpu.cuda(),
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens_padded,
        max_query_len=max(query_lens),
        max_seq_len=int(seq_lens_cpu.max()),
        block_table_tensor=block_rows.cuda(),
        slot_mapping=torch.zeros(num_tokens_padded, dtype=torch.int64, device="cuda"),
        seq_lens_cpu_upper_bound=seq_lens_cpu,
    )
    kwargs = {}
    if num_accepted is not None:
        kwargs["num_accepted_tokens"] = torch.tensor(
            num_accepted + [1] * num_pad, dtype=torch.int32, device="cuda"
        )
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            (num_drafts or [-1] * num_reqs) + [-1] * num_pad, dtype=torch.int32
        )
    return builder.build(0, common, **kwargs)


class _Setup:
    """One fused layer and its base-layer twin over identical weights and an
    identical initial cache; ``run_fused``/``run_base`` drive each on its own
    cache copy."""

    def __init__(self, monkeypatch, num_spec: int):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )
        from vllm.model_executor.models import solar_open2

        self.num_spec = num_spec
        conv_shape, rec_shape = MambaStateShapeCalculator.kda_state_shape(
            1, NUM_HEADS, HEAD_DIM, conv_kernel_size=CONV_WIDTH, num_spec=num_spec
        )
        self.init_conv = torch.randn(
            NUM_BLOCKS, *conv_shape, device="cuda", dtype=torch.bfloat16
        )
        self.init_rec = 0.1 * torch.randn(NUM_BLOCKS, *rec_shape, device="cuda")
        self.conv_weight = 0.5 * torch.randn(3 * PROJ, 1, CONV_WIDTH, device="cuda")
        self.A_log = 0.5 * torch.randn(1, 1, NUM_HEADS, 1, device="cuda") - 1
        self.dt_bias = 0.1 * torch.randn(PROJ, device="cuda")

        self.forward_context = SimpleNamespace(attn_metadata=None)
        monkeypatch.setattr(
            solar_open2, "get_forward_context", lambda: self.forward_context
        )

        # Distinct, shuffled, non-null block ids for each request's ladder.
        ids = 1 + torch.randperm(NUM_BLOCKS - 1)[: 8 * (num_spec + 1)]
        self.req_blocks = ids.view(8, num_spec + 1).to(torch.int32)

    def fresh_cache(self):
        return [self.init_conv.clone(), self.init_rec.clone()]

    def inputs(self, num_tokens: int):
        return SimpleNamespace(
            g1=torch.randn(
                1, num_tokens, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
            ),
            b=torch.randn(num_tokens, NUM_HEADS, device="cuda", dtype=torch.bfloat16),
            qkv=torch.randn(num_tokens, 3 * PROJ, device="cuda", dtype=torch.bfloat16),
        )

    def run_fused(self, meta, x, kv_cache, core_attn_out):
        layer = _fused_layer(
            kv_cache,
            self.conv_weight,
            self.A_log,
            self.dt_bias,
            self.num_spec,
            HEAD_DIM,
        )
        self.forward_context.attn_metadata = {LAYER: meta}
        _fused_forward(
            layer,
            _fused_inputs(x.g1.clone(), x.b.clone(), x.qkv.clone()),
            core_attn_out,
        )

    def run_base(self, meta, x, kv_cache, core_attn_out):
        from vllm.model_executor.models.solar_open2 import (
            SolarOpen2KimiDeltaAttention,
        )

        q_w, k_w, v_w = self.conv_weight.chunk(3, dim=0)
        layer = SimpleNamespace(
            fused_path=False,
            prefix=LAYER,
            kv_cache=tuple(kv_cache),
            head_dim=HEAD_DIM,
            num_spec=self.num_spec,
            allow_neg_eigval=True,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q_conv1d=SimpleNamespace(weight=q_w, bias=None),
            k_conv1d=SimpleNamespace(weight=k_w, bias=None),
            v_conv1d=SimpleNamespace(weight=v_w, bias=None),
        )
        _bind(layer, "_spec_qkv_conv", "_unfused_qkv_conv")
        q, k, v = (t.contiguous() for t in x.qkv.chunk(3, dim=-1))
        self.forward_context.attn_metadata = {LAYER: meta}
        SolarOpen2KimiDeltaAttention._forward(
            layer,
            q_proj_states=q,
            k_proj_states=k,
            v_proj_states=v,
            g1=x.g1.clone(),
            # The base forward gates beta itself: 2 * sigmoid(b) under
            # allow_neg_eigval.
            beta=(2.0 * x.b.float().sigmoid()).unsqueeze(0),
            core_attn_out=core_attn_out,
        )


def _empty_output(num_tokens: int) -> torch.Tensor:
    return torch.full(
        (1, num_tokens, NUM_HEADS, HEAD_DIM),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )


def _accepted(pattern: str, num_reqs: int, num_spec: int) -> list[int]:
    full, mid = num_spec + 1, num_spec // 2 + 1
    return {
        "ones": [1] * num_reqs,
        "mid": [mid] * num_reqs,
        "full": [full] * num_reqs,
        "mixed": ([1, full, mid, 2] * num_reqs)[:num_reqs],
    }[pattern]


def _install_identity_kernels(monkeypatch):
    """Replace every kernel after the fold, in both layers, with one that leaves
    conv_state and recurrent_state untouched, so the cache after _forward is
    exactly what the fold left."""
    from vllm.model_executor.models import solar_open2

    def conv(x, *a, **k):
        return x

    def chunk(**k):
        return torch.zeros_like(k["v"]), k["initial_state"]

    monkeypatch.setattr(solar_open2, "causal_conv1d_update", conv)
    monkeypatch.setattr(solar_open2, "causal_conv1d_fn", conv)
    monkeypatch.setattr(solar_open2, "chunk_kda_with_fused_gate", chunk)
    monkeypatch.setattr(
        solar_open2, "fused_sigmoid_gating_delta_rule_update", lambda **k: None
    )
    monkeypatch.setattr(solar_open2, "fused_kda_gate", lambda g, *a, **k: g)
    monkeypatch.setattr(
        solar_open2,
        "fused_recurrent_kda",
        lambda **k: (torch.zeros_like(k["v"]), k["initial_state"]),
    )


@requires_gpu
@pytest.mark.parametrize("regime", ["decode", "prefill"])
@pytest.mark.parametrize("pattern", ["ones", "mid", "full", "mixed"])
def test_fold_matches_base_layer(monkeypatch, regime, pattern):
    """The fold is pure indexing, so the fused layer's must be bit-identical to
    the base layer's on the same cache, and the identity at num_accepted == 1."""
    num_spec = 3
    torch.manual_seed(0)
    s = _Setup(monkeypatch, num_spec)
    _install_identity_kernels(monkeypatch)
    builder = _make_builder(num_spec)

    reqs = [0, 1, 2, 3]
    query_lens = [1] * 4 if regime == "decode" else [5, 3, 7, 2]
    accepted = _accepted(pattern, len(reqs), num_spec)
    meta = _build(builder, query_lens, [40, 50, 60, 70], s.req_blocks[reqs], accepted)
    assert meta.spec_sequence_masks is None
    assert meta.non_spec_committed_state_indices is not None
    assert meta.non_spec_num_accepted_tokens is not None
    if regime == "decode":
        assert meta.num_decodes == 4 and meta.num_prefills == 0
    else:
        assert meta.num_prefills == 4 and meta.num_decodes == 0
        assert bool(meta.has_initial_state.all())

    x = s.inputs(sum(query_lens))
    fused_kv, base_kv = s.fresh_cache(), s.fresh_cache()
    s.run_fused(meta, x, fused_kv, _empty_output(sum(query_lens)))
    s.run_base(meta, x, base_kv, _empty_output(sum(query_lens)))

    for name, f, b, init in zip(
        ("conv", "recurrent"), fused_kv, base_kv, (s.init_conv, s.init_rec)
    ):
        torch.testing.assert_close(f, b, rtol=0, atol=0, msg=f"{name} state")
        if pattern == "ones":
            assert torch.equal(f, init), f"{name}: fold at num_accepted == 1"
        else:
            assert not torch.equal(f, init), f"{name}: the fold did not run"


@requires_gpu
@pytest.mark.parametrize("case", ["plain", "after_spec_step", "cudagraph_padded"])
def test_single_token_decode_matches_base_layer(monkeypatch, case):
    """Fused decode (one conv over 3P channels + fused gate/recurrent kernel on
    raw g1 and b) against the base layer (three convs + fused_kda_gate +
    fused_recurrent_kda on gated beta). The convs are per-channel identical, so
    conv_state is bit-exact; the recurrent kernels differ, so output (bf16) and
    recurrent state (fp32) are compared at the default dtype tolerances.

    ``after_spec_step`` carries fold metadata with mixed acceptance;
    ``cudagraph_padded`` is a FULL-mode batch padded with a null-block request.
    """
    num_spec = 0 if case == "plain" else 3
    torch.manual_seed(0)
    s = _Setup(monkeypatch, num_spec)
    builder = _make_builder(num_spec, full_cuda_graph=case == "cudagraph_padded")

    reqs = [0, 1, 2, 3, 4]
    query_lens = [1] * len(reqs)
    accepted = None
    num_reqs_padded = None
    if case == "after_spec_step":
        accepted = _accepted("mixed", len(reqs), num_spec)
    elif case == "cudagraph_padded":
        accepted = [1] * len(reqs)
        num_reqs_padded = 8
    meta = _build(
        builder,
        query_lens,
        [30 + 9 * r for r in reqs],
        s.req_blocks[reqs],
        accepted,
        num_reqs_padded,
    )
    assert meta.num_decodes == (num_reqs_padded or len(reqs))
    assert meta.num_prefills == 0 and meta.spec_sequence_masks is None

    n = meta.num_actual_tokens
    x = s.inputs(n)
    fused_kv, base_kv = s.fresh_cache(), s.fresh_cache()
    fused_out, base_out = _empty_output(n), _empty_output(n)
    s.run_fused(meta, x, fused_kv, fused_out)
    s.run_base(meta, x, base_kv, base_out)
    torch.cuda.synchronize()

    num_real = len(reqs)
    assert not torch.isnan(fused_out[0, :num_real]).any()
    torch.testing.assert_close(fused_out[0, :num_real], base_out[0, :num_real])
    torch.testing.assert_close(fused_kv[0], base_kv[0], rtol=0, atol=0)
    torch.testing.assert_close(fused_kv[1], base_kv[1])

    touched = torch.zeros(NUM_BLOCKS, dtype=torch.bool, device="cuda")
    touched[s.req_blocks[reqs].flatten().long()] = True
    assert not torch.equal(fused_kv[1][touched], s.init_rec[touched])
    torch.testing.assert_close(
        fused_kv[1][~touched], s.init_rec[~touched], rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# GPU: speculative steps against the base layer
# ---------------------------------------------------------------------------


def _spec_batch(regime: str, num_spec: int):
    """(query_lens, num_computed, num_drafts, num_reqs_padded) for requests
    0.. of _Setup.req_blocks."""
    k = num_spec
    if regime == "all_spec":
        return [k + 1] * 4, [40, 50, 60, 70], [k] * 4, None
    if regime == "ragged_spec":
        drafts = [k, 1, k - 1, 2]
        return [1 + d for d in drafts], [40, 50, 60, 70], drafts, None
    if regime == "mixed":
        # spec, fresh prefill, spec, a plain decode the builder reclassifies as
        # a prefill, continued prefill
        return [k + 1, 9, k + 1, 1, 6], [40, 0, 60, 33, 20], [k, -1, k, -1, -1], None
    assert regime == "cudagraph_padded"
    return [k + 1] * 3, [40, 50, 60], [k] * 3, 5


def _run_step(s, meta, fused_kv, base_kv):
    n = meta.num_actual_tokens
    x = s.inputs(n)
    fused_out, base_out = _empty_output(n), _empty_output(n)
    s.run_fused(meta, x, fused_kv, fused_out)
    s.run_base(meta, x, base_kv, base_out)
    torch.cuda.synchronize()
    return fused_out, base_out


def _assert_step_matches_base(fused_out, base_out, fused_kv, base_kv, num_tokens):
    """Conv is per-channel, so one conv over 3P channels is bit-exact with
    three over P; the recurrent kernels differ, so output (bf16) and recurrent
    state (fp32) match at the default dtype tolerances."""
    assert not torch.isnan(base_out[0, :num_tokens]).any()
    assert not torch.isnan(fused_out[0, :num_tokens]).any(), (
        "a real token row of core_attn_out was never written"
    )
    torch.testing.assert_close(fused_out[0, :num_tokens], base_out[0, :num_tokens])
    torch.testing.assert_close(
        fused_kv[0], base_kv[0], rtol=0, atol=0, msg="conv state"
    )
    torch.testing.assert_close(fused_kv[1], base_kv[1], msg="recurrent state")


def _assert_untouched_outside(s, kv, blocks: torch.Tensor):
    touched = torch.zeros(NUM_BLOCKS, dtype=torch.bool, device="cuda")
    touched[blocks.flatten().long()] = True
    for name, state, init in zip(("conv", "recurrent"), kv, (s.init_conv, s.init_rec)):
        torch.testing.assert_close(
            state[~touched], init[~touched], rtol=0, atol=0, msg=name
        )


@requires_gpu
@pytest.mark.parametrize("num_spec", [3, 7])
@pytest.mark.parametrize(
    "regime", ["all_spec", "ragged_spec", "mixed", "cudagraph_padded"]
)
@pytest.mark.parametrize("pattern", ["ones", "full", "mixed"])
def test_spec_step_matches_base_layer(monkeypatch, num_spec, regime, pattern):
    """One speculative step through the fused branch (one conv over the whole
    conv_state, fused gate + recurrent kernel on raw g1 and b, 2-D ladder
    indices) against the base layer's generic speculative branch.

    ``pattern`` is what each request accepted on its previous step, i.e.
    which ladder slot the step resumes from. ``mixed`` interleaves spec
    requests with prefills (with and without initial state) and a plain
    decode the builder turns into a prefill; ``cudagraph_padded`` is a
    FULL-mode batch padded with null-block requests."""
    torch.manual_seed(0)
    s = _Setup(monkeypatch, num_spec)
    builder = _make_builder(num_spec, full_cuda_graph=regime == "cudagraph_padded")
    query_lens, num_computed, num_drafts, num_reqs_padded = _spec_batch(
        regime, num_spec
    )
    reqs = list(range(len(query_lens)))
    accepted = [
        1 if d < 0 and q > 1 else a
        for a, q, d in zip(
            _accepted(pattern, len(reqs), num_spec), query_lens, num_drafts
        )
    ]
    meta = _build(
        builder,
        query_lens,
        num_computed,
        s.req_blocks[reqs],
        accepted,
        num_reqs_padded,
        num_drafts,
    )
    assert meta.spec_sequence_masks is not None and meta.num_decodes == 0
    spec_reqs = [r for r in reqs if num_drafts[r] >= 0]
    assert meta.num_spec_decodes == len(spec_reqs)
    if regime == "mixed":
        assert meta.num_prefills == 3
        assert meta.has_initial_state.tolist() == [False, True, True]
    else:
        assert meta.num_prefills == 0
        assert meta.non_spec_state_indices_tensor is None
    if regime == "cudagraph_padded":
        assert builder.use_full_cuda_graph
        assert meta.spec_state_indices_tensor.size(0) == num_reqs_padded
        assert meta.num_actual_tokens > sum(query_lens)

    fused_kv, base_kv = s.fresh_cache(), s.fresh_cache()
    fused_out, base_out = _run_step(s, meta, fused_kv, base_kv)
    _assert_step_matches_base(
        fused_out, base_out, fused_kv, base_kv, sum(query_lens)
    )

    _assert_untouched_outside(s, fused_kv, s.req_blocks[reqs])
    for r in spec_reqs:
        for slot in range(query_lens[r]):
            block = int(s.req_blocks[r, slot])
            assert not torch.equal(fused_kv[1][block], s.init_rec[block]), (
                f"request {r}: ladder slot {slot} was not written"
            )


@requires_gpu
@pytest.mark.parametrize("num_spec", [3, 7])
def test_spec_trajectory_matches_base_layer(monkeypatch, num_spec):
    """Several steps on one cache: all-spec, mixed spec + new prefill, a plain
    decode step that folds the speculative state back, then two spec steps.
    Each step resumes from what the previous one left, so a wrong ladder slot
    or conv window anywhere shows up in a later step."""
    torch.manual_seed(0)
    s = _Setup(monkeypatch, num_spec)
    builder = _make_builder(num_spec)
    k, full, mid = num_spec, num_spec + 1, num_spec // 2 + 1
    num_computed = [100, 110, 120, 0]
    num_accepted = [1, 1, 1, 1]
    # (batch as (request, num_drafts or -1, query_len), tokens each drafting
    # request accepts; a non-drafting request accepts its whole query)
    steps = [
        ([(0, k, full), (1, k, full), (2, k, full)], [full, 1, mid]),
        ([(2, k, full), (3, -1, 7), (0, k, full), (1, k, full)], [2, None, full, mid]),
        ([(0, -1, 1), (1, -1, 1), (2, -1, 1), (3, -1, 1)], [None] * 4),
        ([(3, k, full), (1, k, full), (0, k, full), (2, k, full)], [mid, full, 1, 2]),
        ([(0, k, full), (1, k, full), (2, k, full), (3, k, full)], [2, mid, full, 1]),
    ]

    fused_kv, base_kv = s.fresh_cache(), s.fresh_cache()
    for i, (rows, step_accepted) in enumerate(steps):
        reqs = [r for r, _, _ in rows]
        query_lens = [q for _, _, q in rows]
        num_drafts = [d for _, d, _ in rows]
        spec = any(d >= 0 for d in num_drafts)
        meta = _build(
            builder,
            query_lens,
            [num_computed[r] for r in reqs],
            s.req_blocks[reqs],
            [num_accepted[r] for r in reqs],
            num_drafts=num_drafts if spec else None,
        )
        assert (meta.spec_sequence_masks is not None) == spec
        if not spec:
            assert meta.non_spec_committed_state_indices is not None
        fused_out, base_out = _run_step(s, meta, fused_kv, base_kv)
        try:
            _assert_step_matches_base(
                fused_out, base_out, fused_kv, base_kv, sum(query_lens)
            )
        except AssertionError as e:
            raise AssertionError(f"step {i}: {e}") from e
        for r, q, a in zip(reqs, query_lens, step_accepted):
            num_computed[r] += q if a is None else a
            num_accepted[r] = 1 if a is None else a

    _assert_untouched_outside(s, fused_kv, s.req_blocks[:4])


# ---------------------------------------------------------------------------
# GPU: NaN-poisoned output, both layers
# ---------------------------------------------------------------------------


def _poison_batch(regime: str, num_spec: int):
    """(query_lens, num_computed, num_drafts, num_reqs_padded) for requests 0..
    of _Setup.req_blocks, and the (num_prefills, num_decodes, num_spec_decodes)
    the builder must emit for it."""
    return {
        "pure_prefill": (([5, 3, 7, 2], [0, 12, 0, 30], None, None), (4, 0, 0)),
        "pure_decode": (([1] * 4, [40, 50, 60, 70], None, None), (0, 4, 0)),
        "prefill_decode": (([1, 1, 6, 9], [40, 50, 0, 20], None, None), (2, 2, 0)),
        "cudagraph_padded_decode": (
            ([1] * 5, [30, 39, 48, 57, 66], None, 8),
            (0, 8, 0),
        ),
        "all_spec": (_spec_batch("all_spec", num_spec), (0, 0, 4)),
        "ragged_spec": (_spec_batch("ragged_spec", num_spec), (0, 0, 4)),
        "spec_prefill": (_spec_batch("mixed", num_spec), (3, 0, 2)),
        "cudagraph_padded_all_spec": (
            _spec_batch("cudagraph_padded", num_spec),
            (0, 0, 3),
        ),
    }[regime]


@requires_gpu
@pytest.mark.parametrize("fused_path", [False, True], ids=["base", "fused"])
@pytest.mark.parametrize(
    "regime",
    [
        "pure_prefill",
        "pure_decode",
        "prefill_decode",
        "cudagraph_padded_decode",
        "all_spec",
        "ragged_spec",
        "spec_prefill",
        "cudagraph_padded_all_spec",
    ],
)
def test_every_real_row_is_written(monkeypatch, regime, fused_path):
    """core_attn_out is torch.empty in the model, so a branch that skips a real
    token row leaks garbage downstream. Poison it with NaN and require every
    real row to come back finite."""
    num_spec = 3
    torch.manual_seed(0)
    s = _Setup(monkeypatch, num_spec)
    (query_lens, num_computed, num_drafts, num_reqs_padded), expected = (
        _poison_batch(regime, num_spec)
    )
    builder = _make_builder(num_spec, full_cuda_graph=num_reqs_padded is not None)
    reqs = list(range(len(query_lens)))
    accepted = [
        1 if d < 0 and q > 1 else a
        for a, q, d in zip(
            _accepted("mixed", len(reqs), num_spec),
            query_lens,
            num_drafts or [-1] * len(reqs),
        )
    ]
    meta = _build(
        builder,
        query_lens,
        num_computed,
        s.req_blocks[reqs],
        accepted,
        num_reqs_padded,
        num_drafts,
    )
    assert (meta.num_prefills, meta.num_decodes, meta.num_spec_decodes) == expected
    num_real = sum(query_lens)
    assert (meta.num_actual_tokens > num_real) == (num_reqs_padded is not None)

    n = meta.num_actual_tokens
    out = _empty_output(n)
    run = s.run_fused if fused_path else s.run_base
    run(meta, s.inputs(n), s.fresh_cache(), out)
    torch.cuda.synchronize()

    # Rows past num_real are cudagraph padding, which the model drops.
    written = torch.isfinite(out[0, :num_real]).flatten(1).all(dim=1)
    assert bool(written.all()), (
        f"token rows {torch.nonzero(~written).flatten().tolist()} of "
        f"{num_real} were not written"
    )
