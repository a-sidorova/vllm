# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring contract for the Solar-Open2 KDA speculative-decoding path.

``SolarOpen2KimiDeltaAttention._forward`` routes a batch to up to four Triton
kernels. Which index tensor reaches which kernel, along which axis each tensor
is gathered, and whether ``num_accepted_tokens`` survives the trip are the parts
that fail *silently* -- every shape stays valid and the model just conditions on
the wrong recurrent state.

Numerical correctness needs a GPU (see test_kda_spec_decode_cudagraph.py). This
module pins the wiring instead, by replacing each kernel with a spy and
asserting on the arguments it received, so it runs on CPU.

Regimes covered (the names match tests/v1/attention/test_gdn_metadata_builder.py):
  pure_prefill, pure_regular_decode, pure_spec_decode,
  mixed_decode_and_spec_decode, prefill_decode_and_spec_decode
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

pytestmark = pytest.mark.skip_global_cleanup

HEAD_DIM = 4
NUM_HEADS = 2
PROJ = NUM_HEADS * HEAD_DIM  # per-stream projection width
CONV_WIDTH = 4
NUM_BLOCKS = 16


class _Spy:
    """Records calls and returns a correctly-shaped stand-in."""

    def __init__(self, shape_fn):
        self.calls = []
        self._shape_fn = shape_fn

    def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self._shape_fn(*args, **kwargs)

    @property
    def call(self):
        assert len(self.calls) == 1, f"expected exactly 1 call, got {len(self.calls)}"
        return self.calls[0]


def _install_spies(monkeypatch, chunk_fill=0.0, recurrent_fill=0.0):
    from vllm.model_executor.models import solar_open2 as mod

    # causal_conv1d_update(x, conv_state, weight, bias, ...) -> same shape as x
    conv_update = _Spy(lambda x, *a, **k: torch.zeros_like(x))
    # causal_conv1d_fn(x=(dim, tokens), ...) -> same shape
    conv_fn = _Spy(lambda x, *a, **k: torch.zeros_like(x))

    # (q=(1, tokens, h, d), ...) -> (out, last_state) where last_state carries
    # one entry per *sequence*, i.e. len(cu_seqlens) - 1.
    def _kernel_ret(fill):
        def _ret(**k):
            num_seqs = k["cu_seqlens"].numel() - 1
            return (
                torch.full_like(k["q"], fill),
                torch.zeros(num_seqs, NUM_HEADS, HEAD_DIM, HEAD_DIM),
            )

        return _ret

    chunk = _Spy(_kernel_ret(chunk_fill))
    recurrent = _Spy(_kernel_ret(recurrent_fill))
    gate = _Spy(lambda g, *a, **k: torch.zeros_like(g))

    monkeypatch.setattr(mod, "causal_conv1d_update", conv_update)
    monkeypatch.setattr(mod, "causal_conv1d_fn", conv_fn)
    monkeypatch.setattr(mod, "chunk_kda_with_fused_gate", chunk)
    monkeypatch.setattr(mod, "fused_recurrent_kda", recurrent)
    monkeypatch.setattr(mod, "fused_kda_gate", gate)
    return SimpleNamespace(
        conv_update=conv_update,
        conv_fn=conv_fn,
        chunk=chunk,
        recurrent=recurrent,
        gate=gate,
    )


def _conv_state(num_spec: int) -> torch.Tensor:
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    state_len = CONV_WIDTH - 1 + num_spec
    if is_conv_state_dim_first():
        return torch.zeros(NUM_BLOCKS, 3 * PROJ, state_len)
    return torch.zeros(NUM_BLOCKS, state_len, 3 * PROJ)


def _make_layer(num_spec: int, conv_state=None, recurrent_state=None):
    from vllm.model_executor.models.solar_open2 import SolarOpen2KimiDeltaAttention

    if conv_state is None:
        conv_state = _conv_state(num_spec)
    if recurrent_state is None:
        recurrent_state = torch.zeros(NUM_BLOCKS, NUM_HEADS, HEAD_DIM, HEAD_DIM)

    def _conv1d():
        return SimpleNamespace(
            weight=torch.zeros(PROJ, 1, CONV_WIDTH), bias=torch.zeros(PROJ)
        )

    layer = SimpleNamespace(
        prefix="layer",
        kv_cache=(conv_state, recurrent_state),
        head_dim=HEAD_DIM,
        num_spec=num_spec,
        fused_path=False,
        allow_neg_eigval=True,
        A_log=torch.zeros(1, 1, NUM_HEADS, 1),
        dt_bias=torch.zeros(NUM_HEADS * HEAD_DIM),
        q_conv1d=_conv1d(),
        k_conv1d=_conv1d(),
        v_conv1d=_conv1d(),
    )
    for name in ("_spec_qkv_conv", "_unfused_qkv_conv"):
        setattr(
            layer, name, MethodType(getattr(SolarOpen2KimiDeltaAttention, name), layer)
        )
    return layer


def _run_forward(
    monkeypatch,
    meta: GDNAttentionMetadata,
    num_spec: int = 0,
    layer=None,
    beta=None,
    chunk_fill=0.0,
    recurrent_fill=0.0,
):
    """Drive the real _forward over stub kernels and return the spies."""
    from vllm.model_executor.models import solar_open2 as mod

    spies = _install_spies(monkeypatch, chunk_fill, recurrent_fill)
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": meta}),
    )

    if layer is None:
        layer = _make_layer(num_spec)
    n = meta.num_actual_tokens
    if beta is None:
        beta = torch.zeros(1, n, NUM_HEADS)
    core_attn_out = torch.zeros(1, n, NUM_HEADS, HEAD_DIM)
    mod.SolarOpen2KimiDeltaAttention._forward(
        layer,
        q_proj_states=torch.zeros(n, PROJ),
        k_proj_states=torch.zeros(n, PROJ),
        v_proj_states=torch.zeros(n, PROJ),
        g1=torch.zeros(1, n, NUM_HEADS, HEAD_DIM),
        beta=beta,
        core_attn_out=core_attn_out,
    )
    spies.core_attn_out = core_attn_out
    return spies


def _spec_metadata(
    *,
    num_spec_decodes,
    num_spec,
    num_prefills=0,
    num_prefill_tokens=0,
    cudagraph_pad_to=0,
    prefill_first=False,
):
    """A batch of `num_spec_decodes` drafting requests, plus optional prefill.

    ``cudagraph_pad_to`` reproduces the FULL cudagraph branch of
    ``GDNAttentionMetadataBuilder.build``, which pads ``spec_query_start_loc``
    out to ``batch_size + 1`` and ``num_accepted_tokens`` out to ``batch_size``
    so captured shapes stay static. Consumers must therefore slice by
    ``num_spec_decodes`` rather than trust the length.
    """
    spec_tokens = num_spec_decodes * (num_spec + 1)
    total = spec_tokens + num_prefill_tokens
    # Block ids start at 1: 0 is NULL_BLOCK_ID and the kernels skip it.
    spec_state = (
        1 + torch.arange(num_spec_decodes * (num_spec + 1), dtype=torch.int32)
    ).view(num_spec_decodes, num_spec + 1)

    spec_qsl = torch.arange(0, spec_tokens + 1, num_spec + 1, dtype=torch.int32)
    num_accepted = torch.full((num_spec_decodes,), num_spec + 1, dtype=torch.int32)
    if cudagraph_pad_to:
        batch_size = cudagraph_pad_to
        assert batch_size >= num_spec_decodes
        # Trailing entries repeat the final offset, i.e. zero-length sequences.
        spec_qsl = torch.cat(
            [
                spec_qsl,
                torch.full(
                    (batch_size - num_spec_decodes,), spec_tokens, dtype=torch.int32
                ),
            ]
        )
        num_accepted = torch.cat(
            [num_accepted, torch.ones(batch_size - num_spec_decodes, dtype=torch.int32)]
        )
        spec_state = torch.cat(
            [
                spec_state,
                torch.zeros(
                    batch_size - num_spec_decodes, num_spec + 1, dtype=torch.int32
                ),
            ]
        )

    if prefill_first:
        non_spec_token_indx = torch.arange(num_prefill_tokens, dtype=torch.int64)
        spec_token_indx = torch.arange(num_prefill_tokens, total, dtype=torch.int64)
    else:
        spec_token_indx = torch.arange(spec_tokens, dtype=torch.int64)
        non_spec_token_indx = torch.arange(spec_tokens, total, dtype=torch.int64)

    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=spec_tokens,
        num_actual_tokens=total,
        has_initial_state=(
            torch.zeros(num_prefills, dtype=torch.bool) if num_prefills else None
        ),
        spec_query_start_loc=spec_qsl,
        non_spec_query_start_loc=(
            torch.tensor([0, num_prefill_tokens], dtype=torch.int32)
            if num_prefills
            else None
        ),
        spec_state_indices_tensor=spec_state,
        non_spec_state_indices_tensor=(
            torch.tensor([15], dtype=torch.int32) if num_prefills else None
        ),
        spec_sequence_masks=torch.ones(num_spec_decodes, dtype=torch.bool),
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted,
    )


# ---------------------------------------------------------------------------
# Regressions: the non-speculative regimes must be untouched
# ---------------------------------------------------------------------------


def test_pure_prefill_still_uses_the_chunk_kernel(monkeypatch):
    meta = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=6,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=6,
        has_initial_state=torch.zeros(1, dtype=torch.bool),
        non_spec_query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([3], dtype=torch.int32),
    )
    spies = _run_forward(monkeypatch, meta)

    assert len(spies.conv_fn.calls) == 3, "one causal_conv1d_fn per q/k/v stream"
    assert len(spies.chunk.calls) == 1
    assert len(spies.recurrent.calls) == 0
    assert len(spies.conv_update.calls) == 0


def test_pure_regular_decode_still_uses_the_recurrent_kernel(monkeypatch):
    meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([3, 4], dtype=torch.int32),
    )
    spies = _run_forward(monkeypatch, meta)

    assert len(spies.conv_update.calls) == 3
    assert len(spies.recurrent.calls) == 1
    assert len(spies.chunk.calls) == 0
    # A non-speculative batch must not claim accepted tokens.
    assert spies.recurrent.call.kwargs.get("num_accepted_tokens") is None
    for c in spies.conv_update.calls:
        assert c.kwargs.get("num_accepted_tokens") is None
        # One token per sequence: the kernel infers the batch from x.shape.
        assert c.kwargs.get("query_start_loc") is None


# ---------------------------------------------------------------------------
# The speculative path
# ---------------------------------------------------------------------------


def test_pure_spec_decode_does_not_touch_the_non_spec_path(monkeypatch):
    """The all-speculative batch is the one where non_spec tensors are None."""
    meta = _spec_metadata(num_spec_decodes=2, num_spec=3)
    spies = _run_forward(monkeypatch, meta, num_spec=3)

    assert len(spies.conv_update.calls) == 3, "one per q/k/v stream"
    assert len(spies.recurrent.calls) == 1
    assert len(spies.chunk.calls) == 0
    assert len(spies.conv_fn.calls) == 0


def test_spec_conv_uses_column_zero_and_forwards_accepted_tokens(monkeypatch):
    """Conv takes the *first* block column; the recurrent kernel takes all of them.

    Swapping them is silent: both are valid, in-bounds block ids.
    """
    meta = _spec_metadata(num_spec_decodes=2, num_spec=3)
    spies = _run_forward(monkeypatch, meta, num_spec=3)

    expected_conv_idx = meta.spec_state_indices_tensor[:, 0][: meta.num_spec_decodes]
    for c in spies.conv_update.calls:
        assert torch.equal(c.kwargs["conv_state_indices"], expected_conv_idx)
        assert c.kwargs["num_accepted_tokens"] is meta.num_accepted_tokens
        assert c.kwargs["query_start_loc"] is meta.spec_query_start_loc
        assert c.kwargs["max_query_len"] == meta.spec_state_indices_tensor.size(-1)
        # The speculative conv window is longer than the plain-decode one, so
        # the kernel's own bounds check does not apply.
        assert c.kwargs["validate_data"] is False

    rec = spies.recurrent.call.kwargs
    assert rec["ssm_state_indices"] is meta.spec_state_indices_tensor, (
        "recurrent kernel needs the full ladder, not just column 0"
    )
    assert rec["num_accepted_tokens"] is meta.num_accepted_tokens
    assert torch.equal(
        rec["cu_seqlens"],
        meta.spec_query_start_loc[: meta.num_spec_decodes + 1],
    )


def test_spec_conv_uses_a_distinct_state_and_weight_per_stream(monkeypatch):
    """q/k/v are three independent conv streams over three views of one cache.

    Passing the same view three times would silently convolve q's history into
    k and v.
    """
    meta = _spec_metadata(num_spec_decodes=2, num_spec=3)
    spies = _run_forward(monkeypatch, meta, num_spec=3)

    states = [c.args[1] for c in spies.conv_update.calls]
    assert len({s.data_ptr() for s in states}) == 3, (
        "each stream must get its own conv_state chunk"
    )
    for s in states:
        assert s.shape[-2] == PROJ, "chunk(3, dim=-2) splits the channel axis"


def test_mixed_batch_gathers_spec_and_non_spec_on_the_right_axes(monkeypatch):
    """q/k/v are (tokens, dim) but g1/beta carry a leading batch axis.

    Gathering g1/beta on dim 0 would select along the size-1 batch axis and
    quietly return the whole tensor.
    """
    meta = _spec_metadata(
        num_spec_decodes=2, num_spec=3, num_prefills=1, num_prefill_tokens=5
    )
    spies = _run_forward(monkeypatch, meta, num_spec=3)

    assert len(spies.chunk.calls) == 1, "the non-spec half still runs"
    assert len(spies.recurrent.calls) == 1, "the spec half runs too"

    spec_tokens = meta.num_spec_decode_tokens
    rec = spies.recurrent.call.kwargs
    assert rec["q"].shape[1] == spec_tokens
    assert rec["g"].shape[1] == spec_tokens, "g1 must be gathered on dim 1"
    assert rec["beta"].shape[1] == spec_tokens, "beta must be gathered on dim 1"

    chunk = spies.chunk.call.kwargs
    assert chunk["q"].shape[1] == meta.num_prefill_tokens
    assert chunk["raw_g"].shape[1] == meta.num_prefill_tokens
    assert chunk["beta"].shape[1] == meta.num_prefill_tokens


def test_spec_half_is_gated_locally_and_prefill_half_stays_raw(monkeypatch):
    """The recurrent kernel takes a pre-gated g; the chunk kernel gates its own."""
    meta = _spec_metadata(
        num_spec_decodes=2, num_spec=3, num_prefills=1, num_prefill_tokens=5
    )
    spies = _run_forward(monkeypatch, meta, num_spec=3)

    assert len(spies.gate.calls) == 1, (
        "fused_kda_gate applies to the speculative half only"
    )
    assert spies.gate.call.args[0].shape[0] == meta.num_spec_decode_tokens


def test_beta_is_not_re_sigmoided(monkeypatch):
    """The unfused ``forward`` already applies ``sigmoid`` (and the neg-eigval
    ``* 2``) to beta, so ``_forward`` must hand it to the kernels unchanged."""
    meta = _spec_metadata(
        num_spec_decodes=2,
        num_spec=3,
        num_prefills=1,
        num_prefill_tokens=5,
        prefill_first=True,
    )
    beta = torch.linspace(-1.0, 1.0, meta.num_actual_tokens * NUM_HEADS).view(
        1, meta.num_actual_tokens, NUM_HEADS
    )
    spies = _run_forward(monkeypatch, meta, num_spec=3, beta=beta)

    assert torch.equal(
        spies.recurrent.call.kwargs["beta"],
        beta.index_select(1, meta.spec_token_indx),
    )
    assert torch.equal(
        spies.chunk.call.kwargs["beta"],
        beta.index_select(1, meta.non_spec_token_indx),
    )


def test_mixed_batch_merge_restores_original_token_order(monkeypatch):
    """Outputs come back split; they must be scattered back by token index."""
    meta = _spec_metadata(
        num_spec_decodes=2,
        num_spec=3,
        num_prefills=1,
        num_prefill_tokens=5,
        prefill_first=True,
    )
    spies = _run_forward(
        monkeypatch, meta, num_spec=3, chunk_fill=2.0, recurrent_fill=1.0
    )

    out = spies.core_attn_out[0]
    assert torch.all(out[meta.non_spec_token_indx] == 2.0)
    assert torch.all(out[meta.spec_token_indx] == 1.0)


def test_cudagraph_padded_batch_slices_by_num_spec_decodes(monkeypatch):
    """Under FULL cudagraph the speculative tensors are padded to batch_size.

    The recurrent kernel derives its sequence count from ``len(cu_seqlens) - 1``,
    so handing it the padded tensor would run zero-length phantom sequences
    against padded state-index rows. The conv takes its batch from
    ``conv_state_indices``, so the padded ``query_start_loc`` is harmless there.
    """
    meta = _spec_metadata(num_spec_decodes=2, num_spec=3, cudagraph_pad_to=5)
    assert meta.spec_query_start_loc.numel() == 6, "fixture must model the padding"

    spies = _run_forward(monkeypatch, meta, num_spec=3)

    rec = spies.recurrent.call.kwargs
    assert rec["cu_seqlens"].numel() == meta.num_spec_decodes + 1

    for c in spies.conv_update.calls:
        assert c.kwargs["query_start_loc"] is meta.spec_query_start_loc
        assert c.kwargs["conv_state_indices"].numel() == meta.num_spec_decodes


def test_uniform_multi_token_decode_indexes_conv_state_by_sequence(monkeypatch):
    """A non-speculative decode batch can carry more than one token per sequence.

    Speculation widens the reorder threshold to ``num_spec + 1``, so a uniform
    multi-token batch without drafts (FULL cudagraph capture) is still a
    "decode". ``non_spec_state_indices_tensor`` holds one block id per
    *sequence*, so slicing it by the *token* count is wrong, and the kernel
    needs the sequence boundaries.
    """
    num_seqs, tokens_per_seq = 4, 4
    total = num_seqs * tokens_per_seq
    meta = GDNAttentionMetadata(
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
    spies = _run_forward(monkeypatch, meta, num_spec=tokens_per_seq - 1)

    assert len(spies.conv_update.calls) == 3
    for c in spies.conv_update.calls:
        assert c.kwargs["conv_state_indices"].numel() == num_seqs
        assert c.kwargs.get("query_start_loc") is not None
        assert c.kwargs.get("max_query_len") == tokens_per_seq


# ---------------------------------------------------------------------------
# Leaving the speculative path: fold the committed state back
# ---------------------------------------------------------------------------


def test_non_spec_step_folds_committed_state_to_canonical_position(monkeypatch):
    """After a speculative step the state sits at ladder column
    ``num_accepted - 1`` and the conv window is rolled by the same amount. A
    request that stops drafting must be folded back before any non-speculative
    reader runs.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    num_spec = 3
    conv_state = _conv_state(num_spec)
    conv_view = conv_state if is_conv_state_dim_first() else conv_state.transpose(1, 2)
    state_len = conv_view.size(-1)
    conv_view.copy_(
        torch.arange(state_len, dtype=conv_state.dtype).expand_as(conv_view)
        + 100 * torch.arange(NUM_BLOCKS, dtype=conv_state.dtype).view(-1, 1, 1)
    )
    recurrent_state = torch.arange(NUM_BLOCKS, dtype=torch.float32).view(
        -1, 1, 1, 1
    ) * torch.ones(NUM_BLOCKS, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    layer = _make_layer(num_spec, conv_state, recurrent_state)

    # Request A accepted 3 tokens (state in ladder block 6), B accepted 1.
    blk0 = torch.tensor([4, 8], dtype=torch.int32)
    committed = torch.tensor([6, 8], dtype=torch.int32)
    accepted = torch.tensor([3, 1], dtype=torch.int32)
    conv_before = conv_view[blk0.long()].clone()
    meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        non_spec_state_indices_tensor=blk0,
        non_spec_committed_state_indices=committed,
        non_spec_num_accepted_tokens=accepted,
    )
    _run_forward(monkeypatch, meta, num_spec=num_spec, layer=layer)

    assert torch.all(recurrent_state[4] == 6.0), "state copied from ladder block"
    assert torch.all(recurrent_state[8] == 8.0), "accepted=1 is a no-op"

    shifted = conv_view[4]
    expected = torch.cat(
        [conv_before[0][:, 2:], conv_before[0][:, -1:].expand(-1, 2)], dim=-1
    )
    assert torch.equal(shifted, expected), "conv window left-shifted by accepted-1"
    assert torch.equal(conv_view[8], conv_before[1]), "accepted=1 is a no-op"


def test_fused_path_refuses_spec_metadata(monkeypatch):
    meta = _spec_metadata(num_spec_decodes=2, num_spec=3)
    layer = _make_layer(3)
    layer.fused_path = True
    with pytest.raises(NotImplementedError, match="speculative"):
        _run_forward(monkeypatch, meta, num_spec=3, layer=layer)
