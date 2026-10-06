# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

import importlib
from dataclasses import dataclass, fields
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # No speculative config, decode alongside prefill
    "regular_decode_with_prefill": GDNBuildTestCase(
        seq_lens=[40, 100],
        query_lens=[1, 50],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=1,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    num_speculative_blocks: int = 0,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=num_speculative_blocks,
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    block_table: torch.Tensor | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    if block_table is not None:
        common = common.replace(block_table_tensor=block_table)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize("full_cuda_graph", [False, True])
@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_update_block_table_matches_build(
    test_case: GDNBuildTestCase, full_cuda_graph: bool, mamba_cache_mode: str
):
    """update_block_table() on another group's metadata matches build()."""
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    src, dst, ref = (
        _create_gdn_builder(test_case.num_speculative_tokens, full_cuda_graph)
        for _ in range(3)
    )
    for builder in (src, dst, ref):
        builder.vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    if mamba_cache_mode == "align":
        # MRV2 precomputes these for every Mamba group each step.
        dst.mamba_aligned_state_indices = ref.mamba_aligned_state_indices = (
            mamba_get_block_table_tensor(
                common.block_table_tensor, common.seq_lens, ref.kv_cache_spec, "align"
            )
        )
    draft_tokens = test_case.num_decode_draft_tokens
    expected = _build(ref, batch, draft_tokens, common.block_table_tensor)
    source = _build(src, batch, draft_tokens)
    index_fields = (
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "prefill_state_indices",
        "non_spec_committed_state_indices",
    )
    source_indices = [getattr(source, f) for f in index_fields]
    source_indices = [t if t is None else t.clone() for t in source_indices]
    meta = dst.update_block_table(
        source, common.block_table_tensor, common.slot_mapping
    )

    for field, source_index in zip(index_fields, source_indices):
        actual = getattr(meta, field)
        torch.testing.assert_close(actual, getattr(expected, field))
        # The source group's indices are untouched.
        torch.testing.assert_close(getattr(source, field), source_index)
        # FULL graph state indices land in this group's own buffers.
        if full_cuda_graph and meta.num_prefills == 0 and actual is not None:
            assert actual.data_ptr() == getattr(dst, field).data_ptr()


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def _build_non_spec(
    batch: BatchSpec,
    is_prefilling: list[bool] | None,
    full_cuda_graph: bool = False,
):
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(
        is_prefilling=None
        if is_prefilling is None
        else torch.tensor(is_prefilling, dtype=torch.bool)
    )
    builder = _create_gdn_builder(full_cuda_graph=full_cuda_graph)
    return builder, common_attn_metadata, builder.build(0, common_attn_metadata)


@pytest.mark.parametrize(
    ("seq_len", "query_len", "is_prefilling", "num_prefills"),
    [
        pytest.param(1, 1, True, 1, id="first-chunk"),
        pytest.param(65, 1, True, 0, id="resumed-chunk"),
        pytest.param(0, 0, True, 0, id="padding"),
        pytest.param(1, 1, None, 0, id="missing-prefill-flag"),
    ],
)
def test_one_token_chunk_classification(
    seq_len: int,
    query_len: int,
    is_prefilling: bool | None,
    num_prefills: int,
):
    """Only a real first chunk with a prefill flag needs state initialization."""
    _, _, meta = _build_non_spec(
        BatchSpec(seq_lens=[100, seq_len], query_lens=[1, query_len]),
        is_prefilling=None if is_prefilling is None else [False, is_prefilling],
    )

    assert meta.num_prefills == num_prefills
    assert meta.num_decodes == 2 - num_prefills
    assert meta.num_prefill_tokens == num_prefills
    assert meta.num_decode_tokens == 1 + query_len - num_prefills
    if num_prefills:
        assert meta.has_initial_state is not None
        assert meta.has_initial_state.tolist() == [True, False]
    else:
        assert meta.has_initial_state is None


def test_one_token_first_chunk_excludes_padding():
    """Neither padding requests nor padding tokens count as prefill work."""
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 1, 0, 0], query_lens=[1, 1, 0, 0]),
        BLOCK_SIZE,
        DEVICE,
    ).replace(
        is_prefilling=torch.tensor([False, True, False, False], dtype=torch.bool),
        num_actual_tokens=4,
    )
    meta = _create_gdn_builder().build(0, common)

    assert meta.num_decodes == 1
    assert meta.num_prefills == 1
    assert meta.num_decode_tokens == 1
    assert meta.num_prefill_tokens == 1


def test_cudagraph_capture_batch_stays_decode_only():
    """Capture rows have no history, but must still select decode kernels."""
    batch = BatchSpec(seq_lens=[1] * 4, query_lens=[1] * 4)
    builder, common_attn_metadata, _ = _build_non_spec(
        batch, [False] * 4, full_cuda_graph=True
    )
    meta = builder.build_for_cudagraph_capture(common_attn_metadata)

    assert meta.num_prefills == 0
    assert meta.num_decodes == 4
    assert meta.has_initial_state is None
    staged = meta.non_spec_state_indices_tensor
    assert staged is not None
    assert staged.data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    torch.testing.assert_close(staged, common_attn_metadata.block_table_tensor[:, 0])


def test_cudagraph_capture_metadata_avoids_device_to_host_copy(
    monkeypatch: pytest.MonkeyPatch,
):
    """`build_for_cudagraph_capture` is not capture-only: a DP rank with nothing
    scheduled re-stages its FULL-graph metadata through it on every dummy step,
    so it must not synchronize the device. The host-side draft counts have to
    come from `query_start_loc_cpu` and match the device-derived values."""
    num_speculative_tokens = 2
    batch = BatchSpec(seq_lens=[50, 30], query_lens=[3, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.zeros(batch.batch_size, dtype=torch.bool))
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens, full_cuda_graph=True
    )

    device_to_host_copies: list[torch.Size] = []
    original_cpu = torch.Tensor.cpu

    def counting_cpu(self: torch.Tensor, *args, **kwargs):
        device_to_host_copies.append(self.shape)
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    actual = builder.build_for_cudagraph_capture(common_attn_metadata)
    monkeypatch.undo()

    assert not device_to_host_copies, device_to_host_copies

    num_accepted_tokens = torch.diff(common_attn_metadata.query_start_loc)
    reference = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens,
        (num_accepted_tokens - 1).cpu(),
    )
    assert actual.num_spec_decodes == batch.batch_size
    assert actual.num_decodes == 0
    assert actual.num_prefills == 0
    for field in fields(type(actual)):
        actual_value = getattr(actual, field.name)
        reference_value = getattr(reference, field.name)
        if isinstance(actual_value, torch.Tensor):
            torch.testing.assert_close(actual_value, reference_value)
        else:
            assert actual_value == reference_value, field.name


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific test")
@pytest.mark.parametrize(
    "builder_path, preserves_device_query_start_loc",
    [
        pytest.param(
            "rocm_aiter_unified_attn.RocmAiterUnifiedAttentionMetadataBuilder",
            True,
            id="rocm_aiter_unified_attn",
        ),
        pytest.param(
            "rocm_attn.RocmAttentionMetadataBuilder",
            False,
            id="rocm_attn",
        ),
    ],
)
def test_gdn_capture_after_rocm_builder_sees_spec_decodes(
    builder_path: str, preserves_device_query_start_loc: bool
):
    """FULL graph capture hands one CommonAttentionMetadata to every KV-cache
    group in order. Hybrid models with attention groups ahead of GDN/KDA groups
    must still capture the spec-decode branch of the linear-attention layers."""
    module_name, class_name = builder_path.rsplit(".", 1)
    rocm_builder_cls = getattr(
        importlib.import_module(f"vllm.v1.attention.backends.{module_name}"),
        class_name,
    )
    rocm_builder = object.__new__(rocm_builder_cls)
    rocm_metadata = MagicMock()
    rocm_metadata.seq_lens = torch.tensor([1048576] * 4, dtype=torch.int32)
    rocm_builder.build = MagicMock(return_value=rocm_metadata)

    num_speculative_tokens = 7
    batch = BatchSpec(seq_lens=[64, 72, 80, 96], query_lens=[8] * 4)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    # On a CPU device query_start_loc.cpu() aliases the device tensor; the
    # runner always passes two distinct tensors.
    common = common.replace(query_start_loc_cpu=common.query_start_loc_cpu.clone())
    gdn_builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens, full_cuda_graph=True
    )

    rocm_builder.build_for_cudagraph_capture(common)
    meta = gdn_builder.build_for_cudagraph_capture(common)

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_decodes == 0
    assert meta.num_prefills == 0
    assert meta.spec_sequence_masks is not None
    assert meta.spec_query_start_loc is not None
    assert (
        meta.spec_query_start_loc.data_ptr()
        == gdn_builder.spec_query_start_loc.data_ptr()
    )
    if preserves_device_query_start_loc:
        torch.testing.assert_close(
            meta.spec_query_start_loc[: batch.batch_size + 1],
            torch.tensor([0, 8, 16, 24, 32], dtype=torch.int32),
        )


def test_full_cudagraph_replay_step_reuses_capture_buffers():
    """A FULL graph reads the GDN spec metadata from the addresses seen at
    capture; every later build() must write the new step into those same
    buffers."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
        num_speculative_blocks=num_speculative_tokens,
    )

    def make_common(batch: BatchSpec, first_block: int):
        common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
        # Wide enough for the last `1 + num_speculative_blocks` blocks that
        # mamba_cache_mode="align" gathers per request.
        num_cols = 16
        block_table = first_block + torch.arange(
            batch.batch_size * num_cols, dtype=torch.int32, device=DEVICE
        ).view(batch.batch_size, num_cols)
        return common.replace(block_table_tensor=block_table)

    capture_batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    capture_common = make_common(capture_batch, first_block=1)
    captured = builder.build_for_cudagraph_capture(capture_common)
    assert captured.num_spec_decodes == capture_batch.batch_size
    assert captured.spec_state_indices_tensor is not None
    assert captured.spec_query_start_loc is not None
    assert captured.num_accepted_tokens is not None
    assert captured.spec_sequence_masks is not None

    step_batch = BatchSpec(seq_lens=[120, 70], query_lens=[4, 2])
    step_common = make_common(step_batch, first_block=500)
    num_accepted_tokens = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)
    step = builder.build(
        0,
        step_common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=torch.tensor([3, 1], dtype=torch.int32),
    )

    assert step.num_spec_decodes == step_batch.batch_size
    assert step.spec_state_indices_tensor is not None
    assert step.spec_query_start_loc is not None
    assert step.num_accepted_tokens is not None
    assert step.spec_sequence_masks is not None
    for name in (
        "spec_state_indices_tensor",
        "spec_query_start_loc",
        "num_accepted_tokens",
        "spec_sequence_masks",
    ):
        buffer = getattr(builder, name)
        assert getattr(captured, name).data_ptr() == buffer.data_ptr(), name
        assert getattr(step, name).data_ptr() == buffer.data_ptr(), name

    expected_state_indices = mamba_get_block_table_tensor(
        step_common.block_table_tensor,
        step_common.seq_lens,
        builder.kv_cache_spec,
        builder.vllm_config.cache_config.mamba_cache_mode,
    )[:, : num_speculative_tokens + 1]
    assert expected_state_indices.shape == (
        step_batch.batch_size,
        num_speculative_tokens + 1,
    )
    torch.testing.assert_close(step.spec_state_indices_tensor, expected_state_indices)
    torch.testing.assert_close(
        step.spec_query_start_loc,
        torch.tensor([0, 4, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(step.num_accepted_tokens, num_accepted_tokens)
    assert bool(step.spec_sequence_masks.all())


@pytest.mark.parametrize("mixed", [False, True], ids=["all-non-spec", "mixed"])
def test_non_spec_rows_carry_committed_state_indices(mixed: bool):
    """Rows not drafting this step point at the ladder block holding their
    committed state (column num_accepted - 1), in persistent buffers."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        num_speculative_blocks=num_speculative_tokens,
    )
    if mixed:
        batch = BatchSpec(seq_lens=[40, 30, 50], query_lens=[1, 4, 1])
        draft_tokens = [-1, 3, -1]
        non_spec_rows = [0, 2]
    else:
        batch = BatchSpec(seq_lens=[40, 30, 50], query_lens=[1, 1, 1])
        draft_tokens = [-1, -1, -1]
        non_spec_rows = [0, 1, 2]
    block_table = 1 + torch.arange(
        batch.batch_size * 8, dtype=torch.int32, device=DEVICE
    ).view(batch.batch_size, 8)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        block_table_tensor=block_table
    )
    num_accepted_tokens = torch.tensor([3, 2, 1], dtype=torch.int32, device=DEVICE)

    meta = builder.build(
        0,
        common,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=torch.tensor(draft_tokens, dtype=torch.int32),
    )

    assert meta.non_spec_committed_state_indices is not None
    assert meta.non_spec_num_accepted_tokens is not None
    rows = torch.tensor(non_spec_rows)
    accepted = num_accepted_tokens[rows]
    ladder = mamba_get_block_table_tensor(
        block_table,
        common.seq_lens,
        builder.kv_cache_spec,
        builder.vllm_config.cache_config.mamba_cache_mode,
    )
    torch.testing.assert_close(meta.non_spec_num_accepted_tokens, accepted)
    torch.testing.assert_close(
        meta.non_spec_committed_state_indices,
        ladder[rows, accepted.long() - 1],
    )
    assert (
        meta.non_spec_committed_state_indices.data_ptr()
        == builder.non_spec_committed_state_indices.data_ptr()
    )
    assert (
        meta.non_spec_num_accepted_tokens.data_ptr()
        == builder.non_spec_num_accepted_tokens.data_ptr()
    )
