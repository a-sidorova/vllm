# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Solar-Open2 KDA speculative branch under real CUDA graph replay.

A FULL decode graph replays ``SolarOpen2KimiDeltaAttention._forward`` with no
metadata passed in: the kernels read whatever addresses were captured, and only
the persistent buffers that ``GDNAttentionMetadataBuilder.build`` rewrites in
place carry the live step. These tests capture the speculative branch once with
the real builder, replay it over steps whose acceptance counts, batch order and
request padding all differ from the capture batch, and require outputs and
conv/recurrent state to be bit-identical to an eager run of the same steps.

The eager reference uses a builder in PIECEWISE mode, i.e. the KDA op runs
outside the graph on fresh metadata.

Each test runs twice: on the generic (unfused) branch of
``SolarOpen2KimiDeltaAttention._forward``, and (``fused_path``) on its fused
branch, which runs one conv over concat[q, k, v] and
``fused_sigmoid_gating_delta_rule_update`` on raw g1 and b, fed the way the
fused model feeds it: q/k/v/b as column slices of one in_proj output.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph replay needs a GPU"
)

LAYER = "layer"
# Solar-Open2 KDA at TP=4: 64 heads / 4 ranks, head_dim 128, short conv 4.
NUM_HEADS = 16
HEAD_DIM = 128
PROJ = NUM_HEADS * HEAD_DIM
CONV_WIDTH = 4
BLOCK_SIZE = 16
MAX_NUM_SEQS = 64
NUM_BLOCKS = 64
# Requests that take turns in the batch, each owning 1 + K mamba blocks.
NUM_POOL_REQS = 6
BATCH = 4
# Per-rank columns of the fused in_proj output: q, k, v, b, then f_a and g_a.
IN_PROJ_SPLIT = [PROJ, PROJ, PROJ, NUM_HEADS, 2 * HEAD_DIM]
RECURRENT_KERNELS = ("fused_recurrent_kda", "fused_sigmoid_gating_delta_rule_update")
# What each path launches while the all-spec batch is captured.
SPEC_LAUNCHES = {
    False: SimpleNamespace(num_convs=3, recurrent="fused_recurrent_kda"),
    True: SimpleNamespace(
        num_convs=1, recurrent="fused_sigmoid_gating_delta_rule_update"
    ),
}


def _make_builder(num_spec: int, full_cuda_graph: bool) -> GDNAttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
    )
    # The serving recipe passes --no-enable-prefix-caching, which resolves the
    # mamba cache mode to "none": the block table row is the request's 1 + K
    # state blocks, used as-is.
    vllm_config.cache_config.enable_prefix_caching = False
    vllm_config.cache_config.mamba_cache_mode = "none"
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY if full_cuda_graph else CUDAGraphMode.PIECEWISE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=num_spec
    )
    shapes = MambaStateShapeCalculator.kda_state_shape(
        1, NUM_HEADS, HEAD_DIM, conv_kernel_size=CONV_WIDTH, num_spec=num_spec
    )
    dtypes = (torch.bfloat16, torch.float32)
    page_size = sum(
        int(np.prod(shape)) * get_dtype_size(dtype)
        for shape, dtype in zip(shapes, dtypes)
    )
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=shapes,
        dtypes=dtypes,
        # Hybrid models pad the mamba page up to the attention page size.
        page_size_padded=round_up(page_size, 1 << 16),
        mamba_cache_mode="none",
        num_speculative_blocks=num_spec,
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=[LAYER],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )
    assert builder.use_full_cuda_graph == full_cuda_graph
    return builder


def _state_views(spec: MambaSpec, raw: torch.Tensor) -> list[torch.Tensor]:
    """Per-layer conv/recurrent views laid out as attn_utils binds them: both
    states share one padded page per block."""
    views = []
    offset = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes):
        size = get_dtype_size(dtype)
        target_shape = (NUM_BLOCKS, *shape)
        stride = torch.empty(target_shape, device="meta").stride()
        views.append(
            torch.as_strided(
                raw.view(dtype),
                size=target_shape,
                stride=(spec.page_size_bytes // size, *stride[1:]),
                storage_offset=offset // size,
            )
        )
        offset += stride[0] * size
    return views


def _make_layer(kv_cache: list[torch.Tensor], weights: dict, num_spec: int):
    from vllm.model_executor.models.solar_open2 import SolarOpen2KimiDeltaAttention

    def conv1d(name):
        return SimpleNamespace(weight=weights[name], bias=None)

    layer = SimpleNamespace(
        prefix=LAYER,
        kv_cache=tuple(kv_cache),
        head_dim=HEAD_DIM,
        num_spec=num_spec,
        fused_path=False,
        allow_neg_eigval=True,
        A_log=weights["A_log"],
        dt_bias=weights["dt_bias"],
        q_conv1d=conv1d("q_conv1d"),
        k_conv1d=conv1d("k_conv1d"),
        v_conv1d=conv1d("v_conv1d"),
    )
    for name in ("_spec_qkv_conv", "_unfused_qkv_conv"):
        setattr(
            layer, name, MethodType(getattr(SolarOpen2KimiDeltaAttention, name), layer)
        )
    return layer


def _make_fused_layer(kv_cache: list[torch.Tensor], weights: dict, num_spec: int):
    from vllm.model_executor.models.solar_open2 import SolarOpen2KimiDeltaAttention

    conv_weight = torch.cat(
        [weights[name] for name in ("q_conv1d", "k_conv1d", "v_conv1d")]
    )
    layer = SimpleNamespace(
        fused_path=True,
        prefix=LAYER,
        kv_cache=tuple(kv_cache),
        head_dim=HEAD_DIM,
        local_num_heads=NUM_HEADS,
        num_spec=num_spec,
        A_log=weights["A_log"],
        dt_bias=weights["dt_bias"],
        conv1d=SimpleNamespace(weight=conv_weight, bias=None),
        allow_neg_eigval=True,
    )
    for name in ("_spec_qkv_conv", "_fused_qkv_conv"):
        setattr(
            layer, name, MethodType(getattr(SolarOpen2KimiDeltaAttention, name), layer)
        )
    return layer


def _random_fused_inputs(num_tokens: int) -> dict[str, torch.Tensor]:
    return {
        "in_proj": torch.randn(
            num_tokens, sum(IN_PROJ_SPLIT), device="cuda", dtype=torch.bfloat16
        ),
        "g1": torch.randn(
            1, num_tokens, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        ),
    }


def _fused_op_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """What SolarOpen2KimiDeltaAttention.forward hands the op: q/k/v as adjacent
    column slices of the in_proj output (so the conv input is a zero-copy view)
    and beta as the raw b logits, a strided column slice of the same buffer."""
    q, k, v, b, _ = inputs["in_proj"].split(IN_PROJ_SPLIT, dim=-1)
    return {
        "q_proj_states": q,
        "k_proj_states": k,
        "v_proj_states": v,
        "g1": inputs["g1"],
        "beta": b.unsqueeze(0),
    }


def _base_op_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same step as the base layer takes it: separate q/k/v and the gated
    beta, 2 * sigmoid(b) under kda_allow_neg_eigval."""
    q, k, v, b, _ = inputs["in_proj"].split(IN_PROJ_SPLIT, dim=-1)
    return {
        "q_proj_states": q.contiguous(),
        "k_proj_states": k.contiguous(),
        "v_proj_states": v.contiguous(),
        "g1": inputs["g1"].clone(),
        "beta": (2.0 * b.float().sigmoid()).unsqueeze(0),
    }


def _random_inputs(num_tokens: int) -> dict[str, torch.Tensor]:
    def proj():
        return torch.randn(num_tokens, PROJ, device="cuda", dtype=torch.bfloat16)

    return {
        "q_proj_states": proj(),
        "k_proj_states": proj(),
        "v_proj_states": proj(),
        "g1": torch.randn(
            1, num_tokens, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        ),
        # Solar sets kda_allow_neg_eigval, so beta = 2 * sigmoid(b).
        "beta": 2 * torch.randn(1, num_tokens, NUM_HEADS, device="cuda").sigmoid(),
    }


def _empty_output(num_tokens: int) -> torch.Tensor:
    return torch.full(
        (1, num_tokens, NUM_HEADS, HEAD_DIM),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )


class _Harness:
    """One KDA layer driven two ways over the same sequence of steps.

    The graph side owns a FULL-mode builder, V2-runner-style persistent input
    buffers and a CUDA graph captured from a dummy all-spec batch. The eager
    side owns a PIECEWISE-mode builder and its own copy of the KV cache.

    With ``fused_path`` a third, base-layer side runs each step eagerly on the
    eager side's metadata and its own cache copy, as an independent reference:
    replay-vs-eager cannot see a bug the graph and eager sides share.

    ``launches`` records the conv and recurrent-kernel calls made while the
    graph was being captured.
    """

    def __init__(
        self,
        monkeypatch,
        num_spec: int,
        zero_query_start_loc: bool,
        fused_path: bool = False,
    ):
        from vllm.model_executor.models import solar_open2 as mod

        self.num_spec = num_spec
        self.graph_builder = _make_builder(num_spec, full_cuda_graph=True)
        self.eager_builder = _make_builder(num_spec, full_cuda_graph=False)
        spec = self.graph_builder.kv_cache_spec
        assert isinstance(spec, MambaSpec)

        self.graph_raw = torch.empty(
            NUM_BLOCKS * spec.page_size_bytes, dtype=torch.uint8, device="cuda"
        )
        graph_kv = _state_views(spec, self.graph_raw)
        graph_kv[0].copy_(torch.randn(graph_kv[0].shape))
        graph_kv[1].copy_(0.1 * torch.randn(graph_kv[1].shape))
        self.eager_raw = self.graph_raw.clone()
        eager_kv = _state_views(spec, self.eager_raw)
        self.graph_kv, self.eager_kv = graph_kv, eager_kv

        weights = {
            name: 0.5 * torch.randn(PROJ, 1, CONV_WIDTH, device="cuda")
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        }
        weights["A_log"] = 0.5 * torch.randn(1, 1, NUM_HEADS, 1, device="cuda") - 1
        weights["dt_bias"] = 0.1 * torch.randn(PROJ, device="cuda")
        if fused_path:
            make_layer = _make_fused_layer
            self._make_inputs = _random_fused_inputs
            self._op_inputs = _fused_op_inputs
            self.base_raw = self.graph_raw.clone()
            self.base_kv = _state_views(spec, self.base_raw)
            self.base_layer = _make_layer(self.base_kv, weights, num_spec)
        else:
            make_layer = _make_layer
            self._make_inputs = _random_inputs
            self._op_inputs = dict
            self.base_layer = None
        # Both paths are branches of the same _forward, picked by the layer's
        # fused_path; the base side runs the generic branch.
        self._kda_forward = mod.SolarOpen2KimiDeltaAttention._forward
        self._base_forward = mod.SolarOpen2KimiDeltaAttention._forward
        self.graph_layer = make_layer(graph_kv, weights, num_spec)
        self.eager_layer = make_layer(eager_kv, weights, num_spec)

        self.forward_context = SimpleNamespace(attn_metadata=None)
        monkeypatch.setattr(mod, "get_forward_context", lambda: self.forward_context)

        self._recording = False
        self.launches: dict[str, list[SimpleNamespace]] = {}
        for name in (
            "causal_conv1d_update",
            "fused_recurrent_kda",
            "fused_sigmoid_gating_delta_rule_update",
        ):
            self._record(monkeypatch, mod, name)

        # Distinct, shuffled, non-null block ids: every (request, ladder slot)
        # pair addresses its own state.
        ids = 1 + torch.randperm(NUM_BLOCKS - 1)[: NUM_POOL_REQS * (num_spec + 1)]
        self.req_blocks = ids.view(NUM_POOL_REQS, num_spec + 1).to(torch.int32)
        self.num_computed = [100 + 7 * r for r in range(NUM_POOL_REQS)]
        self.num_accepted = [1] * NUM_POOL_REQS

        max_tokens = BATCH * (num_spec + 1)
        self.query_start_loc_buf = torch.zeros(
            MAX_NUM_SEQS + 1, dtype=torch.int32, device="cuda"
        )
        self.seq_lens_buf = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device="cuda")
        self.block_table_buf = torch.zeros(
            MAX_NUM_SEQS, num_spec + 1, dtype=torch.int32, device="cuda"
        )
        self.slot_mapping_buf = torch.zeros(
            max_tokens, dtype=torch.int64, device="cuda"
        )
        self.num_accepted_buf = torch.ones(
            MAX_NUM_SEQS, dtype=torch.int32, device="cuda"
        )
        self.graph_inputs = self._make_inputs(max_tokens)
        self.graph_output = _empty_output(max_tokens)
        self.graph, self.captured = self._capture(zero_query_start_loc)

    def _record(self, monkeypatch, module, name: str):
        fn = getattr(module, name)
        calls = self.launches[name] = []

        def recorded(*args, **kwargs):
            if self._recording:
                calls.append(SimpleNamespace(args=args, kwargs=kwargs))
            return fn(*args, **kwargs)

        monkeypatch.setattr(module, name, recorded)

    def _run(self, layer, meta: GDNAttentionMetadata, inputs, core_attn_out):
        self.forward_context.attn_metadata = {LAYER: meta}
        self._kda_forward(
            layer, **self._op_inputs(inputs), core_attn_out=core_attn_out
        )

    def _common(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        block_rows: torch.Tensor,
        num_reqs_padded: int,
        num_tokens_padded: int,
        persistent: bool,
    ) -> CommonAttentionMetadata:
        num_reqs = len(query_lens)
        num_pad = num_reqs_padded - num_reqs
        query_start_loc_np = np.full(
            num_reqs_padded + 1, sum(query_lens), dtype=np.int32
        )
        query_start_loc_np[0] = 0
        np.cumsum(query_lens, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
        seq_lens_cpu = torch.tensor(seq_lens + [0] * num_pad, dtype=torch.int32)
        block_rows = torch.cat(
            [block_rows, torch.zeros(num_pad, block_rows.size(1), dtype=torch.int32)]
        )
        if persistent:
            query_start_loc = self.query_start_loc_buf[: num_reqs_padded + 1]
            query_start_loc.copy_(query_start_loc_cpu)
            seq_lens_gpu = self.seq_lens_buf[:num_reqs_padded]
            seq_lens_gpu.copy_(seq_lens_cpu)
            block_table = self.block_table_buf[:num_reqs_padded]
            block_table.copy_(block_rows)
            slot_mapping = self.slot_mapping_buf[:num_tokens_padded]
        else:
            query_start_loc = query_start_loc_cpu.cuda()
            seq_lens_gpu = seq_lens_cpu.cuda()
            block_table = block_rows.cuda()
            slot_mapping = torch.zeros(
                num_tokens_padded, dtype=torch.int64, device="cuda"
            )
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens_gpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max(query_lens),
            max_seq_len=int(seq_lens_cpu.max()),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
        )

    def _capture(self, zero_query_start_loc: bool):
        """Mirror CudaGraphManager: a warmup pass and a captured pass, each on
        metadata from build_for_cudagraph_capture over the dummy batch (query
        length 1 + K, seq_len == query_len, all-null block table)."""
        query_len = self.num_spec + 1
        num_tokens = BATCH * query_len
        graph = torch.cuda.CUDAGraph()
        for warmup in (True, False):
            common = self._common(
                [query_len] * BATCH,
                [query_len] * BATCH,
                torch.zeros(BATCH, self.num_spec + 1, dtype=torch.int32),
                BATCH,
                num_tokens,
                persistent=True,
            )
            if zero_query_start_loc:
                # An attention builder that zeroes the shared query start
                # locations at capture hides the spec decodes from GDN.
                common.query_start_loc.zero_()
                common.query_start_loc_cpu.zero_()
            meta = self.graph_builder.build_for_cudagraph_capture(common)
            if warmup:
                self._run(self.graph_layer, meta, self.graph_inputs, self.graph_output)
                torch.cuda.synchronize()
            else:
                self._recording = True
                with torch.cuda.graph(graph):
                    self._run(
                        self.graph_layer, meta, self.graph_inputs, self.graph_output
                    )
                self._recording = False
        torch.cuda.synchronize()
        return graph, meta

    def step(self, rows: list[tuple[int, int]], accepted: list[int], replay: bool):
        """Run one step on both sides.

        ``rows`` is the batch in order, as ``(request, num_drafts)``; a request
        with no drafts runs a plain one-token decode. ``accepted`` is how many
        tokens each row accepts, which the next step of that request sees as
        ``num_accepted_tokens``. With ``replay`` the graph side replays the
        captured graph (padded to the capture batch); otherwise it runs
        eagerly on the FULL-mode builder, as a non-graphable batch would.
        """
        reqs = [r for r, _ in rows]
        query_lens = [1 + d for _, d in rows]
        seq_lens = [self.num_computed[r] + q for r, q in zip(reqs, query_lens)]
        block_rows = self.req_blocks[reqs]
        num_reqs, num_tokens = len(rows), sum(query_lens)
        draft_counts = [d if d > 0 else -1 for _, d in rows]
        inputs = self._make_inputs(num_tokens)

        if replay:
            assert all(d == self.num_spec for _, d in rows)
            num_reqs_padded = BATCH
            num_tokens_padded = BATCH * (self.num_spec + 1)
        else:
            num_reqs_padded, num_tokens_padded = num_reqs, num_tokens
        num_pad = num_reqs_padded - num_reqs
        self.num_accepted_buf[:num_reqs_padded].copy_(
            torch.tensor([self.num_accepted[r] for r in reqs] + [1] * num_pad)
        )
        graph_meta = self.graph_builder.build(
            0,
            self._common(
                query_lens,
                seq_lens,
                block_rows,
                num_reqs_padded,
                num_tokens_padded,
                persistent=True,
            ),
            num_accepted_tokens=self.num_accepted_buf[:num_reqs_padded],
            num_decode_draft_tokens_cpu=torch.tensor(
                draft_counts + [-1] * num_pad, dtype=torch.int32
            ),
        )
        if replay:
            for name, x in inputs.items():
                dst = self.graph_inputs[name]
                (dst[:num_tokens] if x.dim() == 2 else dst[:, :num_tokens]).copy_(x)
            self.graph_output.fill_(float("nan"))
            self.graph.replay()
            # Rows past num_tokens are request padding; nothing constrains them.
            graph_out = self.graph_output[0, :num_tokens].clone()
        else:
            out = _empty_output(num_tokens)
            self._run(
                self.graph_layer,
                graph_meta,
                {k: v.clone() for k, v in inputs.items()},
                out,
            )
            graph_out = out[0]

        eager_meta = self.eager_builder.build(
            0,
            self._common(query_lens, seq_lens, block_rows, num_reqs, num_tokens, False),
            num_accepted_tokens=torch.tensor(
                [self.num_accepted[r] for r in reqs],
                dtype=torch.int32,
                device="cuda",
            ),
            num_decode_draft_tokens_cpu=torch.tensor(draft_counts, dtype=torch.int32),
        )
        eager_out = _empty_output(num_tokens)
        self._run(
            self.eager_layer,
            eager_meta,
            {k: v.clone() for k, v in inputs.items()},
            eager_out,
        )
        if self.base_layer is not None:
            self.base_out = _empty_output(num_tokens)
            self.forward_context.attn_metadata = {LAYER: eager_meta}
            self._base_forward(
                self.base_layer,
                **_base_op_inputs(inputs),
                core_attn_out=self.base_out,
            )
        torch.cuda.synchronize()

        for r, (_, d), a in zip(reqs, rows, accepted):
            assert 1 <= a <= d + 1
            self.num_computed[r] += a
            self.num_accepted[r] = a
        return graph_out, eager_out[0], graph_meta

    def ladder_blocks(self, reqs: list[int]) -> torch.Tensor:
        return self.req_blocks[reqs].flatten().long().cuda()

    def assert_same_state(self):
        for name, g, e in zip(("conv", "recurrent"), self.graph_kv, self.eager_kv):
            torch.testing.assert_close(g, e, rtol=0, atol=0, msg=f"{name} state")
        assert torch.equal(self.graph_raw, self.eager_raw)


def _spec_schedule(k: int) -> list[tuple[list[int], list[int]]]:
    """(requests in batch order, tokens each accepts). Covers full and single
    acceptance, a reordered batch, a request swapped in, and a step with fewer
    requests than the captured batch (request padding)."""
    full, mid = k + 1, k // 2 + 1
    return [
        ([0, 1, 2, 3], [full, 1, mid, 2]),
        ([3, 1, 4, 0], [1, full, full, mid]),
        ([5, 2, 3], [mid, 1, full]),
        ([2, 0, 5, 4], [2, full, 1, mid]),
        ([1, 4, 3, 2], [full, 2, mid, 1]),
    ]


def _assert_step_matches(h: _Harness, graph_out, eager_out, reqs, rec_before):
    assert not torch.isnan(eager_out).any()
    assert not torch.isnan(graph_out).any(), (
        "replay left the sentinel in core_attn_out: a kernel exited without "
        "writing its output"
    )
    torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)
    h.assert_same_state()
    if h.base_layer is not None:
        # Conv is per-channel, so one conv over 3P channels is bit-exact with
        # three over P; the recurrent kernels differ, so output (bf16) and
        # recurrent state (fp32) match at the default dtype tolerances.
        torch.testing.assert_close(
            h.graph_kv[0], h.base_kv[0], rtol=0, atol=0, msg="conv state vs base"
        )
        torch.testing.assert_close(
            h.graph_kv[1], h.base_kv[1], msg="recurrent state vs base"
        )
        torch.testing.assert_close(graph_out, h.base_out[0])
    blocks = h.ladder_blocks(reqs)
    assert not torch.equal(h.graph_kv[1][blocks], rec_before[blocks]), (
        "the step did not advance the recurrent state of the batch"
    )


by_path = pytest.mark.parametrize("fused_path", [False, True], ids=["base", "fused"])


def _assert_captured_spec_branch(h: _Harness, fused_path: bool):
    """The graph holds the path's speculative branch: its recurrent kernel on
    the 2-D ladder and the accepted counts, and not the other path's."""
    captured = h.captured
    expected = SPEC_LAUNCHES[fused_path]
    convs = h.launches["causal_conv1d_update"]
    assert len(convs) == expected.num_convs
    for name in RECURRENT_KERNELS:
        if name != expected.recurrent:
            assert not h.launches[name], f"{name} was captured"
    assert len(h.launches[expected.recurrent]) == 1, expected.recurrent
    rec = h.launches[expected.recurrent][0].kwargs
    assert rec["ssm_state_indices"] is captured.spec_state_indices_tensor
    assert rec["num_accepted_tokens"] is captured.num_accepted_tokens
    for conv in convs:
        assert conv.kwargs["num_accepted_tokens"] is captured.num_accepted_tokens
    if fused_path:
        in_proj = h.graph_inputs["in_proj"]
        x = convs[0].args[0]
        assert x.data_ptr() == in_proj.data_ptr()
        assert x.stride() == (in_proj.stride(0), 1), (
            "the captured conv must read the in_proj output in place"
        )
        assert rec["b"].data_ptr() == in_proj[:, 3 * PROJ].data_ptr(), "raw b in place"
        assert rec["out"].data_ptr() == h.graph_output.data_ptr(), (
            "the captured kernel must write the persistent core_attn_out"
        )


@by_path
@pytest.mark.parametrize("num_spec", [3, 7])
def test_full_graph_replay_of_spec_branch_matches_eager(
    monkeypatch, num_spec, fused_path
):
    torch.manual_seed(0)
    h = _Harness(
        monkeypatch, num_spec, zero_query_start_loc=False, fused_path=fused_path
    )

    captured = h.captured
    assert captured.num_spec_decodes == BATCH
    assert captured.num_decodes == 0 and captured.num_prefills == 0
    assert captured.spec_sequence_masks is not None
    _assert_captured_spec_branch(h, fused_path)
    h.assert_same_state()

    for reqs, accepted in _spec_schedule(num_spec):
        rec_before = h.graph_kv[1].clone()
        graph_out, eager_out, meta = h.step(
            [(r, num_spec) for r in reqs], accepted, replay=True
        )
        assert meta.num_spec_decodes == len(reqs)
        for name in ("spec_state_indices_tensor", "spec_query_start_loc"):
            assert (
                getattr(meta, name).data_ptr() == getattr(captured, name).data_ptr()
            ), name
        assert meta.num_accepted_tokens is not None
        assert captured.num_accepted_tokens is not None
        assert (
            meta.num_accepted_tokens.data_ptr()
            == captured.num_accepted_tokens.data_ptr()
        )
        _assert_step_matches(h, graph_out, eager_out, reqs, rec_before)


@by_path
def test_capture_with_zeroed_query_start_loc_diverges(monkeypatch, fused_path):
    """Negative control: a capture whose query start locations were zeroed by
    another builder must be caught by the replay-vs-eager comparison above."""
    num_spec = 7
    torch.manual_seed(0)
    h = _Harness(
        monkeypatch, num_spec, zero_query_start_loc=True, fused_path=fused_path
    )

    assert h.captured.num_spec_decodes == 0
    assert h.captured.num_decodes == BATCH
    assert h.captured.spec_sequence_masks is None

    reqs, accepted = _spec_schedule(num_spec)[0]
    graph_before = h.graph_raw.clone()
    graph_out, eager_out, meta = h.step(
        [(r, num_spec) for r in reqs], accepted, replay=True
    )
    assert meta.num_spec_decodes == len(reqs)
    assert not torch.isnan(eager_out).any()

    assert torch.equal(h.graph_raw, graph_before), (
        "the stale non-spec graph should not reach any real state block"
    )
    assert not torch.equal(h.graph_raw, h.eager_raw)
    assert not torch.equal(graph_out, eager_out)


@by_path
@pytest.mark.parametrize("transition", ["decode", "mixed"])
def test_fold_after_replayed_spec_steps_matches_eager(
    monkeypatch, transition, fused_path
):
    """Replayed spec steps leave state on the draft ladder and the conv window
    rolled; the next non-speculative read folds it back using
    non_spec_committed_state_indices / non_spec_num_accepted_tokens. That fold
    runs eagerly (only uniform 1 + K batches are graphed) but on state written
    by the graph, and the requests then go back to replayed spec steps.

    ``decode``: every request runs a plain one-token decode.
    ``mixed``: two requests decode, two keep drafting, so the decodes are
    reclassified as prefills and take the chunk kernel.
    """
    num_spec = 7
    full, mid = num_spec + 1, num_spec // 2 + 1
    torch.manual_seed(0)
    h = _Harness(
        monkeypatch, num_spec, zero_query_start_loc=False, fused_path=fused_path
    )
    reqs = [0, 1, 2, 3]

    for accepted in ([full, 1, mid, 2], [2, full, mid, 1]):
        rec_before = h.graph_kv[1].clone()
        graph_out, eager_out, _ = h.step(
            [(r, num_spec) for r in reqs], accepted, replay=True
        )
        _assert_step_matches(h, graph_out, eager_out, reqs, rec_before)
    assert any(h.num_accepted[r] > 1 for r in reqs)

    if transition == "decode":
        rows = [(r, 0) for r in reqs]
        accepted = [1] * len(rows)
    else:
        rows = [(0, 0), (2, num_spec), (1, 0), (3, num_spec)]
        accepted = [1, mid, 1, full]
    rec_before = h.graph_kv[1].clone()
    graph_out, eager_out, meta = h.step(rows, accepted, replay=False)
    if transition == "decode":
        assert meta.num_decodes == len(rows) and meta.num_spec_decodes == 0
    else:
        assert meta.num_prefills == 2 and meta.num_spec_decodes == 2
    assert meta.non_spec_committed_state_indices is not None
    assert meta.non_spec_num_accepted_tokens is not None
    _assert_step_matches(h, graph_out, eager_out, reqs, rec_before)

    rec_before = h.graph_kv[1].clone()
    graph_out, eager_out, _ = h.step(
        [(r, num_spec) for r in reqs], [mid, 1, full, 2], replay=True
    )
    _assert_step_matches(h, graph_out, eager_out, reqs, rec_before)
