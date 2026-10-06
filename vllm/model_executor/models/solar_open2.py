# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 Upstage AI.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only SolarOpen2 model compatible with HuggingFace weights.

SolarOpen2 is a hybrid model: decoder layers alternate between grouped-query
attention (GQA, optionally gated) and Kimi Delta Attention (KDA / gated delta
net linear attention), on top of a grouped-sigmoid MoE. The set of GQA layers
is given by ``gqa_layers`` (or a periodic ``gqa_interval``); all other layers
are KDA linear-attention layers.
"""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from einops import rearrange
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
    resolve_layer_fused_shared_expert,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.kda import (
    FusedRMSNormGated,
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def is_gqa_layer(config, layer_idx: int) -> bool:
    """Whether a decoder layer uses GQA attention (vs KDA linear attention).

    ``gqa_layers`` takes priority over the periodic ``gqa_interval``.
    """
    gqa_layers = getattr(config, "gqa_layers", None)
    if gqa_layers is not None:
        return layer_idx in gqa_layers
    return (layer_idx + 1) % config.gqa_interval == 0


def _use_amd_fused_kda(vllm_config: VllmConfig) -> bool:
    # The fused projection/conv path does not implement speculative KDA state
    # forking and accepted-state recovery. Keep the generic KDA path for
    # speculative serving; target-only ROCm serving retains the fused path.
    return rocm_aiter_ops.is_enabled() and vllm_config.speculative_config is None


def _fused_qkv_view(q, k, v):
    """``[n, 3*w]`` concat of q/k/v, zero-copy when they are adjacent column-slices
    of one buffer (the common case: the fused in_proj output). The custom-op /
    torch.compile boundary does not guarantee that adjacency, so fall back to a
    real ``torch.cat`` when it does not hold -- correctness never depends on it."""
    w = q.shape[-1]
    if (
        q.stride(-1) == 1
        and q.untyped_storage().data_ptr()
        == k.untyped_storage().data_ptr()
        == v.untyped_storage().data_ptr()
        and k.storage_offset() == q.storage_offset() + w
        and v.storage_offset() == q.storage_offset() + 2 * w
    ):
        return q.as_strided((q.shape[0], 3 * w), q.stride(), q.storage_offset())
    return torch.cat([q, k, v], dim=-1)


def _make_fused_qkv_conv1d_weight_loader(dims, tp_size, tp_rank):
    """Load the separate q/k/v conv1d checkpoint tensors into ONE fused conv1d
    param at load time (Olmo pattern). ``dims`` = per-section un-sharded output
    sizes [q, k, v]; ``loaded_shard_id`` in {0,1,2} picks the section."""
    sharded_dims = [d // tp_size for d in dims]

    def weight_loader(param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in (0, 1, 2)
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = dims[loaded_shard_id] // tp_size
        tp_start = tp_rank * shard_size
        sharded = loaded_weight[tp_start : tp_start + shard_size]
        offset = sum(sharded_dims[:loaded_shard_id])
        param.data[offset : offset + shard_size].copy_(sharded)

    return weight_loader


def _load_in_proj_dup_shard(
    param, loaded_weight, shard_id, output_sizes, head_dim, tp_size
):
    """Load a DUPLICATED shard (f_a/g_a) of the KDA fused in_proj.

    These logically-replicated projections have their output rows duplicated
    tp_size times in the merged param. The checkpoint holds ONE copy
    ([head_dim, hidden]); every rank must load that FULL copy into its own
    per-rank slice (no source narrowing). The destination offset is the sum of
    the prior shards' per-rank sizes.
    """
    per_rank = [o // tp_size for o in output_sizes]
    offset = sum(per_rank[:shard_id])
    dst = param.data.narrow(0, offset, head_dim)
    dst.copy_(loaded_weight)


class SolarOpen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class SolarOpen2MoE(nn.Module):
    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        # ROCm/AITER shared-expert fusion (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS):
        # the shared expert runs as appended routed-expert slots in the same
        # grouped GEMM instead of as a separate module. Declined per layer when
        # its quant spec differs from the routed experts' (e.g. Quark
        # checkpoints that keep shared experts bf16 next to MXFP4 experts).
        self.is_fused_shared_expert_enabled = False
        if config.n_shared_experts is not None:
            self.is_fused_shared_expert_enabled = resolve_layer_fused_shared_expert(
                quant_config, prefix
            )

        if config.n_shared_experts is None or self.is_fused_shared_expert_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = SolarOpen2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            n_shared_experts=config.n_shared_experts
            if self.is_fused_shared_expert_enabled
            else None,
            fuse_shared_experts=self.is_fused_shared_expert_enabled,
            router_logits_dtype=self.gate.out_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # The runner owns the gate and computes router logits itself.
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        return final_hidden_states.view(num_tokens, hidden_dim)


class SolarOpen2Attention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 131072,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # ROCm/AITER fused zero-copy attention path. Off by default so other
        # vendors keep the stock contiguous-qkv behaviour.
        self.fused_path = rocm_aiter_ops.is_enabled()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.use_rope = getattr(config, "use_rope", False)
        if self.use_rope:
            config.rope_parameters.setdefault("partial_rotary_factor", 0.5)
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
            )
        else:
            self.rotary_emb = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.use_gqa_gate = config.use_gqa_gate
        if self.use_gqa_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=getattr(config, "use_gqa_gate_bias", False),
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if not self.fused_path:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # Fused path passes the strided q/k/v split views straight to attention:
        # both consumers accept strided input (reshape_and_cache_flash reads
        # per-token key/value strides; aiter's Triton unified_attention passes
        # q.stride(0/1) into the kernel), so the per-GQA-layer copy is avoided.
        if self.use_qk_norm:
            q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(
                q.shape
            )
            k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(
                k.shape
            )

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.use_gqa_gate:
            gate, _ = self.g_proj(hidden_states)
            attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)[0]


def kda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        g1=g1,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="kda_attention",
    op_func=kda_attention,
    mutates_args=["core_attn_out"],
    fake_impl=kda_attention_fake,
)


class SolarOpen2KimiDeltaAttention(GatedDeltaNetAttention):
    """KDA linear attention for SolarOpen2.

    The checkpoint stores separate q/k/v/b/f_a projections and q/k/v short
    convolutions, and ``kda_allow_neg_eigval`` needs beta = 2 * sigmoid(b),
    which the shared Kimi GDN layer's raw-beta kernels cannot express. So this
    layer keeps its own projections and feeds a precomputed beta to the FLA
    KDA kernels.
    """

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        self.allow_neg_eigval = getattr(config, "kda_allow_neg_eigval", False)
        # The ROCm AITER gated-RMSNorm fusion pass (cudagraph path) queries
        # num_v_heads / head_v_dim on GDN layers to size the gated-norm pattern.
        # For KDA the gated o_norm operates per value head == (num_heads, head_dim).
        self.num_v_heads = self.num_heads
        self.head_v_dim = self.head_dim

        projection_size = self.head_dim * self.num_heads
        self.conv_size = kda_config["short_conv_kernel_size"]

        # ROCm/AITER fused input-projection + conv path. Off by default so other
        # vendors keep separate projections/convs and the two-kernel decode.
        self.fused_path = _use_amd_fused_kda(vllm_config)
        if self.fused_path:
            self._init_fused_projections(projection_size, prefix)
        else:
            self._init_unfused_projections(projection_size, prefix)

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _init_fused_projections(self, projection_size: int, prefix: str) -> None:
        # Fold the six projections that read hidden_states (q/k/v/b/f_a/g_a) into
        # one MergedColumnParallelLinear. q/k/v/b are column-parallel; f_a/g_a are
        # logically REPLICATED (f_b/g_b read the full head_dim on every rank), so
        # their output rows are duplicated tp times in the merge (output_size
        # head_dim*tp -> sharded /tp -> each rank recovers the full head_dim). Same
        # compute/memory as the replicated modules, but one GEMM launch.
        tp = self.tp_size
        self.in_proj_output_sizes = [
            projection_size,
            projection_size,
            projection_size,
            self.num_heads,
            self.head_dim * tp,
            self.head_dim * tp,
        ]
        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            self.in_proj_output_sizes,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj",
        )
        # per-rank split sizes (output_sizes // tp) in the forward output.
        self.in_split = [size // tp for size in self.in_proj_output_sizes]

        # One conv1d over concat[q,k,v], materialized at load time from the
        # separate checkpoint convs via a fused weight loader (Olmo pattern).
        # conv_state is stored whole, so decode/prefill run a single conv.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size * 3,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_fused_qkv_conv1d_weight_loader(
                    [projection_size, projection_size, projection_size],
                    self.tp_size,
                    get_tensor_model_parallel_rank(),
                )
            },
        )

    def _init_unfused_projections(self, projection_size: int, prefix: str) -> None:
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.b_proj",
        )
        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_a_proj",
        )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        if self.fused_path:
            # Single fused GEMM for all six input projections; split per-rank
            # outputs. f_a/g_a slices are the full head_dim (duplicated tp times).
            fused = self.in_proj(hidden_states)[0]
            q, k, v, b, f_a, g_a = fused.split(self.in_split, dim=-1)
            # Pass RAW b logits (pre-sigmoid) through the op's beta slot; _forward
            # derives the gated beta = sigmoid(b)*[2] where needed (prefill gates
            # there; the fused decode kernel sigmoids inline). Left as a strided
            # column-slice view of the fused in_proj output: prefill's .float()
            # rematerializes it anyway and decode reads it in place via
            # stride_b_tok, so a copy here is one dead kernel launch per layer.
            beta = b.unsqueeze(0)
        else:
            q = self.q_proj(hidden_states)[0]
            k = self.k_proj(hidden_states)[0]
            v = self.v_proj(hidden_states)[0]
            beta = self.b_proj(hidden_states)[0].float().sigmoid()
            if self.allow_neg_eigval:
                beta = beta * 2.0
            beta = beta.unsqueeze(0)
            f_a = self.f_a_proj(hidden_states)[0]
            g_a = self.g_a_proj(hidden_states)[0]

        # Pass the RAW (ungated) g1 to torch.ops.vllm.kda_attention. The op
        # applies the KDA decay gate internally (chunk_kda_with_fused_gate in
        # prefill, fused_kda_gate / the fused decode kernel in decode).
        # Pre-gating here double-applies the decay and degenerates generation.
        g1 = self.f_b_proj(f_a)[0]
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        g_proj_states = self.g_b_proj(g_a)[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        # Rows the op leaves unwritten (cudagraph padding, profile/dummy runs)
        # only ever reach row-local ops and are discarded.
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            g1,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            # Profile/warmup dummy runs skip mamba-family metadata.
            return
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        if spec_sequence_masks is not None and self.fused_path:
            raise NotImplementedError(
                "SolarOpen2 fused KDA path does not support speculative decoding."
            )
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        non_spec_committed_state_indices = (
            attn_metadata_narrowed.non_spec_committed_state_indices
        )
        non_spec_num_accepted_tokens = (
            attn_metadata_narrowed.non_spec_num_accepted_tokens
        )
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        constant_caches = self.kv_cache

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        # A speculative step leaves a request's state "rolled": the recurrent
        # state after draft j sits in ladder block j and the conv window starts
        # at column num_accepted - 1. Every non-speculative reader below assumes
        # the canonical position, so fold the committed state back first. No-op
        # when the request accepted exactly one token.
        if (
            non_spec_committed_state_indices is not None
            and non_spec_num_accepted_tokens is not None
            and non_spec_state_indices_tensor is not None
        ):
            blk0 = non_spec_state_indices_tensor
            recurrent_state[blk0] = recurrent_state[non_spec_committed_state_indices]
            # One shift covers q/k/v: they are channel slices of one block.
            shift = (non_spec_num_accepted_tokens.long() - 1).clamp_(min=0)
            state_len = conv_state.size(-1)
            cols = (
                shift.unsqueeze(1)
                + torch.arange(state_len, device=conv_state.device).unsqueeze(0)
            ).clamp_(max=state_len - 1)
            window = conv_state[blk0]
            conv_state[blk0] = window.gather(
                2, cols.unsqueeze(1).expand(-1, window.size(1), -1)
            )

        # Split the batch into the speculative part and the rest. q/k/v are
        # (tokens, dim) while g1 is (1, tokens, h, d) and beta is (1, tokens, h),
        # so they gather on different axes.
        if spec_sequence_masks is not None:
            if (
                attn_metadata_narrowed.num_prefills == 0
                and attn_metadata_narrowed.num_decodes == 0
            ):
                q_spec, k_spec, v_spec = q_proj_states, k_proj_states, v_proj_states
                g1_spec, beta_spec = g1, beta
                q_ns = k_ns = v_ns = g1_ns = beta_ns = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                q_spec = q_proj_states.index_select(0, spec_token_indx)
                k_spec = k_proj_states.index_select(0, spec_token_indx)
                v_spec = v_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                q_ns = q_proj_states.index_select(0, non_spec_token_indx)
                k_ns = k_proj_states.index_select(0, non_spec_token_indx)
                v_ns = v_proj_states.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
        else:
            q_spec = k_spec = v_spec = g1_spec = beta_spec = None
            q_ns, k_ns, v_ns = q_proj_states, k_proj_states, v_proj_states
            g1_ns, beta_ns = g1, beta

        core_attn_out_spec = None
        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            assert q_spec is not None and k_spec is not None and v_spec is not None
            assert g1_spec is not None and beta_spec is not None
            q_spec, k_spec, v_spec = self._spec_qkv_conv(
                q_spec, k_spec, v_spec, conv_state, attn_metadata_narrowed
            )
            q_spec, k_spec, v_spec = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q_spec, k_spec, v_spec),
            )
            # fused_recurrent_kda takes a pre-gated g, while
            # chunk_kda_with_fused_gate gates raw_g itself.
            g1_spec = fused_kda_gate(
                rearrange(g1_spec, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
            ).unsqueeze(0)
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
            )

        core_attn_out_non_spec = None
        if q_ns is not None:
            assert k_ns is not None and v_ns is not None
            assert g1_ns is not None and beta_ns is not None
            if self.fused_path:
                q, k, v = self._fused_qkv_conv(
                    q_ns,
                    k_ns,
                    v_ns,
                    conv_state,
                    attn_metadata_narrowed,
                )
            else:
                q, k, v = self._unfused_qkv_conv(
                    q_ns,
                    k_ns,
                    v_ns,
                    conv_state,
                    attn_metadata_narrowed,
                )

            q, k, v = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q, k, v),
            )

            if attn_metadata_narrowed.num_prefills > 0:
                if self.fused_path:
                    # The fused path carries RAW b logits; gate them here.
                    beta_ns = beta_ns.float().sigmoid()
                    if self.allow_neg_eigval:
                        beta_ns = beta_ns * 2.0
                assert non_spec_state_indices_tensor is not None
                assert has_initial_state is not None
                zero_idx = non_spec_state_indices_tensor[~has_initial_state]
                recurrent_state[zero_idx] = 0
                initial_state = recurrent_state[
                    non_spec_state_indices_tensor
                ].contiguous()
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_kda_with_fused_gate(
                    q=q,
                    k=k,
                    v=v,
                    raw_g=g1_ns,
                    beta=beta_ns,
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                )
                # Init cache
                recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
            elif self.fused_path:
                # Decode: single fused gate+recurrent kernel. Pass RAW g1 (a) and
                # RAW b logits; the kernel applies softplus/-exp(A_log) gating and
                # sigmoid(b)[*2] internally. Decode tokens live at the front of
                # the non-spec batch and are written straight into core_attn_out.
                assert non_spec_query_start_loc is not None
                n_dec = attn_metadata_narrowed.num_decodes
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=rearrange(g1_ns, "1 n h d -> n h d")[:n_dec],
                    b=beta_ns[:, :n_dec],
                    dt_bias=self.dt_bias,
                    q=q[:, :n_dec],
                    k=k[:, :n_dec],
                    v=v[:, :n_dec],
                    initial_state=recurrent_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: n_dec + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=True,
                    neg_eigval=self.allow_neg_eigval,
                    out=core_attn_out[:, :n_dec],
                )
                return
            else:
                assert non_spec_query_start_loc is not None
                g1_ns = fused_kda_gate(
                    rearrange(g1_ns, "1 n h d -> n (h d)"),
                    self.A_log,
                    self.head_dim,
                    g_bias=self.dt_bias,
                ).unsqueeze(0)
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = fused_recurrent_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g1_ns,
                    beta=beta_ns,
                    initial_state=recurrent_state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata_narrowed.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                )

        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            # Mixed batch: scatter both halves back to the batch's token order.
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_spec.dtype,
                device=core_attn_out_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged[0, :num_actual_tokens]
        elif core_attn_out_spec is not None:
            core_attn_out[0, :num_actual_tokens] = core_attn_out_spec[
                0, :num_actual_tokens
            ]
        else:
            assert core_attn_out_non_spec is not None
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]

    def _spec_qkv_conv(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        conv_state: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        assert spec_state_indices_tensor is not None
        # The conv state lives in the first block of each request's ladder; the
        # recurrent kernel takes the whole 2-D ladder instead.
        conv_indices = spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes]
        max_query_len = spec_state_indices_tensor.size(-1)
        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        def _conv(x, conv_state_chunk, conv1d):
            return causal_conv1d_update(
                x,
                conv_state_chunk,
                conv1d.weight.view(conv1d.weight.size(0), conv1d.weight.size(2)),
                conv1d.bias,
                activation="silu",
                conv_state_indices=conv_indices,
                num_accepted_tokens=attn_metadata.num_accepted_tokens,
                query_start_loc=attn_metadata.spec_query_start_loc,
                max_query_len=max_query_len,
                # The speculative window is width - 1 + num_spec wide, which the
                # kernel's own bounds check does not expect.
                validate_data=False,
            )

        q = _conv(q_proj_states, conv_state_q, self.q_conv1d)
        k = _conv(k_proj_states, conv_state_k, self.k_conv1d)
        v = _conv(v_proj_states, conv_state_v, self.v_conv1d)
        return q, k, v

    def _fused_qkv_conv(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        conv_state: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ONE causal conv over concat[q,k,v] via the merged conv1d weight, then
        # split. The conv input is a zero-copy strided view over the adjacent
        # q/k/v slices of the fused in_proj output (_fused_qkv_view); the split
        # yields strided views that the fused decode kernel reads in place.
        qkv_conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        split_size = q_proj_states.shape[-1]
        state_indices = attn_metadata.non_spec_state_indices_tensor
        if attn_metadata.num_prefills > 0:
            qkv = _fused_qkv_view(q_proj_states, k_proj_states, v_proj_states)
            qkv = causal_conv1d_fn(
                qkv.transpose(0, 1),
                qkv_conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=attn_metadata.has_initial_state,
                cache_indices=state_indices,
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        else:
            assert state_indices is not None
            qkv = causal_conv1d_update(
                _fused_qkv_view(q_proj_states, k_proj_states, v_proj_states),
                conv_state,
                qkv_conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=state_indices[: attn_metadata.num_actual_tokens],
                validate_data=True,
            )
        q, k, v = qkv.split(split_size, dim=-1)
        return q, k, v

    def _unfused_qkv_conv(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        conv_state: torch.Tensor,
        attn_metadata_narrowed: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )
        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )
        if attn_metadata_narrowed.num_prefills > 0:
            q_proj_states = q_proj_states.transpose(0, 1)
            k_proj_states = k_proj_states.transpose(0, 1)
            v_proj_states = v_proj_states.transpose(0, 1)
            q = causal_conv1d_fn(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
        else:
            assert non_spec_state_indices_tensor is not None
            num_decodes = attn_metadata_narrowed.num_decodes
            # One block id per sequence, not per token.
            decode_conv_indices = non_spec_state_indices_tensor[:num_decodes]
            # Speculative decoding widens the reorder threshold to num_spec + 1,
            # so a draft-free batch can still be a "decode" with several tokens
            # per sequence (e.g. FULL cudagraph capture). The kernel infers its
            # batch from x.shape[0] unless handed the sequence boundaries.
            if q_proj_states.size(0) == num_decodes:
                decode_query_start_loc = None
                decode_max_query_len = -1
            else:
                assert non_spec_query_start_loc is not None
                decode_query_start_loc = non_spec_query_start_loc[: num_decodes + 1]
                decode_max_query_len = self.num_spec + 1
            q = causal_conv1d_update(
                q_proj_states,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                query_start_loc=decode_query_start_loc,
                max_query_len=decode_max_query_len,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                query_start_loc=decode_query_start_loc,
                max_query_len=decode_max_query_len,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                query_start_loc=decode_query_start_loc,
                max_query_len=decode_max_query_len,
                validate_data=True,
            )
        return q, k, v


class SolarOpen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        if is_gqa_layer(config, layer_idx):
            self.self_attn = SolarOpen2Attention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position_embeddings=max_position_embeddings,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=config.attention_bias,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                use_qk_norm=getattr(config, "use_qk_norm", False),
            )
        else:
            self.self_attn = SolarOpen2KimiDeltaAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = SolarOpen2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = SolarOpen2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class SolarOpen2Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.fused_kda_path = _use_amd_fused_kda(vllm_config)

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SolarOpen2DecoderLayer(
                config=config,
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            SolarOpen2MoE,
            "mlp",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # Fused shared experts occupy the appended expert slots.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if self.is_fused_shared_expert_enabled
                else 0
            ),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        qkv_shard_names = ("q_proj", "k_proj", "v_proj")

        # ROCm/AITER fused KDA in_proj/conv1d routing. When off, KDA layers keep
        # separate projections/convs, which fall through to the default loaders
        # below (q/k/v via the qkv_shard_names skip, the rest as plain params).
        fused_path = self.fused_kda_path
        if fused_path:
            # q/k/v/b are standard merged shards (source narrowed by tp_rank);
            # f_a/g_a are DUPLICATED shards (full source broadcast to every rank's
            # slice). Shard order matches the KDA in_proj output_sizes.
            kda_std_shard = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "b_proj": 3}
            kda_dup_shard = {"f_a_proj": 4, "g_a_proj": 5}
            kda_conv_shard = {"q_conv1d": 0, "k_conv1d": 1, "v_conv1d": 2}
            kda_head_dim = self.config.linear_attn_config["head_dim"]
            kda_num_heads = self.config.linear_attn_config["num_heads"]
            tp_size = get_tensor_model_parallel_world_size()
            kda_proj_size = kda_head_dim * kda_num_heads
            kda_in_proj_output_sizes = [
                kda_proj_size,
                kda_proj_size,
                kda_proj_size,
                kda_num_heads,
                kda_head_dim * tp_size,
                kda_head_dim * tp_size,
            ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue
            parts = name.split(".")
            if fused_path and "layers" in parts:
                j = parts.index("layers")
                layer_idx = (
                    int(parts[j + 1])
                    if j + 1 < len(parts) and parts[j + 1].isdigit()
                    else None
                )
                if layer_idx is not None and not is_gqa_layer(self.config, layer_idx):
                    # conv1d routing first (q_conv1d must not be caught by q_proj).
                    src = next((s for s in kda_conv_shard if f".{s}." in name), None)
                    if src is not None:
                        pname = name.replace(src, "conv1d")
                        param = params_dict[pname]
                        param.weight_loader(param, loaded_weight, kda_conv_shard[src])
                        loaded_params.add(pname)
                        continue
                    src = next(
                        (
                            s
                            for s in (*kda_std_shard, *kda_dup_shard)
                            if f".{s}." in name
                        ),
                        None,
                    )
                    if src is not None:
                        pname = name.replace(src, "in_proj")
                        param = params_dict[pname]
                        if src in kda_std_shard:
                            param.weight_loader(
                                param, loaded_weight, kda_std_shard[src]
                            )
                        else:
                            _load_in_proj_dup_shard(
                                param,
                                loaded_weight,
                                kda_dup_shard[src],
                                kda_in_proj_output_sizes,
                                kda_head_dim,
                                tp_size,
                            )
                        loaded_params.add(pname)
                        continue

            is_fusion_moe_shared_experts_layer = (
                self.is_fused_shared_expert_enabled and ("mlp.shared_experts" in name)
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # KDA (non-GQA) layers have their own q_proj/k_proj/v_proj
                # ColumnParallelLinear modules that must NOT be folded into
                # qkv_proj; let them fall through to the default loader.
                if weight_name in qkv_shard_names and name.startswith("layers."):
                    layer_idx = int(name.split(".")[1])
                    if not is_gqa_layer(self.config, layer_idx):
                        continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                # Fused shared experts load through expert_params_mapping below.
                if is_fusion_moe_shared_experts_layer:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                # A fused checkpoint ships one (possibly widened) shared_experts
                # tensor without expert indices; split it evenly into
                # n_shared_experts slices loaded into the appended slots
                # mlp.experts.{n_routed_experts + j}.*. gate/up split along
                # dim 0, down along dim 1.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # This is an expert weight and should not be attempted
                        # to load as other weights later.
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here.
                        # Instead, create a new variable.
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # It is an expert weight but not mapped to this rank.
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if name is not None and not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params


class SolarOpen2ForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsPP,
    SupportsLoRA,
    SupportsEagle3,
    MixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        # KDA full fused input projection: in_proj = concat[q,k,v,b,f_a,g_a].
        # Registering it lets the quark resolver decompose the fused name into its
        # shards (all in the checkpoint's quant exclude list), so in_proj is built
        # bf16 like its shards instead of falling through to the global fp4 config.
        "in_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = SolarOpen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights = []

        # Set MoE hyperparameters
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = config.n_group

        self.moe_layers = []
        self.moe_mlp_layers: list[SolarOpen2MoE] = []

        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, SolarOpen2DecoderLayer)
            if isinstance(layer.mlp, SolarOpen2MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def extract_moe_parameters(self, example_moe: "SolarOpen2MoE | None") -> None:
        if example_moe is None:
            raise RuntimeError("No SolarOpen2MoE layer found in model.layers.")
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def get_spec_layer_idx_from_weight_name(config, weight_name: str) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if f"layers.{layer_idx + i}." in weight_name:
                return layer_idx + i
    return None
