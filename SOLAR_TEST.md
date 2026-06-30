# Solar-Open — Tested Configuration & Porting Notes

Support for Upstage **Solar-Open** (`SolarOpenForCausalLM`, `model_type:
solar_open`) on vLLM v0.22.1 (ROCm). Validated on AMD Instinct MI355 with the
`Solar-Open-100B` (bf16) and `Solar-Open-100B-MXFP4` (quark MXFP4) checkpoints.

## Architecture

DeepSeek-V2-style MoE with Llama/Qwen3-style attention:

- Dense + MoE decoder layers (`first_k_dense_replace` dense layers, then MoE).
- Routed experts: sigmoid scoring + `e_score_correction_bias`, grouped top-k
  (`n_group`/`topk_group`), `norm_topk_prob`, `routed_scaling_factor`.
- One shared expert (bf16, runs alongside the routed experts).
- Attention: GQA, optional qk-norm (`use_qk_norm`, absent on current
  checkpoints), partial rotary factor, YaRN rope.

The HF `SolarOpenConfig` (transformers >= 5.10) supplies defaults the
checkpoint may omit (`n_group=1`, `topk_group=1`, `attention_bias=False`) and
synthesizes `rope_parameters` from `rope_scaling`. `use_qk_norm` has no default
and is read with `getattr(config, "use_qk_norm", False)`.

## Files

New:
- `vllm/model_executor/models/solar_open.py`
- `vllm/reasoning/solar_open_reasoning_parser.py`
- `vllm/tool_parsers/solar_open_tool_parser.py`
- `vllm/model_executor/layers/fused_moe/configs/E={16,32}_N=1280` and
  `E=128_N={160,320}` `device_name=NVIDIA_B200.json` (MoE tuning).

Registered via lazy entries in `registry.py`, `reasoning/__init__.py`,
`tool_parsers/__init__.py`.

## Porting notes (against current vLLM MoE API)

- `SharedFusedMoE` (removed) -> `FusedMoE(gate=..., shared_experts=...)`.
- Gate is `GateLinear`; routing reduction is internal to `FusedMoE` (no
  `reduce_results` kwarg, no manual all-reduce).
- Expert weight mapping via `fused_moe_make_expert_params_mapping(self, ...)`
  (now takes the module as first arg).
- Pass `router_logits_dtype=gate.out_dtype`.
- `get_rope(head_dim, max_position=..., rope_parameters=config.rope_parameters)`
  (no `rotary_dim`/`partial_rotary_factor` kwargs; YaRN derived from
  `rope_parameters`).
- `Attention` imported from `vllm.model_executor.layers.attention`.
- Reasoning/tool parser protocol types come from the
  `entrypoints/openai/{engine,chat_completion,responses}/protocol.py` modules.
- Tool parser uses `pyjson5` only if available (graceful JSON / `ast` fallback).
- ROCm aiter shared-expert fusion is supported and gated behind
  `rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()` (dormant by default).

## Tested serving configuration

bf16 (`Solar-Open-100B`), TP=4:

```
SAFETENSORS_FAST_GPU=1
vllm serve <model> --tensor-parallel-size 4 --trust-remote-code \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --enforce-eager
```
- Use the native (non-aiter) MoE path for bf16: do NOT set
  `VLLM_ROCM_USE_AITER_MOE` (the aiter MoE GEMM rejects bf16 expert shapes).

MXFP4 (`Solar-Open-100B-MXFP4`, quark W4A4), TP=4:

```
SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1
vllm serve <model> --tensor-parallel-size 4 --enable-expert-parallel \
  --trust-remote-code --max-model-len 32768 \
  --max-num-batched-tokens 4096 --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 --enable-chunked-prefill --async-scheduling \
  --no-enable-prefix-caching
```
- Quark auto-detects from the checkpoint; do NOT pass `--quantization mxfp4`
  (it conflicts with the `quark` method declared in the config).
- It is an instruct/reasoning model with a chat template emitting
  `<|think|>...<|end|>` / `<|content|>`; drive it via the chat endpoint.

### MXFP4 + tensor parallelism: expert parallel is REQUIRED

The CK MXFP4 MoE GEMM requires the per-partition intermediate size to be a
multiple of 256. With `moe_intermediate_size=1280`, a plain TP split is not
256-aligned (TP2 -> 640, TP4 -> 320), so vLLM pads it and the padded FP4
block-scale rows are mishandled, producing degenerate output. Observed:

| Config                         | per-GPU intermediate | result    |
|--------------------------------|----------------------|-----------|
| MXFP4 TP=1                     | 1280 (aligned)       | correct   |
| MXFP4 TP=2 / TP=4 (no EP)      | 640 / 320 (padded)   | degenerate|
| MXFP4 TP=4 + expert parallel   | 1280 (whole expert)  | correct   |
| bf16 TP=4                      | n/a                  | correct   |

**Fix:** run MXFP4 with `--enable-expert-parallel` at any TP>1 (each expert
stays whole, no padding). 128 routed experts over TP=4 = 32 experts/GPU.

## Reasoning / tool parsers

`--reasoning-parser solar_open` and `--tool-call-parser solar_open` parse the
Solar special-token format (`<|think|>`, `<|content|>`, `<|tool_calls|>`,
`<|tool_call:begin|>` ...). Both are registered and import-clean.

## Relation to the UpstageAI vLLM fork

This is an independent forward-port. The upstream Solar-Open support
(github.com/UpstageAI/vllm, `v0.12.0-solar-open`) was written against vLLM
v0.12.0; this port targets v0.22.1, whose MoE / serving / parser APIs have
since changed substantially. Notable differences:

- **MoE rewritten for the current API.** The fork used `SharedFusedMoE`
  (removed in v0.22.1) with a manual all-reduce and an `nn.Linear` gate. This
  port uses `FusedMoE(gate=GateLinear, shared_experts=..., router_logits_dtype=
  ...)` with internal reduction, and `fused_moe_make_expert_params_mapping`
  (now module-first), plus ROCm-aiter shared-expert-fusion handling.
- **rope / attention API.** `get_rope` no longer takes `rotary_dim` /
  `partial_rotary_factor`; `Attention` moved to
  `vllm.model_executor.layers.attention`.
- **Parser locations.** Tool parsers now live under `vllm/tool_parsers/`, and
  protocol types under `entrypoints/openai/{engine,chat_completion,responses}/`.
- **Additive registration only.** The fork edited shared serving files
  (`serving_chat.py`, `protocol.py`, `sampling_params.py`, ...) — including a
  hard-coded `tool_call_id_type = "solar_open"` affecting all models. This port
  touches only lazy-registry entries, so no other model's behavior changes.
- **Reasoning parser.** Dropped the fork's global `json` encoder monkey-patch;
  `pyjson5` is an optional import with a JSON/`ast` fallback.
- **Scope.** The fork's optional token-level logits processors
  (`solar_open_logits_processor.py`, `parallel_tool_call_logits_processor.py`)
  are not ported; they are standalone and not required to run the model.
