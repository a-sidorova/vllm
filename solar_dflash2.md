# Solar Pro4 DFlash2 (MI355X, TP4)

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KDA_SPEC_DECODE=1
export VLLM_DFLASH_DRAFT_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1

vllm serve /models/solar-w4a4 \
  --served-model-name=solar-pro4 \
  --host=0.0.0.0 \
  --port=8000 \
  --trust-remote-code \
  --dtype=auto \
  --tensor-parallel-size=4 \
  --enable-expert-parallel \
  --gpu-memory-utilization=0.90 \
  --max-model-len=65536 \
  --max-num-seqs=64 \
  --max-num-batched-tokens=65536 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --attention-backend=ROCM_AITER_UNIFIED_ATTN \
  --compilation-config='{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config='{"method":"dflash","model":"/models/dflash2-v1-epoch3","num_speculative_tokens":7,"draft_sample_method":"greedy","draft_tensor_parallel_size":4,"attention_backend":"TRITON_ATTN"}' \
  --tool-call-parser=solar_open2 \
  --reasoning-parser=solar_open2 \
  --enable-auto-tool-choice \
  --chat-template=/app/solar-open2-v0.6.jinja \
  --default-chat-template-kwargs='{"think_render_option":"interleaved","reasoning_effort":"none"}'
```

`VLLM_KDA_SPEC_DECODE=1` is required: Solar-Open2 refuses speculative decoding
without it. `VLLM_DFLASH_DRAFT_CUDAGRAPH=0` runs the draft model eagerly while
the target keeps its FULL decode CUDA graphs. `"cudagraph_mode":"PIECEWISE"`
also works.
