# Solar-Open2 (MXFP4 + FP8) — "1 Oct" fused-router image

vLLM ROCm image for Solar-Open2 on MI355X:

> Baseline + eliminated element-wise small kernels + fused MoE-router kernel
> (`VLLM_ROCM_USE_AITER_FUSED_ROUTER=1`, ROCm/aiter#5150) for **MXFP4 and FP8
> per-token** experts + tuned decode MXFP4 MoE + tuned FP8 35B MoE and a8w8
> GEMMs.

The "8 Sep combined" image (`vllm-solar:0908-combined`, the image behind
`SOLAR_35B_MXFP4_TP1_MI355X_reference.html`) plus the FP8 router: aiter
`adcfd7630` and engine `d8cde836bd`, one commit on each side.

One image serves all three checkpoints; only the launch arguments differ. The
35B MXFP4 runs TP1 (the reference report); the 35B FP8 runs TP1 (§Run); the
250B MXFP4 runs TP4 with expert parallel, per §2 of
`SOLAR_250B_V025_UPDATE_08_09_2026.html`.

Self-contained: engine, router kernel and tuning are all in the image.

## Inputs

| | Value |
|---|---|
| Base image | `vllm/vllm-openai-rocm:v0.25.1`, pinned by digest `sha256:84459732…` |
| Engine | `asidorov/features/solar_35b_fp8_model_optimizations` @ `d8cde836bd` (`8665505227` + this directory + the FP8 router commit) |
| Engine remote | `github.com/UpstageAI/upstage-vllm-amd` (**private**) |
| Router | ROCm/aiter#5150 + FP8 per-token mode @ `adcfd7630` (on top of `7fb161ea1`, the 8 Sep router) |
| Router remote | `github.com/sshlyapn/aiter`, branch `asidorov/features/fused_router_fp8` |
| Target arch | `gfx950` (MI355X only) |
| AITER / flydsl | `0.1.16.post2` / `0.2.0` — both untouched |

## Build

Requires: Docker and a ROCm host. No GPU needed.

```bash
git clone -b asidorov/features/solar_35b_fp8_model_optimizations \
    https://github.com/UpstageAI/upstage-vllm-amd
cd upstage-vllm-amd/solar-image
./fetch-sources.sh
docker build -t vllm-solar:1001-fp8 .
```

Do not `--depth 1` the clone: `fetch-sources.sh` exports the engine from
`d8cde836bd` in your own checkout, so a full clone needs no network at all. On
a shallow clone it falls back to fetching that commit from the remote.

Takes ~15 min: ~10 for `skinny_gemms.hip.o`, ~3.5 for `module_gemm_a8w8`,
~35 s for the router.

`fetch-sources.sh` pins both revisions by commit and md5-checks all six router
files and the four tuning CSVs, so it either reproduces the exact intended
inputs or fails loudly.

## Verify

Both need a GPU.

```bash
GPU="--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host
     --shm-size 16G -e HIP_VISIBLE_DEVICES=0"

# 1. Is the image assembled correctly?
docker run --rm $GPU --entrypoint verify-fused-router vllm-solar:1001-fp8

# 2. Does the kernel actually run?
docker run --rm $GPU --entrypoint bash \
  -v "$PWD/router_smoke.py:/tmp/rs.py:ro" \
  vllm-solar:1001-fp8 -lc 'python3 /tmp/rs.py'
```

Check (1) launches nothing. It asserts:

- the overlay landed, and every kwarg `vllm/_aiter_ops.py` forwards exists on
  `fused_moe_router`;
- the router accepts MXFP4 and FP8 per-token at hidden 2048/4096, with and
  without a fused shared expert, and refuses hidden 1024 and MXFP8;
- the FP8 engine path is present: the FP8 oracle offers
  `AiterFusedRouterExperts`, and `QuarkW8A8Fp8MoEMethod` has `apply_monolithic`;
- `module_gemm_a8w8.so` is newer than the a8w8 table it should be built from.

It catches a botched overlay or signature drift that would otherwise surface
as a `TypeError` deep inside the first decode batch.

Check (2) launches the router at two decode shapes — MXFP4 250B (hidden 4096,
80 experts, topk 7) and FP8 35B (hidden 2048, 128 experts, topk 4) — and
asserts shape and finiteness. The first call should take seconds at most; ~35 s
means the router was not compiled into the image.

## Run

`VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_FUSED_ROUTER=1` are baked in;
the router needs both.

### 35B MXFP4, TP1 — the configuration the reference report measures

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  -v /path/to/models:/models \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm-solar:1001-fp8 \
  /models/solar-35b \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --quantization=quark \
  --tensor-parallel-size=1 \
  --attention-backend=ROCM_AITER_UNIFIED_ATTN \
  --max-model-len=33920 --max-num-seqs=512 \
  --compilation-config='{"max_cudagraph_capture_size":512}' \
  --max-num-batched-tokens=65536 \
  --gpu-memory-utilization=0.9 --no-enable-prefix-caching \
  --chat-template=/models/solar-35b/chat_template.jinja
```

`--chat-template` is required: the checkpoint is a reasoning model and the
template emits the opening `<|think:start|>`. Without it, output format and
accuracy both degrade.

### 35B FP8, TP1

The cfg7 configuration (`benchmark/configs/35b_fp8/cfg7_*`): FP8 weights and
activations, FP8 KV cache, the FP8 router, and the tuned FP8 MoE and a8w8
tables — all of which the image now supplies by default.

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  -v /path/to/models:/models \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm-solar:1001-fp8 \
  /models/solar-35b-fp8 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --quantization=quark \
  --tensor-parallel-size=1 \
  --attention-backend=ROCM_AITER_UNIFIED_ATTN \
  --max-model-len=33920 --max-num-seqs=592 \
  --compilation-config='{"max_cudagraph_capture_size":592}' \
  --max-num-batched-tokens=65536 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.9 --no-enable-prefix-caching \
  --chat-template=/models/solar-35b-fp8/chat_template.jinja \
  --logits-processors \
  vllm.model_executor.models.solar_open2_logits_processor:SolarOpen2TemplateLogitsProcessor
```

The router covers Quark FP8 with per-channel weights and dynamic per-token
activations only. Static activation scales or block-quantized FP8 decline at
selection and run the unfused path.

### 250B MXFP4, TP4 + expert parallel

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  -v /path/to/models:/models \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  vllm-solar:1001-fp8 \
  /models/Solar-Open2-250B-mxfp4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --quantization quark \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --max-model-len 65536 --max-num-seqs 128 \
  --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
  --chat-template /models/Solar-Open2-250B-mxfp4/chat_template.jinja \
  --logits-processors \
  vllm.model_executor.models.solar_open2_logits_processor:SolarOpen2TemplateLogitsProcessor
```

### Tuning CSVs

Baked into `/tuning`, with the env vars preset — nothing to mount:

| Env var | File | Contents |
|---|---|---|
| `AITER_CONFIG_FMOE` | `fmoe_tuned_solar.csv` | Written at build by `tuning_check.py`: the two tables below, concatenated. |
| | ↳ `fmoe_tuned_mxfp4_decode_solar.csv` | 38 rows. MXFP4: 80 routed experts / topk 7 (§2) at padded_M 64 and 128; 81 experts / topk 8 (fused shared expert) at padded_M 1–128, tuned + `flydsl_fallback`. |
| | ↳ `fmoe_tuned_35b_fp8_tp1.csv` | 14 rows. FP8 per-token, 128 experts / topk 4, token buckets 1–32768 (8 and 256 deliberately absent). From `tuning/fp8_35b/fp8_moe/`, the table cfg7 benchmarked. |
| `AITER_CONFIG_GEMM_BF16` | `aiter_gemm_tuning.csv` | 1612 rows, 80 N/K pairs, gfx950 / gfx942 / gfx1250. Same schema as AITER's stock `configs/bf16_tuned_gemm.csv`, which it replaces wholesale. |
| `AITER_CONFIG_GEMM_A8W8` | `a8w8_tuned_gemm_35b_fp8.csv` | 379 rows, 6 N/K pairs, gfx950 only: every a8w8 linear of the 35B FP8 checkpoint. `tuning/fp8_35b/kda_proj_plain/a8w8_tuned_gemm_plain_full.csv`. Replaces the stock `a8w8_tuned_gemm.csv` wholesale — only FP8 checkpoints run a8w8 GEMMs. |

`AITER_CONFIG_FMOE` and `AITER_CONFIG_GEMM_BF16` are read at runtime: point
either elsewhere to override, no rebuild needed.

**`AITER_CONFIG_GEMM_A8W8` is different.** The table is compiled into
`module_gemm_a8w8.so` as a lookup map, and AITER never rebuilds an existing
`.so` when the variable changes. Pointing it elsewhere at runtime changes only
the Python-side splitK lookup, not the kernels. To change the table, rebuild
the module (step 5), e.g. by editing the CSV and rebuilding the image.

The build fails if a file does not parse, has null cells, or has duplicate
index rows (which would make the lookup arbitrary). It also fails if the a8w8
lookup header does not carry exactly one entry per CSV row, since the codegen
silently drops rows for any other `(gfx, cu_num)`.

The MoE tables are merged at build time rather than given to AITER as a
colon-separated list. AITER would merge such a list itself, but it redoes the
merge on every start, and on a duplicate it rewrites the source files in
place.

Outside those shapes AITER falls back to heuristics, which is the intended
behaviour — the 80-expert shape is measured at exactly two token counts.

### Confirming the router engaged

`AiterFusedRouterExperts` is a Python class, so it never appears in a rocprof
trace. The only signal is the startup log, where the oracle logs the selected
class (`Using AiterFusedRouterExperts` from `oracle/mxfp4.py` or
`oracle/fp8.py`). A refusal is logged at debug level, so use
`VLLM_LOGGING_LEVEL=DEBUG` to see the reason.

The router is capped at 128 tokens and requires `num_expert_group == 1` and
`topk_group == 1`. It is a decode optimization and will not fire on large
prefills.

## Files

| File | Role | Runs |
|---|---|---|
| `Dockerfile` | the build | `docker build` |
| `fetch-sources.sh` | fetches `vllm-src/` + `aiter-overlay/` by commit | before build |
| `.dockerignore` | keeps `.git` and `__pycache__` out of the context | `docker build` |
| `tuning/*.csv` | AITER fmoe, bf16 and a8w8 GEMM configs, copied to `/tuning` | in build, step 4b |
| `tuning_check.py` | validates the CSVs, writes the merged fmoe table | in build, step 4b |
| `build_kernel.py` | compiles the router and `module_gemm_a8w8` into the image | in build, step 5 |
| `verify.sh` | baked in as `verify-fused-router` | after build, GPU |
| `router_smoke.py` | real kernel launches, MXFP4 and FP8 | after build, GPU |

`vllm-src/` (93 MB) and `aiter-overlay/` (280 KB) are generated by
`fetch-sources.sh`. Neither should be committed.

## How it works

**Engine (steps 1–2).** The branch is `v0.25.1` + 22 commits, and its only
non-Python file is `csrc/rocm/skinny_gemms.cu` (wvSplitK out-of-bounds fix).
That one file is the entire reason a compile happens. `build_ext` rebuilds all
33 HIP objects and 4 modules, but only `_rocm_C.abi3.so` is copied into
site-packages, since `skinny_gemms.cu` lands in that module alone.

The FP8 commit (`d8cde836bd`) is Python only. It maps Quark FP8 PTPC to the
router in `rocm_aiter_moe.py`, has `oracle/fp8.py` offer
`AiterFusedRouterExperts` first when the flag is on, as the MXFP4 oracle does,
and adds `QuarkW8A8Fp8MoEMethod.apply_monolithic`. The same commit sits on
`sshliapn/features/fused_router_fp8` as `e2330b3615`; that branch adds only
KV-connector / MORI changes this image does not need.

**Router (step 3).** Injected as six files rather than by reinstalling AITER.
This is safe because the branch sits on exactly the AITER revision the image
ships: base `620287969`, then the MXFP4 router (`25979546b`), hidden 2048
(`7fb161ea1`) and the FP8 mode (`adcfd7630`). Before the router commits,
`fused_moe.py`, `__init__.py` and `optCompilerConfig.json` all md5-match the
installed copies. So no AITER drift and no flydsl bump. The alternative
branches are based on `v0.1.21`/`v0.1.22` and pin flydsl 0.3.2, which would
mean replacing both packages wholesale.

The FP8 mode changes four of the six files. `__init__.py` and
`optCompilerConfig.json` are byte-identical to the 8 Sep image, so the
module name and its sources are unchanged: the FP8 kernels are extra
instantiations in the same translation unit.

Files land in two trees because AITER resolves kernel sources from
`aiter_meta/` (a *sibling* of `aiter/`) but reads the build registry from the
package:

```
aiter/fused_moe.py, aiter/__init__.py,
aiter/ops/fused_moe_router.py,
aiter/jit/optCompilerConfig.json             -> package  (105 -> 106 modules)
csrc/kernels/fused_moe_router.cu,
csrc/py_itfs_cu/fused_moe_router_entry.cu    -> aiter_meta
```

The `.cu` files are one translation unit — the entry `#include`s the kernel —
so registering only the entry in `srcs` is correct.

**FP8 data path.** In FP8 mode the router's phase 1 reproduces
`dynamic_per_token_scaled_quant_kernel` and writes a token-indexed `[M, K]`
fp8 / `[M, 1]` fp32 input. The 2-stage path takes it via `a1_prequant`. The
1-stage path, which the default heuristic picks from M=17, passes it as-is to
`fmoe_g1u1`. The expert kernels it lands on — `module_moe_asm` and the
`f8_f8 … silu_per_token` CK 2-stage modules — ship prebuilt in the base image,
so they add no first-call JIT.

**Kernels (step 5).** Two modules are compiled into image layers:

- `module_fused_moe_router`. AITER would otherwise JIT it on the first decode
  batch (~35 s). That write lands in the *container's* writable layer, not
  the image, so every fresh container pays it again, inside its first request.
- `module_gemm_a8w8`. The base image's AITER is built with
  `PREBUILD_KERNELS=1`, so this module already exists, built from the stock
  table, and is never rebuilt on its own. Step 5 rebuilds it from
  `/tuning/a8w8_tuned_gemm_35b_fp8.csv` (~3.5 min).

No GPU is required because only the compile runs, never a kernel. Two device
probes on the `import aiter` path are stubbed in `build_kernel.py`:
`chip_info._detect_native()` shells out to `rocminfo` (pinned via `GPU_ARCHS`),
and triton's `arch_info` probes the active driver then falls back to `jax`,
which is not installed. These stubs track AITER's current import chain; a new
import-time device probe would break the build loudly, which is the right
failure mode. `GPU_ARCHS=gfx950` also fixes the a8w8 codegen's build target at
`cu_num` 256, which is what the tuned table is keyed on.

## Two traps this Dockerfile works around

1. **`PYTORCH_ROCM_ARCH` cannot be an ARG here.** The base image sets it as an
   `ENV` (the full 9-arch list), and ENV beats a same-named ARG during
   expansion — the override is silently ignored and you get a 9-arch build that
   takes ~9x longer. The ARG is named `SOLAR_ROCM_ARCH` for this reason.
2. **`setuptools_scm` has no `.git`.** Sources are exported with `git archive`,
   so the version must be pinned via `SETUPTOOLS_SCM_PRETEND_VERSION` or
   `setup.py` aborts.

## Rebuilding the C++ extension later

The image keeps a full HIP toolchain (`hipcc`, `cmake`, `ninja`, `g++`) but not
the vLLM C++ sources — step 2 deletes `/vllm_src`. Mount them back:

```bash
docker run --rm -v "$PWD/vllm-src:/src:ro" --entrypoint bash \
  vllm-solar:1001-fp8 -lc '
    cp -r /src /w && cd /w
    PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=$(nproc) \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.25.2.dev22+gd8cde836bd \
      python3 setup.py build_ext --inplace'
```

Produces a 91 MB gfx950 `_rocm_C.abi3.so`, no GPU needed. `build_rust` fails at
the end with `can't find Rust compiler` — harmless, it runs after the C++ work
and the image already ships `_rust_tool_parser.abi3.so`. Compiling the `.cu` by
hand with `hipcc` fails on `torch/all.h` unless you add
`torch/include/torch/csrc/api/include`; let `setup.py` assemble the flags.

## Reproducibility

Every input is pinned: both sources by commit, the base image by digest, the
six router files and four tuning CSVs md5-checked.

Compiled *outputs* are not bit-reproducible — the HIP compiler is
nondeterministic. Two builds of the 8 Sep image from identical sources produced
router kernels differing in 27752 of 2986616 bytes, both verified working.
Judge these artifacts by behaviour, not by hash.

## Status

Not built as an image yet. Each step was run outside Docker, on a host
running the same base image with the same AITER and engine revisions:

- `fetch-sources.sh` resolves both pins, and all six overlay md5s and four
  tuning md5s match.
- The step-4 FP8 checks pass on `d8cde836bd`. The `apply_monolithic` check
  correctly rejects the MXFP4-only engine `8665505227`.
- `tuning_check.py` passes. The merged fmoe table is the two source tables
  byte for byte (52 rows).
- `build_kernel.py module_fused_moe_router module_gemm_a8w8` with
  `GPU_ARCHS=gfx950` builds the router in 34 s (3.6 MB, MXFP4 + FP8) and
  `module_gemm_a8w8` in 209 s, with exactly 379 tuned shapes in the lookup
  header.
- `verify.sh` passes on gfx950 against those modules.
- `router_smoke.py` passes both shapes. The FP8 launch takes the tuned 1-stage
  row from the merged table.

Not verified:

- `docker build` itself, the in-image `vllm serve`, and any benchmark or
  accuracy run on this image.
- The FP8 throughput figures come from cfg7, which ran the same revisions
  outside this image.
- The wvSplitK fix is compiled in, but the fault it repairs was not
  reproduced.
