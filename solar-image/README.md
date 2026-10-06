# Solar-Open2 (MXFP4 + FP8) — v0.31.0 fused-router image

vLLM ROCm image for Solar-Open2 on MI355X:

> Baseline + eliminated element-wise small kernels + fused MoE-router kernel
> (`VLLM_ROCM_USE_AITER_FUSED_ROUTER=1`, ROCm/aiter#5150) for **MXFP4 and FP8
> per-token** experts + tuned decode MXFP4 MoE + tuned FP8 35B MoE and a8w8
> GEMMs.

The "1 Oct" image (`vllm-solar:1001-fp8`, v0.25.1-based) moved onto vLLM
v0.31.0: the Solar-Open2 series ported to `asidorov/port/solar-v0.31.0`
(`d05acb0848`), and the router rebased onto AITER v0.1.23 (`9689185d3`), the
AITER the v0.31.0 image ships.

One image serves all three checkpoints; only the launch arguments differ. The
35B MXFP4 runs TP1 (the reference report); the 35B FP8 runs TP1 (§Run); the
250B MXFP4 runs TP4 with expert parallel, per §2 of
`SOLAR_250B_V025_UPDATE_08_09_2026.html`.

Self-contained: engine, router kernel and tuning are all in the image.

## Inputs

| | Value |
|---|---|
| Base image | `vllm/vllm-openai-rocm:v0.31.0`, pinned by digest `sha256:749f6f3f…` |
| Engine | `asidorov/port/solar-v0.31.0` @ `d05acb0848` (`v0.31.0` + 19 commits: the Solar-Open2 series, the fused router, this directory) |
| Engine remote | `github.com/UpstageAI/upstage-vllm-amd` (**private**) |
| Router | ROCm/aiter#5150 + hidden 2048 + FP8 per-token mode @ `9689185d3` (4 commits on `v0.1.23`, `50da036ac`) |
| Router remote | `github.com/sshlyapn/aiter`, branch `sshliapn/feature/fused_router_0.1.23` |
| Target arch | `gfx950` (MI355X only) |
| AITER / flydsl | `0.1.23` / `0.3.4.1` — both untouched |

## Build

Requires: Docker and a ROCm host. No GPU needed.

```bash
git clone -b asidorov/port/solar-v0.31.0 \
    https://github.com/UpstageAI/upstage-vllm-amd
cd upstage-vllm-amd/solar-image
./fetch-sources.sh
docker build -t vllm-solar:v031-fp8 .
```

Do not `--depth 1` the clone: `fetch-sources.sh` exports the engine from
`d05acb0848` in your own checkout, so a full clone needs no network at all. On
a shallow clone it falls back to fetching that commit from the remote.

Outside Docker on a 256-core host: ~4 min for `_rocm_C`, ~35 s for the router,
and 4–12 min for `module_gemm_a8w8` (209 s alone on v0.25.1; 686 s on 0.1.23
while sharing the host with the `_rocm_C` build).

`fetch-sources.sh` pins both revisions by commit and md5-checks all nine router
files and the four tuning CSVs, so it either reproduces the exact intended
inputs or fails loudly.

## Verify

Both need a GPU.

```bash
GPU="--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host
     --shm-size 16G -e HIP_VISIBLE_DEVICES=0"

# 1. Is the image assembled correctly?
docker run --rm $GPU --entrypoint verify-fused-router vllm-solar:v031-fp8

# 2. Does the kernel actually run?
docker run --rm $GPU --entrypoint bash \
  -v "$PWD/router_smoke.py:/tmp/rs.py:ro" \
  vllm-solar:v031-fp8 -lc 'python3 /tmp/rs.py'
```

Check (1) launches nothing. It asserts:

- the overlay landed, and every kwarg `vllm/_aiter_ops.py` forwards exists on
  `fused_moe_router`;
- `VLLM_USE_V2_MODEL_RUNNER` resolves to off (see §Run);
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

`VLLM_USE_V2_MODEL_RUNNER=0` is baked in too. v0.31.0 defaults to Model Runner
V2, which refuses to start with `SolarOpen2TemplateLogitsProcessor` (a V1
interface logits processor). Do not override it while passing
`--logits-processors`.

Shared-expert fusion stays opt-in
(`-e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`). It only applies when the
checkpoint quantizes the shared expert like the routed ones
(`Solar-Open2-250B-mxfp4-shared-expert`); on the other checkpoints Quark
excludes the shared expert and the fusion is declined per layer.

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
  vllm-solar:v031-fp8 \
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
  vllm-solar:v031-fp8 \
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
  vllm-solar:v031-fp8 \
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

`vllm-src/` and `aiter-overlay/` are generated by `fetch-sources.sh`. Neither
should be committed.

## How it works

**Engine (steps 1–2).** The branch is `v0.31.0` + 19 commits, and its only
non-Python file is `csrc/rocm/skinny_gemms.cu` (wvSplitK out-of-bounds fix,
`f40c0bee45`; stock v0.31.0 does not carry it). That one file is the entire
reason a compile happens. Step 1 configures the tree with CMake and builds the
`_rocm_C` target alone, since `skinny_gemms.cu` lands in that module only; the
result (~130 MB, gfx950) replaces the image's copy. The Python tree is then
copied over the installed package. The archive carries no `_version.py`, so
the image's copy stays and vLLM keeps reporting `0.31.0+rocm723`.

The FP8 router commit (`bd2603f7e7`) is Python only. It maps Quark FP8 PTPC to
the router in `rocm_aiter_moe.py`, has `oracle/fp8.py` offer
`AiterFusedRouterExperts` first when the flag is on, as the MXFP4 oracle does,
and adds `QuarkW8A8Fp8MoEMethod.apply_monolithic`.

**Router (step 3).** Injected as nine files rather than by reinstalling AITER.
This is safe because the branch sits on exactly the AITER revision the image
ships: `v0.1.23` (`50da036ac`), then the MXFP4 router (`6e84d85e2`), hidden
2048 (`25c12fea1`), the FP8 mode (`331970f9e`) and the sort-extent padding fix
(`9689185d3`). The five files the commits modify all md5-match the installed
copies at `v0.1.23`; the other four are new. So no AITER drift and no flydsl
bump.

Files land in two trees because AITER resolves kernel sources and headers from
`aiter_meta/` (a *sibling* of `aiter/`) but reads the build registry from the
package:

```
aiter/fused_moe.py, aiter/__init__.py,
aiter/ops/fused_moe_router.py,
aiter/jit/optCompilerConfig.json             -> package
csrc/kernels/fused_moe_router.cu,
csrc/pybind/fused_moe_router_pybind.cu,
csrc/include/fused_moe_router.h,
csrc/include/rocm_ops.hpp,
csrc/include/aiter_hip_common.h              -> aiter_meta
```

`module_fused_moe_router` compiles the pybind file and the kernel file. The two
shared headers only gain router declarations; modules the base image already
ships are not rebuilt.

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

## Three traps this Dockerfile works around

1. **`PYTORCH_ROCM_ARCH` cannot be an ARG here.** The base image sets it as an
   `ENV` (the full 9-arch list), and ENV beats a same-named ARG during
   expansion — the override is silently ignored and you get a 9-arch build that
   takes ~9x longer. The ARG is named `SOLAR_ROCM_ARCH` for this reason.
2. **`setup.py build_ext` is not used.** Since v0.31 `setup.py` pulls in
   setuptools-rust, and without `.git` it also needs a pinned
   `setuptools_scm` version. Building only the `_rocm_C` CMake target avoids
   both, and skips the extensions the image already ships.
3. **`triton_kernels` is untracked.** `git archive` does not export
   `vllm/third_party/triton_kernels`, so CMake configure would clone
   ROCm/triton. `TRITON_KERNELS_SRC_DIR` points it at the image's installed
   copy instead.

## Rebuilding the C++ extension later

The image keeps a full HIP toolchain (`hipcc`, `cmake`, `ninja`, `g++`) but not
the vLLM C++ sources — step 2 deletes `/vllm_src`. Mount them back:

```bash
docker run --rm -v "$PWD/vllm-src:/src:ro" --entrypoint bash \
  vllm-solar:v031-fp8 -lc '
    SP=/usr/local/lib/python3.12/dist-packages
    mkdir /b && cd /b
    PYTORCH_ROCM_ARCH=gfx950 \
    TRITON_KERNELS_SRC_DIR=$SP/vllm/third_party/triton_kernels \
      cmake /src -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DVLLM_TARGET_DEVICE=rocm -DVLLM_PYTHON_EXECUTABLE=$(which python3) \
        -DROCM_PATH=/opt/rocm
    cmake --build . --target _rocm_C -j $(nproc)'
```

Produces a ~130 MB gfx950 `_rocm_C.abi3.so` in `/b`, no GPU needed. Compiling
the `.cu` by hand with `hipcc` fails on `torch/all.h` unless you add
`torch/include/torch/csrc/api/include`; let CMake assemble the flags.

## Reproducibility

Every input is pinned: both sources by commit, the base image by digest, the
nine router files and four tuning CSVs md5-checked.

Compiled *outputs* are not bit-reproducible — the HIP compiler is
nondeterministic. Two builds of the 8 Sep image from identical sources produced
router kernels differing in 27752 of 2986616 bytes, both verified working.
Judge these artifacts by behaviour, not by hash.

## Status

Not built as an image yet. Each step was run outside Docker, in a container
of the same `v0.31.0` base image (same `/app/versions.txt`) with the same
AITER and engine revisions:

- `fetch-sources.sh` resolves both pins, and all nine overlay md5s and four
  tuning md5s match. The five modified overlay files' stock copies match
  `v0.1.23` byte for byte.
- Step 1 as written, on the exported `vllm-src/` with `TRITON_KERNELS_SRC_DIR`
  set to the wheel's copy: configure needs no network and `_rocm_C` builds in
  3 min 43 s (131 MB, gfx950).
- The step-4 checks pass against the overlaid AITER 0.1.23 and `d05acb0848`.
- `tuning_check.py` passes against AITER 0.1.23's stock schemas; the merged
  fmoe table is 52 rows. 0.1.23 reads the extra `gfx`/`xbf16`/`flat` columns.
- `build_kernel.py module_fused_moe_router module_gemm_a8w8` with
  `GPU_ARCHS=gfx950` builds the router in 34 s (7.2 MB, MXFP4 + FP8) and
  `module_gemm_a8w8` with exactly 379 tuned shapes in the lookup header.
- `verify.sh` passes on gfx950 against those modules, and fails as intended
  without `VLLM_USE_V2_MODEL_RUNNER=0`.
- `router_smoke.py` passes both shapes.
- Outside the image, the same engine and AITER serve the 35B FP8 (TP1, router
  on/off) and the 250B MXFP4 checkpoints (TP4 + EP, router on, with and
  without shared-expert fusion) with correct output.

Not verified:

- `docker build` itself, the in-image `vllm serve`, and any benchmark or
  accuracy run on this image.
- The 35B MXFP4 checkpoint on v0.31.0.
- The wvSplitK fix is compiled in, but the fault it repairs was not
  reproduced.

Known tuning gap: with shared-expert fusion under expert parallel, v0.31.0's
250B MoE runs at 81 experts / topk 10 per rank, which the MXFP4 table (81 /
topk 8) does not cover, so AITER uses its default heuristics there.
