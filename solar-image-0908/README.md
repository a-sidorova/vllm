# Solar-Open2 (MXFP4) — "New combined (8 Sep)" image

vLLM ROCm image for Solar-Open2 MXFP4 on MI355X:

> Baseline + eliminated element-wise small kernels + fused MoE-router kernel
> (`VLLM_ROCM_USE_AITER_FUSED_ROUTER=1`, ROCm/aiter#5150) + tuned decode MXFP4
> MoE.

One image serves both checkpoints; only the launch arguments differ. The 35B
runs TP1 and is the subject of
`SOLAR_35B_MXFP4_TP1_MI355X_FUSED_ROUTER.html`; the 250B runs TP4 with expert
parallel, per §2 of `SOLAR_250B_V025_UPDATE_08_09_2026.html`.

Self-contained: engine, router kernel and tuning are all in the image.

## Inputs

| | Value |
|---|---|
| Base image | `vllm/vllm-openai-rocm:v0.25.1`, pinned by digest `sha256:84459732…` |
| Engine | `sshliapn/features/solar_35b_model_optimizations_baseline` @ `8665505227` (the branch tip's parent; the tip adds only this directory) |
| Engine remote | `github.com/UpstageAI/upstage-vllm-amd` (**private**) |
| Router | ROCm/aiter#5150 @ `7fb161ea1` (adds hidden 2048 on top of `25979546b`) |
| Router remote | `github.com/sshlyapn/aiter`, branch `sshliapn/features/fused_router_v0.25_vllm_port` |
| Target arch | `gfx950` (MI355X only) |
| AITER / flydsl | `0.1.16.post2` / `0.2.0` — both untouched |

## Build

Requires: Docker and a ROCm host. No GPU needed.

```bash
git clone -b sshliapn/features/solar_35b_model_optimizations_baseline \
    https://github.com/UpstageAI/upstage-vllm-amd
cd upstage-vllm-amd/solar-image-0908
./fetch-sources.sh
docker build -t vllm-solar:0908-combined .
```

Do not `--depth 1` the clone: `fetch-sources.sh` exports the engine from
`8665505227` in your own checkout, so a full clone needs no network at all. On
a shallow clone it falls back to fetching that commit from the remote.

Takes ~10 min, almost all of it `skinny_gemms.hip.o`.

`fetch-sources.sh` pins both revisions by commit and md5-checks all six router
files, so it either reproduces the exact intended inputs or fails loudly.

## Verify

Both need a GPU.

```bash
GPU="--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host
     --shm-size 16G -e HIP_VISIBLE_DEVICES=0"

# 1. Is the image assembled correctly?
docker run --rm $GPU --entrypoint verify-fused-router vllm-solar:0908-combined

# 2. Does the kernel actually run?
docker run --rm $GPU --entrypoint bash \
  -v "$PWD/router_smoke.py:/tmp/rs.py:ro" \
  vllm-solar:0908-combined -lc 'python3 /tmp/rs.py'
```

Check (1) launches nothing — it asserts the AITER/flydsl versions are
untouched, the overlay landed, every kwarg `vllm/_aiter_ops.py` forwards exists
on `fused_moe_router`, and the oracle will select the router. It catches a
botched overlay or signature drift that would otherwise surface as a
`TypeError` deep inside the first decode batch.

Check (2) launches the kernel at the Solar decode shape and asserts shape and
finiteness. Expect `first call: 0.0s` — a ~34 s first call means the kernel was
not compiled into the image.

## Run

`VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_FUSED_ROUTER=1` are baked in;
the router needs both.

### 35B, TP1 — the configuration the 35B report measures

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  -v /path/to/models:/models \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm-solar:0908-combined \
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

### 250B, TP4 + expert parallel

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
  vllm-solar:0908-combined \
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

Baked into `/tuning`, with both env vars preset — nothing to mount:

| Env var | File | Contents |
|---|---|---|
| `AITER_CONFIG_FMOE` | `fmoe_tuned_mxfp4_decode_solar.csv` | 38 rows. 80 routed experts / topk 7 (§2) at padded_M 64 and 128; 81 experts / topk 8 (fused shared expert) at padded_M 1–128, tuned + `flydsl_fallback`. |
| `AITER_CONFIG_GEMM_BF16` | `aiter_gemm_tuning.csv` | 1612 rows, 80 N/K pairs, gfx950 / gfx942 / gfx1250. Same schema as AITER's stock `configs/bf16_tuned_gemm.csv`, which it replaces wholesale. |

Point either env var elsewhere to override; no rebuild needed. The build fails
if a file does not parse, has null cells, or has duplicate index rows (which
would make the lookup arbitrary).

Outside those shapes AITER falls back to heuristics, which is the intended
behaviour — the 80-expert shape is measured at exactly two token counts.

### Confirming the router engaged

`AiterFusedRouterExperts` is a Python class, so it never appears in a rocprof
trace. The only signal is the startup log: selection is an `info_once` in
`oracle/mxfp4.py`, refusal a `debug_once` — use `VLLM_LOGGING_LEVEL=DEBUG` to
see the reason.

The router is capped at 128 tokens and requires `num_expert_group == 1` and
`topk_group == 1`. It is a decode optimization and will not fire on large
prefills.

## Files

| File | Role | Runs |
|---|---|---|
| `Dockerfile` | the build | `docker build` |
| `fetch-sources.sh` | fetches `vllm-src/` + `aiter-overlay/` by commit | before build |
| `.dockerignore` | keeps `.git` and `__pycache__` out of the context | `docker build` |
| `tuning/*.csv` | AITER fmoe + bf16 GEMM configs, copied to `/tuning` | in build, step 4b |
| `build_kernel.py` | compiles the router kernel into the image | in build, step 5 |
| `verify.sh` | baked in as `verify-fused-router` | after build, GPU |
| `router_smoke.py` | real kernel launch | after build, GPU |

`vllm-src/` (106 MB) and `aiter-overlay/` (280 KB) are generated by
`fetch-sources.sh`. Neither should be committed.

## How it works

**Engine (steps 1–2).** The branch is `v0.25.1` + 20 commits, and its only
non-Python file is `csrc/rocm/skinny_gemms.cu` (wvSplitK out-of-bounds fix).
That one file is the entire reason a compile happens. `build_ext` rebuilds all
33 HIP objects and 4 modules, but only `_rocm_C.abi3.so` is copied into
site-packages, since `skinny_gemms.cu` lands in that module alone.

**Router (step 3).** Injected as six files rather than by reinstalling AITER.
This is safe because the branch sits on exactly the AITER revision the image
ships (`7fb161ea1`, two commits on base `620287969`) — `fused_moe.py`,
`__init__.py` and `optCompilerConfig.json` all md5-match the installed copies
before the overlay. So no AITER drift and no flydsl bump. The alternative
branches are based on `v0.1.21`/`v0.1.22` and pin flydsl 0.3.2, which would
mean replacing both packages wholesale.

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

**Kernel (step 5).** AITER would otherwise JIT the router on the first decode
batch (~34 s), and that write lands in the *container's* writable layer, not
the image — so every fresh container pays it again, inside its first request.
Step 5 makes it a layer instead.

No GPU is required because only the compile runs, never the kernel. Two device
probes on the `import aiter` path are stubbed in `build_kernel.py`:
`chip_info._detect_native()` shells out to `rocminfo` (pinned via `GPU_ARCHS`),
and triton's `arch_info` probes the active driver then falls back to `jax`,
which is not installed. These stubs track AITER's current import chain; a new
import-time device probe would break the build loudly, which is the right
failure mode.

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
  vllm-solar:0908-combined -lc '
    cp -r /src /w && cd /w
    PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=$(nproc) \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.25.2.dev20+g8665505227 \
      python3 setup.py build_ext --inplace'
```

Produces a 91 MB gfx950 `_rocm_C.abi3.so`, no GPU needed. `build_rust` fails at
the end with `can't find Rust compiler` — harmless, it runs after the C++ work
and the image already ships `_rust_tool_parser.abi3.so`. Compiling the `.cu` by
hand with `hipcc` fails on `torch/all.h` unless you add
`torch/include/torch/csrc/api/include`; let `setup.py` assemble the flags.

## Reproducibility

Every input is pinned: both sources by commit, the base image by digest, the
six router files and two tuning CSVs md5-checked. Building from a clean tree
containing only the files above reproduces an image that passes both checks —
re-running `fetch-sources.sh` and rebuilding hits the Docker cache on all 24
steps, which is byte-equality of every input.

Compiled *outputs* are not bit-reproducible — the HIP compiler is
nondeterministic. Two builds from identical sources produced router kernels
differing in 27752 of 2986616 bytes, both verified working. Judge these
artifacts by behaviour, not by hash.

## Status

Verified: `verify-fused-router` passes; the kernel launches on gfx950 and
returns a finite `(64, 4096)` bf16 tensor.

Both tuning CSVs are baked in and resolve at runtime; the fmoe one was probed
in-image and hits the tuned rows at every shape it covers.

Not verified: no `vllm serve` run, no real checkpoint, no benchmark — the §2
+5.7% figure is **not** reproduced here. No numerical parity check against the
unfused path. The wvSplitK fix is compiled in but the fault it repairs was not
reproduced.
