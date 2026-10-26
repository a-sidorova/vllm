# Solar-Open2 — vLLM v0.31.0 version optimized on MI355X

vLLM ROCm image for Solar-Open2 on MI355X:

> Baseline + eliminated element-wise small kernels + fused MoE-router kernel
> (`VLLM_ROCM_USE_AITER_FUSED_ROUTER=1`, ROCm/aiter#5150) for **MXFP4 and FP8
> per-token** experts + DFlash and fused KDA under speculative decoding +
> optional FlyDSL INT4 QuickReduce + gfx950 Gluon unified attention tuned for
> spec decode (ROCm/aiter#6336) + tuned GEMMs for the 250B at TP4 (bf16) and
> the 35B FP8 at TP1 (a8w8) + tuned MXFP4 fused MoE for the 250B at TP4 + EP.

Built from the current heads of the two branches:

- engine `asidorov/port/solar-v0.31.0` @ `60d7421943` — the Solar-Open2 series
  and the fused router as in the previous image (`d05acb0848`), plus DFlash
  (`d0d51f2832`), FULL-decode cudagraph regression tests (`63cacdf1e8`), the
  fused KDA path under speculative decoding (`0404794356`),
  `VLLM_ROCM_AITER_UNIFIED_ATTN_BACKEND` (`c59be78a58`), the QuickReduce+RMSNorm
  fusion fix for aiter >= 0.1.19.post1 (`81304a9419`) and the optional FlyDSL
  INT4 QuickReduce backend (`60d7421943`);
- aiter `sshliapn/feature/fused_router_0.1.23` @ `16dc176bf` — AITER v0.1.23
  (the AITER the base image ships) plus four router commits and
  ROCm/aiter#6336 (gfx950 Gluon unified attention for spec decode).

This is the engine + AITER that the
`vllm_0.31_fix_gluon_tuned_moe_flydsl_qr` (cfg10) benchmark campaign ran on:
the AITER the image assembles is byte-identical to `16dc176bf`.

One image serves all three checkpoints; only the launch arguments differ. The
35B MXFP4 and 35B FP8 run TP1; the 250B MXFP4 runs TP4 with expert parallel,
per §2 of `SOLAR_250B_V025_UPDATE_08_09_2026.html`.

Self-contained: engine, router kernel and tuning are all in the image.

## Inputs

| | Value |
|---|---|
| Base image | `vllm/vllm-openai-rocm:v0.31.0`, pinned by digest `sha256:749f6f3f…` |
| Engine | `asidorov/port/solar-v0.31.0` @ `60d7421943` (`v0.31.0` + 31 commits) |
| Engine remote | `github.com/UpstageAI/upstage-vllm-amd` (**private**) |
| AITER | ROCm/aiter#5150 + hidden 2048 + FP8 per-token mode + ROCm/aiter#6336 @ `16dc176bf` (5 commits on `v0.1.23`, `50da036ac`) |
| AITER remote | `github.com/sshlyapn/aiter`, branch `sshliapn/feature/fused_router_0.1.23` |
| Tuning | `tuning/bf16_tuned_gemms_250b_tp4.csv`, `tuning/a8w8_tuned_gemms_35b_fp8_tp1.csv`, `tuning/a4w4_tuned_fmoe_250b_tp4.csv` (see §Tuning) |
| Target arch | `gfx950` (MI355X only) |
| AITER / flydsl | `0.1.23` / `0.3.4.1` — not reinstalled |

## Build

Requires: Docker and a ROCm host. No GPU needed.

```bash
git clone -b asidorov/port/solar-v0.31.0 \
    https://github.com/UpstageAI/upstage-vllm-amd
git clone -b sshliapn/feature/fused_router_0.1.23 \
    https://github.com/sshlyapn/aiter          # optional, see below
cd upstage-vllm-amd/solar-image
./fetch-sources.sh
docker build -t vllm-solar:v031 .
```

`fetch-sources.sh` takes both sources from local checkouts when they have the
pinned commits, and only goes to the network otherwise:

- the engine is exported from `60d7421943` in the checkout this directory lives
  in (`VLLM_GIT` to override). Do not `--depth 1` the clone; on a shallow clone
  it falls back to fetching that commit from the private remote;
- AITER is read from an `aiter` checkout next to `upstage-vllm-amd`
  (`AITER_GIT` to override). Without one, it fetches `16dc176bf` and `v0.1.23`
  by SHA from `github.com/sshlyapn/aiter`.

It pins both revisions by commit, checks that the AITER commits touch exactly
the fifteen overlay files and md5-checks them, and md5-checks the tuning CSVs,
so it either reproduces the exact intended inputs or fails loudly. It also
writes `aiter-overlay/stock.md5`: what the eleven files the branch *modifies*
hash to at `v0.1.23`. `docker build` checks the base image's installed copies
against it before overlaying (step 3), so a base image with any other AITER
fails the build rather than producing a mixed AITER.

To build from newer commits on either branch, update `VLLM_COMMIT` /
`AITER_COMMIT` and the overlay md5s in `fetch-sources.sh`. As long as the
AITER branch stays on `v0.1.23` and only touches those fifteen files, nothing
else changes; if it touches another file, `fetch-sources.sh` says which.

Outside Docker on a 256-core host: ~4 min for `_rocm_C`, ~35 s for the
router and ~11.5 min for `module_gemm_a8w8` (379 shapes, 189 MB).

## Verify

All of these need a GPU. Pick free GPUs with `HIP_VISIBLE_DEVICES`.

```bash
GPU="--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host
     --shm-size 16G -e HIP_VISIBLE_DEVICES=0"

# 1. Is the image assembled correctly?
docker run --rm $GPU --entrypoint verify-fused-router vllm-solar:v031

# 2. Does the kernel actually run?
docker run --rm $GPU --entrypoint bash \
  -v "$PWD/router_smoke.py:/tmp/rs.py:ro" \
  vllm-solar:v031 -lc 'python3 /tmp/rs.py'
```

Check (1) launches nothing. It asserts:

- the overlay landed, and every kwarg `vllm/_aiter_ops.py` forwards exists on
  `fused_moe_router`;
- `VLLM_USE_V2_MODEL_RUNNER` is not forced off, so vLLM can pick Model Runner
  V2 (see §Model runner);
- the router accepts MXFP4 and FP8 per-token at hidden 2048/4096, with and
  without a fused shared expert, and refuses hidden 1024 and MXFP8;
- the FP8 engine path is present: the FP8 oracle offers
  `AiterFusedRouterExperts`, and `QuarkW8A8Fp8MoEMethod` has `apply_monolithic`;
- `module_fused_moe_router.so` is compiled in, and AITER resolves
  `AITER_CONFIG_GEMM_BF16` to the baked-in table;
- vLLM has the QuickReduce+RMSNorm handle-exchange fix and the FlyDSL INT4
  QuickReduce backend, and AITER ships `QuickAllReduceInt4`;
- the unified attention module vLLM's `ROCM_AITER_UNIFIED_ATTN` backend imports
  carries ROCm/aiter#6336;
- AITER resolves `AITER_CONFIG_GEMM_A8W8` to the baked-in 35B FP8 table, and
  `module_gemm_a8w8.so` is newer than that table, i.e. was built from it rather
  than being the base image's stock copy;
- `AITER_CONFIG_FMOE` is unset, the 250B MoE table is linked into
  `aiter/configs/model_configs/`, and every one of its 32 shapes resolves
  through AITER's merged MoE configs to its own tuned kernels.

Check (2) launches the router at two decode shapes — MXFP4 250B (hidden 4096,
80 experts, topk 7) and FP8 35B (hidden 2048, 128 experts, topk 4) — and
asserts shape and finiteness. The first call should take seconds at most; ~35 s
means the router was not compiled into the image.

### Test suites

The image carries neither test tree, so mount them from the same commits the
image was built from (`aiter` at `16dc176bf`, `upstage-vllm-amd` at
`60d7421943`). `pytest` is in the base image; vLLM's `conftest.py` also needs
`tblib` and `anthropic`, installed into the throwaway container below.

```bash
AITER=/path/to/aiter VLLM=/path/to/upstage-vllm-amd
GPU="--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host
     --shm-size 16G -e HIP_VISIBLE_DEVICES=0"

# AITER router: kernel vs the stock 4-kernel sequence, MXFP4 + FP8, shared
# expert, EP (876 cases, ~15 s)
docker run --rm $GPU --entrypoint bash \
  -v "$AITER/op_tests:/aiter_tests:ro" vllm-solar:v031 -lc '
    cd /tmp && python3 -m pytest -q -p no:cacheprovider \
      /aiter_tests/test_fused_moe_router.py'

# vLLM: the router wiring, then the tests added since d05acb0848 (~5 min)
docker run --rm $GPU --entrypoint bash \
  -v "$VLLM/tests:/vllm/tests:ro" \
  -v "$VLLM/pyproject.toml:/vllm/pyproject.toml:ro" vllm-solar:v031 -lc '
    pip install -q tblib==3.1.0 anthropic pytest-mock pytest-timeout
    cd /vllm/tests && python3 -m pytest -q -p no:cacheprovider \
      kernels/moe/test_rocm_aiter_fused_router.py \
      kernels/fla/test_fused_sigmoid_gating_delta_rule.py \
      kernels/mamba/test_kda_spec_decode_cudagraph.py \
      kernels/mamba/test_kda_spec_decode_wiring.py \
      kernels/mamba/test_solar_open2_fused_kda.py \
      models/kimi_k3/test_kda_metadata.py \
      models/test_solar_open2_specdec_prereqs.py \
      v1/attention/test_gdn_metadata_builder.py \
      v1/attention/test_rocm_attention_backends_selection.py \
      v1/spec_decode/test_dflash2.py \
      v1/worker/test_gpu_batch_ordering.py \
      v1/worker/test_mamba_hybrid_model_state.py'
```

Run the vLLM tests from `/vllm/tests`, not the repo root: from the root,
`import vllm` would pick up the source tree instead of the installed package
with the compiled extensions. `pyproject.toml` is mounted for the pytest
markers only.

One upstream test (#53695),
`test_rocm_attention_backends_selection.py::test_auto_selection_for_kv_connector[True-…ROCM_AITER_UNIFIED_ATTN]`,
fails under the image's baked-in `VLLM_ROCM_USE_AITER=1`, since auto-selection
then picks `ROCM_AITER_FA`. It assumes the default environment; with
`-e VLLM_ROCM_USE_AITER=0` the whole file passes.

The router's multi-GPU stress harness
(`op_tests/multigpu_tests/test_fused_moe_router_stress.py`, run with
`HIP_VISIBLE_DEVICES=0,1,2,3`) is stale at `16dc176bf` (unchanged since
`9689185d3`). It still uses the
pre-FP8 buffer names `o4`/`osc` (now `a1`/`a1s`) and calls `_deswizzle()`
without the `scale_n` argument, so it dies with `KeyError: 'o4'`. Patched for
those two changes, it passes all 12 suites.


## Runing

### Tuning

Baked into `/tuning` — nothing to mount. Named
`<kind>_tuned_gemms_<checkpoint>_tp<N>.csv`:

| File | Env var | Preset | Contents |
|---|---|---|---|
| `bf16_tuned_gemms_250b_tp4.csv` | `AITER_CONFIG_GEMM_BF16` | yes | 504 rows, 8 N/K pairs, gfx950 / 256 CUs only: the bf16 GEMMs of the 250B at TP4. Kernels: flydsl, hipBLASLt, torch, triton, asm, skinny. |
| `a8w8_tuned_gemms_35b_fp8_tp1.csv` | `AITER_CONFIG_GEMM_A8W8` | yes, compiled in | 379 rows, gfx950 / 256 CUs only: every FP8 a8w8 linear of the 35B FP8 at TP1 — N/K (2566, 2048), (3072, 128), (3072, 2048), (4096, 2048), (4876, 2048), (9496, 2048). |
| `a4w4_tuned_fmoe_250b_tp4.csv` | — (`model_configs/`) | merged with stock | 32 rows, gfx950 / 256 CUs only: the 250B MXFP4 (per_1x32) MoE at TP4 + EP, hidden 4096, intermediate 1280, tokens 1–32768. E=80 / topk 8 is the routed experts alone, E=81 / topk 10 the fused shared expert. All FlyDSL 2-stage (`flydsl_moe1_*` + `flydsl_moe2_layout_*`). |

The two GEMM tables share the schema of AITER's stock
`configs/bf16_tuned_gemm.csv` / `configs/a8w8_tuned_gemm.csv` and replace the
stock table wholesale. The MoE table does not replace anything; see below.

**bf16** is read at runtime: point `AITER_CONFIG_GEMM_BF16` elsewhere to
override, no rebuild needed. Shapes outside the table fall back to AITER's
heuristics, since the stock table is replaced, not extended. To combine
tables, give AITER a colon-separated list
(`-e AITER_CONFIG_GEMM_BF16=/tuning/a.csv:/path/to/b.csv`); it merges them on
start.

The 35B bf16 GEMMs are not tuned. Their shapes — N/K (2048, 3072), (2048,
2048), (2048, 1024), (128, 2048), (196608, 2048) — are in neither the 250B
table nor AITER's stock one, so they run on AITER's default kernels whichever
is selected.

**a8w8** is compiled into `module_gemm_a8w8.so` (step 5), and AITER never
rebuilds an existing `.so` when the variable changes, so it is preset and must
not be pointed elsewhere at runtime: the Python side would then read splitK
from a table the compiled kernels do not match. To change it, replace the CSV
under `tuning/` and rebuild. The 35B FP8 *requires* it: its KDA `in_proj`
(N=9496, not a multiple of 16) has no untuned fallback — without a tuned entry
AITER hands it to `torch._scaled_mm`, which rejects it, and the server fails to
start. MXFP4 checkpoints run no a8w8 GEMMs and are unaffected.

**fmoe** is read at runtime, through AITER's default resolution rather than an
env var: with `AITER_CONFIG_FMOE` unset, AITER merges `configs/tuned_fmoe.csv`
with every `configs/model_configs/*tuned_fmoe*.csv` at startup (into
`/tmp/aiter_configs/tuned_fmoe.csv`). The table is symlinked there from
`/tuning`, so the 250B gets its tuned kernels and the 35B MoE shapes keep
AITER's stock configs. It was tuned with the EP-aware tuner (320 global
experts, expert mask for EP rank 0 of 4) on the AITER/flydsl the image ships,
and gives a 1.18x geomean (1.04–1.37x) end-to-end MoE speedup over the
untuned path across its 32 shapes.

Setting `AITER_CONFIG_FMOE` replaces that whole merged set, this table
included. To add a table instead, list both:
`-e AITER_CONFIG_FMOE=/tuning/a4w4_tuned_fmoe_250b_tp4.csv:/path/to/b.csv`
(and `aiter/configs/tuned_fmoe.csv` too, to keep the stock rows). The FlyDSL
kernels are not precompiled into the image; FlyDSL compiles them on first use.

The build fails if any table does not parse, has null cells, has duplicate
index rows (which would make the lookup arbitrary), or carries rows for any
target other than gfx950 / 256 CUs (for a8w8 the codegen would silently drop
those rows); if a GEMM table differs in schema from AITER's, or the MoE table
lacks a column of AITER's; if the bf16 table names an unknown kernel library;
if the a8w8 table has a weight dtype other than fp8; if the MoE table has
non-MXFP4 weights or non-FlyDSL kernels, or shares a shape key with any stock
MoE table (AITER's merge would raise on it and rewrite the tables, and the
lookup would pick the stock row); and if the number of shapes compiled into
`module_gemm_a8w8` differs from the a8w8 table's row count.

### Model runner

The image leaves `VLLM_USE_V2_MODEL_RUNNER` unset, so vLLM picks the runner:
Model Runner V2 for Solar-Open2, with or without DFlash2. V2 rejects the
V1-interface `SolarOpen2TemplateLogitsProcessor` at startup, so a launch that
passes `--logits-processors=...SolarOpen2TemplateLogitsProcessor` needs
`-e VLLM_USE_V2_MODEL_RUNNER=0`.

### QuickReduce

TP all-reduces use vLLM's QuickReduce when
`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` is set (cfg10 uses `INT4`). The
AllReduce+RMSNorm fusion on that path now initializes against AITER 0.1.23's
IPC handle API; before `81304a9419` it failed silently and 256–8192-token
all-reduces fell back to bf16 CustomAllreduce.

`-e VLLM_ROCM_QUICK_REDUCE_FLYDSL=1` (default off, INT4 only) swaps the HIP
INT4 kernel for AITER's FlyDSL `QuickAllReduceInt4`, which ships in stock
AITER 0.1.23. It JIT-compiles at startup, while the ranks are in lockstep.
Confirm it in the startup log:

```
Custom quick allreduce: bf16 INT4 uses the AITER FlyDSL kernel (grids=...)
Fused AllReduce+RMSNorm: FlyDSL INT4 QuickReduce + AITER add_rmsnorm.
```

### Unified attention

With `--attention-backend=ROCM_AITER_UNIFIED_ATTN` and
`VLLM_ROCM_AITER_UNIFIED_ATTN_BACKEND=auto`, AITER picks its gfx950 Gluon
kernel. ROCm/aiter#6336 gives spec-decode verify batches (`max_seqlen_q` =
K+1, not pure decode) the split grid; the image carries it.

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
| `fetch-sources.sh` | writes `vllm-src/` + `aiter-overlay/` from the pinned commits | before build |
| `.dockerignore` | keeps `.git` and `__pycache__` out of the context | `docker build` |
| `tuning/bf16_tuned_gemms_250b_tp4.csv` | AITER bf16 GEMM config, 250B TP4, copied to `/tuning` | in build, step 4b |
| `tuning/a8w8_tuned_gemms_35b_fp8_tp1.csv` | AITER a8w8 GEMM config, 35B FP8 TP1, copied to `/tuning` | in build, steps 4b and 5 |
| `tuning/a4w4_tuned_fmoe_250b_tp4.csv` | AITER MXFP4 fused-MoE config, 250B TP4 + EP, copied to `/tuning` and linked into `aiter/configs/model_configs/` | in build, step 4b |
| `tuning_check.py` | validates the CSVs against AITER's schemas | in build, step 4b |
| `build_kernel.py` | compiles the router and the a8w8 GEMM into the image | in build, step 5 |
| `verify.sh` | baked in as `verify-fused-router` | after build, GPU |
| `router_smoke.py` | real kernel launches, MXFP4 and FP8 | after build, GPU |

`vllm-src/` and `aiter-overlay/` are generated by `fetch-sources.sh`. Neither
should be committed.

## How it works

**Engine (steps 1–2).** The branch is `v0.31.0` + 31 commits, and its only
non-Python file is `csrc/rocm/skinny_gemms.cu` (wvSplitK out-of-bounds fix,
`f40c0bee45`; stock v0.31.0 does not carry it). The commits since the previous
image (DFlash, fused KDA under spec decode, the unified-attention backend
switch, the QuickReduce fixes, MoRIIO) are Python and tests only. That one `.cu` file is the entire reason a
compile happens. Step 1 configures the tree with CMake and builds the
`_rocm_C` target alone, since `skinny_gemms.cu` lands in that module only; the
result (~130 MB, gfx950) replaces the image's copy. The Python tree is then
copied over the installed package. The archive carries no `_version.py`, so
the image's copy stays and vLLM keeps reporting `0.31.0+rocm723`.

**AITER (step 3).** Injected as fifteen files rather than by reinstalling
AITER. This is safe because the branch sits on exactly the AITER revision the
image ships: `v0.1.23` (`50da036ac`), then the MXFP4 router (`6e84d85e2`),
hidden 2048 (`25c12fea1`), the FP8 mode (`331970f9e`), the sort-extent padding
fix (`9689185d3`) and the gfx950 Gluon unified attention change
(ROCm/aiter#6336, `16dc176bf`, Python and JSON only). Eleven of the fifteen
files exist at `v0.1.23` and are modified; the build first checks that the
image's copies are byte-identical to `v0.1.23` (`stock.md5`), then overwrites
them. The other four are new. So no AITER drift and no flydsl bump.

Files land in two trees because AITER resolves kernel sources and headers from
`aiter_meta/` (a *sibling* of `aiter/`) but reads the build registry from the
package:

```
aiter/fused_moe.py, aiter/__init__.py,
aiter/ops/fused_moe_router.py,
aiter/jit/optCompilerConfig.json,
aiter/ops/triton/attention/unified_attention.py,
aiter/ops/triton/_triton_kernels/attention/unified_attention.py,
aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py,
aiter/ops/triton/configs/{gfx942/triton,gfx950/triton,gfx950/gluon}/
    attention/unified_attention/DEFAULT.json -> package
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

**Kernels (step 5).** `module_fused_moe_router` is compiled into an image layer.
AITER would otherwise JIT it on the first decode batch (~35 s). That write
lands in the *container's* writable layer, not the image, so every fresh
container pays it again, inside its first request.

`module_gemm_a8w8` is rebuilt in the same step, for a different reason: the
base image ships it prebuilt from AITER's stock a8w8 table, and the codegen
(`gen_instances.py --tune_file $AITER_CONFIG_GEMM_A8W8`) only runs when the
`.so` is missing. The build keeps only rows for the build target (`GPU_ARCHS`
→ gfx950 / 256 CUs), so it counts the `{"gfx950", 256, …}` entries in the
generated `gemm_a8w8_lookup.h` and fails unless all 379 rows made it in.

No GPU is required because only the compile runs, never a kernel. Two device
probes on the `import aiter` path are stubbed in `build_kernel.py`:
`chip_info._detect_native()` shells out to `rocminfo` (pinned via `GPU_ARCHS`),
and triton's `arch_info` probes the active driver then falls back to `jax`,
which is not installed. These stubs track AITER's current import chain; a new
import-time device probe would break the build loudly, which is the right
failure mode.

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
  vllm-solar:v031 -lc '
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
fifteen AITER overlay files, the eleven stock files they replace, and the three
tuning CSVs md5-checked.

Compiled *outputs* are not bit-reproducible — the HIP compiler is
nondeterministic. Two builds of the 8 Sep image from identical sources produced
router kernels differing in 27752 of 2986616 bytes, both verified working.
Judge these artifacts by behaviour, not by hash.

## Status

**Current pins (`60d7421943` / `16dc176bf`)** have not been through
`docker build` yet (no docker on the editing host). Each build step was run
outside Docker on an MI355X, inside the same `vllm-openai-rocm:v0.31.0` base,
against a site-packages assembled the way the Dockerfile assembles it:

- `fetch-sources.sh` passes from the local `aiter` checkout and fetched from
  `github.com/sshlyapn/aiter`; the two give a byte-identical `aiter-overlay/`.
- Step 1: `_rocm_C` builds from `vllm-src/` (131 MB, gfx950 only).
- Step 3: all eleven `stock.md5` entries match both git `v0.1.23` and the
  sha256s in the base image's `amd_aiter-0.1.23.dist-info/RECORD`. Stock
  `v0.1.23` + overlay is byte-identical to `16dc176bf` (2662 files).
- Steps 4 and 4b: every check passes, cut verbatim from the Dockerfile. The new
  QuickReduce and `_gfx950_split_grid` checks fail on the previous pins.
- Step 5: `module_fused_moe_router` in 37 s (7.2 MB), `module_gemm_a8w8` in
  11.3 min with all 379 tuned shapes in the lookup header.
- `verify-fused-router` OK and `router_smoke.py` OK on GPU 0.

Previous pins (`c59be78a58` / `9689185d3`): built as `vllm-solar:v031` on an
MI355X host (gfx950), and tested on GPUs 0–3:

- `fetch-sources.sh` passes both ways: from the local `aiter` checkout, and
  fetched from `github.com/sshlyapn/aiter` with no checkout. The two give a
  byte-identical `aiter-overlay/`.
- `docker build` passes from scratch: `_rocm_C` is rebuilt, the base image's
  five stock AITER files match `v0.1.23`, all step-4 checks pass, the bf16
  table validates (504 rows, 8 N/K pairs), and the router compiles without a
  GPU in 36 s (7.2 MB).
- `verify-fused-router`: OK. `router_smoke.py`: both shapes OK, with a 0.1 s
  first call, so nothing is JIT-compiled.
- `op_tests/test_fused_moe_router.py`: 876 passed.
- Router stress harness on 4 GPUs (with the two test fixes above): 12/12
  suites, including CUDA graphs, multi-stream and multi-device.
- vLLM, `kernels/moe/test_rocm_aiter_fused_router.py` plus every test file
  touched since `d05acb0848`: 460 passed, 13 skipped (CUDA-only), 1 failed.
  The failure is the upstream env-dependent test described above.

Not verified: `vllm serve` of any checkpoint in this image, benchmarks, and
accuracy.

The MoE table was added after that build and has not been through
`docker build` yet. Its step-4b check (`tuning_check.py`) and the new
`verify-fused-router` lookup were run against the same AITER 0.1.23 + router
overlay outside Docker: the merge with all 43 stock MoE tables raises no
duplicate, and all 32 shapes resolve to their tuned kernels on an MI355X. The
table itself was used by the 250B shared-expert DFlash2 runs on vLLM 0.31.
