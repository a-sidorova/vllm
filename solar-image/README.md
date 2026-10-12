# Solar-Open2 (MXFP4 + FP8) — v0.31.0 fused-router image

vLLM ROCm image for Solar-Open2 on MI355X:

> Baseline + eliminated element-wise small kernels + fused MoE-router kernel
> (`VLLM_ROCM_USE_AITER_FUSED_ROUTER=1`, ROCm/aiter#5150) for **MXFP4 and FP8
> per-token** experts + DFlash and fused KDA under speculative decoding +
> tuned bf16 GEMMs for the 250B at TP4.

Built from the current heads of the two branches:

- engine `asidorov/port/solar-v0.31.0` @ `c59be78a58` — the Solar-Open2 series
  and the fused router as in the previous image (`d05acb0848`), plus DFlash
  (`d0d51f2832`), FULL-decode cudagraph regression tests (`63cacdf1e8`), the
  fused KDA path under speculative decoding (`0404794356`) and
  `VLLM_ROCM_AITER_UNIFIED_ATTN_BACKEND` (`c59be78a58`);
- router `sshliapn/feature/fused_router_0.1.23` @ `9689185d3` — AITER v0.1.23
  (the AITER the base image ships) plus four router commits.

One image serves all three checkpoints; only the launch arguments differ. The
35B MXFP4 and 35B FP8 run TP1; the 250B MXFP4 runs TP4 with expert parallel,
per §2 of `SOLAR_250B_V025_UPDATE_08_09_2026.html`.

Self-contained: engine, router kernel and tuning are all in the image.

## Inputs

| | Value |
|---|---|
| Base image | `vllm/vllm-openai-rocm:v0.31.0`, pinned by digest `sha256:749f6f3f…` |
| Engine | `asidorov/port/solar-v0.31.0` @ `c59be78a58` (`v0.31.0` + 24 commits) |
| Engine remote | `github.com/UpstageAI/upstage-vllm-amd` (**private**) |
| Router | ROCm/aiter#5150 + hidden 2048 + FP8 per-token mode @ `9689185d3` (4 commits on `v0.1.23`, `50da036ac`) |
| Router remote | `github.com/sshlyapn/aiter`, branch `sshliapn/feature/fused_router_0.1.23` |
| Tuning | `tuning/bf16_tuned_gemms_250b_tp4.csv` (bf16 GEMMs only) |
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

- the engine is exported from `c59be78a58` in the checkout this directory lives
  in (`VLLM_GIT` to override). Do not `--depth 1` the clone; on a shallow clone
  it falls back to fetching that commit from the private remote;
- AITER is read from an `aiter` checkout next to `upstage-vllm-amd`
  (`AITER_GIT` to override). Without one, it fetches the router commit and
  `v0.1.23` by SHA from `github.com/sshlyapn/aiter`.

It pins both revisions by commit, checks that the router commits touch exactly
the nine overlay files and md5-checks them, and md5-checks the tuning CSV, so it
either reproduces the exact intended inputs or fails loudly. It also writes
`aiter-overlay/stock.md5`: what the five files the router *modifies* hash to at
`v0.1.23`. `docker build` checks the base image's installed copies against it
before overlaying (step 3), so a base image with any other AITER fails the
build rather than producing a mixed AITER.

To build from newer commits on either branch, update `VLLM_COMMIT` /
`AITER_COMMIT` and the overlay md5s in `fetch-sources.sh`. As long as the
router branch stays on `v0.1.23` and only touches those nine files, nothing
else changes; if it touches another file, `fetch-sources.sh` says which.

Outside Docker on a 256-core host: ~4 min for `_rocm_C` and ~35 s for the
router.

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
  `AITER_CONFIG_GEMM_BF16` to the baked-in table.

Check (2) launches the router at two decode shapes — MXFP4 250B (hidden 4096,
80 experts, topk 7) and FP8 35B (hidden 2048, 128 experts, topk 4) — and
asserts shape and finiteness. The first call should take seconds at most; ~35 s
means the router was not compiled into the image.

### Test suites

The image carries neither test tree, so mount them from the same commits the
image was built from (`aiter` at `9689185d3`, `upstage-vllm-amd` at
`c59be78a58`). `pytest` is in the base image; vLLM's `conftest.py` also needs
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
`HIP_VISIBLE_DEVICES=0,1,2,3`) is stale at `9689185d3`. It still uses the
pre-FP8 buffer names `o4`/`osc` (now `a1`/`a1s`) and calls `_deswizzle()`
without the `scale_n` argument, so it dies with `KeyError: 'o4'`. Patched for
those two changes, it passes all 12 suites.


## Runing

### Tuning

Baked into `/tuning`, with the env var preset — nothing to mount:

| Env var | File | Contents |
|---|---|---|
| `AITER_CONFIG_GEMM_BF16` | `bf16_tuned_gemms_250b_tp4.csv` | 504 rows, gfx950 / 256 CUs only: the bf16 GEMMs of the 250B at TP4. Same schema as AITER's stock `configs/bf16_tuned_gemm.csv`, which it replaces wholesale. Kernels: flydsl, hipBLASLt, torch, triton, asm, skinny. |

The table is read at runtime: point `AITER_CONFIG_GEMM_BF16` elsewhere to
override, no rebuild needed. Shapes outside it fall back to AITER's heuristics
— including every bf16 GEMM of the 35B checkpoints, since the stock table is
replaced, not extended. To keep both, give AITER a colon-separated list
(`-e AITER_CONFIG_GEMM_BF16=/tuning/bf16_tuned_gemms_250b_tp4.csv:/path/to/other.csv`);
it merges them on start.

`AITER_CONFIG_FMOE` and `AITER_CONFIG_GEMM_A8W8` are unset, so the MoE and
a8w8 GEMMs use AITER's stock tables. `AITER_CONFIG_FMOE` can be pointed at a
mounted table at runtime. `AITER_CONFIG_GEMM_A8W8` cannot: AITER compiles that
table into `module_gemm_a8w8.so` and never rebuilds an existing `.so` when the
variable changes. To change it, add the CSV under `tuning/` and rebuild that
module in step 5 (`build_kernel.py module_gemm_a8w8`).

The build fails if the bf16 table does not parse, differs in schema from
AITER's, has null cells, has duplicate index rows (which would make the lookup
arbitrary), names an unknown kernel library, or carries rows for any target
other than gfx950 / 256 CUs.

### Model runner

The image leaves `VLLM_USE_V2_MODEL_RUNNER` unset, so vLLM picks the runner:
Model Runner V2 for Solar-Open2, with or without DFlash2. V2 rejects the
V1-interface `SolarOpen2TemplateLogitsProcessor` at startup, so a launch that
passes `--logits-processors=...SolarOpen2TemplateLogitsProcessor` needs
`-e VLLM_USE_V2_MODEL_RUNNER=0`.

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
| `tuning/bf16_tuned_gemms_250b_tp4.csv` | AITER bf16 GEMM config, copied to `/tuning` | in build, step 4b |
| `tuning_check.py` | validates the CSV against AITER's schema | in build, step 4b |
| `build_kernel.py` | compiles the router into the image | in build, step 5 |
| `verify.sh` | baked in as `verify-fused-router` | after build, GPU |
| `router_smoke.py` | real kernel launches, MXFP4 and FP8 | after build, GPU |

`vllm-src/` and `aiter-overlay/` are generated by `fetch-sources.sh`. Neither
should be committed.

## How it works

**Engine (steps 1–2).** The branch is `v0.31.0` + 24 commits, and its only
non-Python file is `csrc/rocm/skinny_gemms.cu` (wvSplitK out-of-bounds fix,
`f40c0bee45`; stock v0.31.0 does not carry it). The commits since the previous
image (DFlash, fused KDA under spec decode, the unified-attention backend
switch) are Python and tests only. That one `.cu` file is the entire reason a
compile happens. Step 1 configures the tree with CMake and builds the
`_rocm_C` target alone, since `skinny_gemms.cu` lands in that module only; the
result (~130 MB, gfx950) replaces the image's copy. The Python tree is then
copied over the installed package. The archive carries no `_version.py`, so
the image's copy stays and vLLM keeps reporting `0.31.0+rocm723`.

**Router (step 3).** Injected as nine files rather than by reinstalling AITER.
This is safe because the branch sits on exactly the AITER revision the image
ships: `v0.1.23` (`50da036ac`), then the MXFP4 router (`6e84d85e2`), hidden
2048 (`25c12fea1`), the FP8 mode (`331970f9e`) and the sort-extent padding fix
(`9689185d3`). Five of the nine files exist at `v0.1.23` and are modified; the
build first checks that the image's copies are byte-identical to `v0.1.23`
(`stock.md5`), then overwrites them. The other four are new. So no AITER drift
and no flydsl bump.

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

**Kernel (step 5).** `module_fused_moe_router` is compiled into an image layer.
AITER would otherwise JIT it on the first decode batch (~35 s). That write
lands in the *container's* writable layer, not the image, so every fresh
container pays it again, inside its first request.

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
nine router files, the five stock files they replace, and the tuning CSV
md5-checked.

Compiled *outputs* are not bit-reproducible — the HIP compiler is
nondeterministic. Two builds of the 8 Sep image from identical sources produced
router kernels differing in 27752 of 2986616 bytes, both verified working.
Judge these artifacts by behaviour, not by hash.

## Status

Built as `vllm-solar:v031` on an MI355X host (gfx950), and tested on GPUs 0–3:

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
