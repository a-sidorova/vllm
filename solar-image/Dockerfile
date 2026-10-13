# Solar-Open2 v0.31.0 image: the Solar-Open2 port onto vLLM v0.31.0 (fused MoE
# router for MXFP4 and FP8 per-token experts, DFlash, fused KDA under
# speculative decoding) plus the 250B TP4 bf16 GEMM tuning and the 35B FP8 TP1
# a8w8 GEMM tuning.
#
# Base: stock vllm/vllm-openai-rocm:v0.31.0. AITER and flydsl are NOT replaced -
# the router is injected as a nine-file overlay onto the AITER the image already
# ships (amd-aiter 0.1.23 / flydsl 0.3.4.1), and its module is compiled here.
#
#   engine  asidorov/port/solar-v0.31.0
#           c59be78a58fb2682a97beb0551b1b655bb1d0815  (v0.31.0 + 24 commits)
#   router  ROCm/aiter#5150 + hidden 2048 + FP8 per-token mode, sshlyapn/aiter
#           sshliapn/feature/fused_router_0.1.23 @ 9689185d3 (v0.1.23 + 4)
#
# One image serves the MXFP4 (35B, 250B) and FP8 (35B) checkpoints; only the
# launch arguments differ.
#
# Build:
#   ./fetch-sources.sh          # regenerates vllm-src/ and aiter-overlay/
#   docker build -t vllm-solar:v031 .

# Pinned by digest, not tag: a re-pushed v0.31.0 would otherwise change the
# image silently. This digest is v0.31.0 as of 2026-10-04.
FROM vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f

ARG PYTHON_VERSION=3.12
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages

# MI355X only. A multi-arch .so takes ~9x longer to build; see §4 of
# AITER_FUSED_ROUTER_SETUP.md before redistributing this image.
#
# This must NOT be named PYTORCH_ROCM_ARCH: the base image sets that as an ENV
# (the full 9-arch list), and ENV wins over a same-named ARG during expansion,
# so the override would be silently ignored and you would get a 9-arch build.
ARG SOLAR_ROCM_ARCH=gfx950

# ---------------------------------------------------------------------------
# 1. vLLM: rebuild _rocm_C for the wvSplitK out-of-bounds fix (f40c0bee45,
#    csrc/rocm/skinny_gemms.cu). It is the only non-Python change on the
#    branch, and the only reason a compile is needed at all.
#
#    Only the _rocm_C target is built, straight through CMake. `setup.py
#    build_ext` would build every extension and, since v0.31, also runs
#    setuptools-rust, which has no toolchain in the image. TRITON_KERNELS_SRC_DIR
#    points at the image's copy so configure does not clone ROCm/triton; the
#    sources are untracked in git, so `git archive` does not carry them.
# ---------------------------------------------------------------------------
COPY vllm-src/ /vllm_src/
RUN set -eu; \
    mkdir /tmp/rocmc_build; \
    cd /tmp/rocmc_build; \
    PYTORCH_ROCM_ARCH=${SOLAR_ROCM_ARCH} \
    TRITON_KERNELS_SRC_DIR=${SITE_PACKAGES}/vllm/third_party/triton_kernels \
        cmake /vllm_src -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DVLLM_TARGET_DEVICE=rocm \
            -DVLLM_PYTHON_EXECUTABLE=$(which python3) \
            -DROCM_PATH=/opt/rocm; \
    cmake --build . --target _rocm_C -j $(nproc); \
    cp _rocm_C.abi3.so ${SITE_PACKAGES}/vllm/_rocm_C.abi3.so; \
    cd /; \
    rm -rf /tmp/rocmc_build

# ---------------------------------------------------------------------------
# 2. vLLM Python drop-in. The source tree has no _rust_tool_parser.abi3.so
#    (no Rust toolchain in the image), so cp -r leaves the image's copy alone.
# ---------------------------------------------------------------------------
RUN set -eu; \
    cp -r /vllm_src/vllm/. ${SITE_PACKAGES}/vllm/; \
    find ${SITE_PACKAGES}/vllm -name '__pycache__' -type d -exec rm -rf {} +; \
    rm -rf /vllm_src

# ---------------------------------------------------------------------------
# 3. AITER fused router (aiter#5150 + hidden 2048 + FP8 per-token),
#    overlay-style. The branch is v0.1.23 (50da036ac), the revision the image
#    ships, plus four router commits, so no AITER reinstall and no flydsl bump.
#
#    That only holds while the five files the commits modify are still the
#    stock v0.1.23 copies, so check that first (stock.md5, written by
#    fetch-sources.sh from the base commit): a base image with a different
#    AITER fails here instead of getting a half-old, half-new AITER.
#
#    Sources and headers go to aiter_meta/csrc/, where AITER_CSRC_DIR points;
#    the build registry lives in the package tree. The router module is
#    compiled in step 5.
# ---------------------------------------------------------------------------
COPY aiter-overlay/stock.md5 /tmp/aiter-stock.md5
RUN set -eu; \
    cd ${SITE_PACKAGES}; \
    md5sum -c /tmp/aiter-stock.md5 || { \
        echo "installed AITER is not v0.1.23; the router overlay does not apply" >&2; \
        exit 1; }; \
    rm -f /tmp/aiter-stock.md5
COPY aiter-overlay/aiter/ ${SITE_PACKAGES}/aiter/
COPY aiter-overlay/csrc/ ${SITE_PACKAGES}/aiter_meta/csrc/
RUN find ${SITE_PACKAGES}/aiter -name '__pycache__' -type d -exec rm -rf {} +

# ---------------------------------------------------------------------------
# 4. Fail the build if anything did not land.
#
#    Build-time checks only: `docker build` has no GPU, and importing
#    aiter.fused_moe calls get_gfx() -> rocminfo, which dies without one. The
#    import- and signature-level checks therefore live in verify.sh, which is
#    baked in and must be run once on a GPU host.
#
#    The FP8 checks name what is new on each side: the router's FP8 entry in
#    AITER's fused-mode table, and on the engine the FP8 quant-method mapping,
#    the FP8 oracle offering the router, and Quark FP8's apply_monolithic (the
#    MXFP4 method already has one, so a plain grep would pass without it).
#    The last two greps pin the engine past d05acb0848: fused KDA under spec
#    decode and the unified-attention backend switch.
# ---------------------------------------------------------------------------
RUN set -eu; \
    test -f ${SITE_PACKAGES}/vllm/_rust_tool_parser.abi3.so; \
    test -f ${SITE_PACKAGES}/vllm/model_executor/models/solar_open2.py; \
    test -f ${SITE_PACKAGES}/aiter/ops/fused_moe_router.py; \
    test -f ${SITE_PACKAGES}/aiter_meta/csrc/kernels/fused_moe_router.cu; \
    test -f ${SITE_PACKAGES}/aiter_meta/csrc/pybind/fused_moe_router_pybind.cu; \
    test -f ${SITE_PACKAGES}/aiter_meta/csrc/include/fused_moe_router.h; \
    grep -q 'fused_moe_router' ${SITE_PACKAGES}/aiter_meta/csrc/include/rocm_ops.hpp; \
    python3 -c "import json,sys; d=json.load(open('${SITE_PACKAGES}/aiter/jit/optCompilerConfig.json')); sys.exit(0 if 'module_fused_moe_router' in d else 1)"; \
    python3 -c "import importlib.metadata as m; assert m.version('flydsl')=='0.3.4.1', m.version('flydsl')"; \
    python3 -c "import importlib.metadata as m; assert m.version('amd-aiter')=='0.1.23', m.version('amd-aiter')"; \
    grep -q 'def fused_moe_router_config_supported' ${SITE_PACKAGES}/aiter/fused_moe.py; \
    grep -q 'shared_expert_weight' ${SITE_PACKAGES}/aiter/fused_moe.py; \
    grep -qF '(QuantType.per_Token, dtypes.fp8): dtypes.fp8' ${SITE_PACKAGES}/aiter/fused_moe.py; \
    grep -q 'QuantType::per_Token' ${SITE_PACKAGES}/aiter_meta/csrc/kernels/fused_moe_router.cu; \
    grep -q 'AiterFusedRouterExperts' ${SITE_PACKAGES}/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py; \
    grep -q 'def _rocm_aiter_fused_router_quant_method' ${SITE_PACKAGES}/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py; \
    grep -q 'AiterFusedRouterExperts' ${SITE_PACKAGES}/vllm/model_executor/layers/fused_moe/oracle/fp8.py; \
    python3 -c "import ast,sys; t=ast.parse(open('${SITE_PACKAGES}/vllm/model_executor/layers/quantization/quark/quark_moe.py').read()); c=[n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='QuarkW8A8Fp8MoEMethod']; sys.exit(0 if c and any(isinstance(f,ast.FunctionDef) and f.name=='apply_monolithic' for f in c[0].body) else 'QuarkW8A8Fp8MoEMethod.apply_monolithic missing')"; \
    grep -q 'VLLM_KDA_SPEC_DECODE' ${SITE_PACKAGES}/vllm/envs.py; \
    grep -q 'VLLM_ROCM_AITER_UNIFIED_ATTN_BACKEND' ${SITE_PACKAGES}/vllm/envs.py; \
    python3 -c "import vllm.model_executor.models.solar_open2, vllm.reasoning, vllm.tool_parsers"

COPY verify.sh /usr/local/bin/verify-fused-router
RUN chmod +x /usr/local/bin/verify-fused-router

# ---------------------------------------------------------------------------
# 4b. Tuned GEMM configs.
#
#     bf16: the 250B TP4 table. Read at runtime, so pointing
#     AITER_CONFIG_GEMM_BF16 elsewhere needs no rebuild. It replaces AITER's
#     stock bf16 table wholesale. The 35B bf16 GEMMs are not tuned: neither
#     this table nor the stock one has their shapes, so they run on AITER's
#     default kernels either way.
#
#     a8w8: the 35B FP8 table (every a8w8 linear of that checkpoint, gfx950 /
#     256 CUs). It is compiled into module_gemm_a8w8.so in step 5, so the env
#     var is preset to it and overriding it means rebuilding that module (see
#     README). Only FP8 checkpoints run a8w8 GEMMs; the MXFP4 ones run none.
#
#     The MoE table is AITER's stock one: AITER_CONFIG_FMOE is left unset.
# ---------------------------------------------------------------------------
COPY tuning/bf16_tuned_gemms_250b_tp4.csv \
     tuning/a8w8_tuned_gemms_35b_fp8_tp1.csv /tuning/
COPY tuning_check.py /tmp/tuning_check.py
RUN set -eu; \
    python3 /tmp/tuning_check.py ${SITE_PACKAGES}; \
    rm -f /tmp/tuning_check.py
ENV AITER_CONFIG_GEMM_BF16=/tuning/bf16_tuned_gemms_250b_tp4.csv
ENV AITER_CONFIG_GEMM_A8W8=/tuning/a8w8_tuned_gemms_35b_fp8_tp1.csv

# The router only engages when both are set; the oracles gate on the second.
ENV VLLM_ROCM_USE_AITER=1
ENV VLLM_ROCM_USE_AITER_FUSED_ROUTER=1

RUN printf '%s\n' \
    'engine: asidorov/port/solar-v0.31.0 @ c59be78a58 (v0.31.0 + 24 commits)' \
    'router: ROCm/aiter#5150 + hidden 2048 + FP8 per-token (9689185d3) overlay on amd-aiter 0.1.23 (flydsl 0.3.4.1)' \
    'PYTORCH_ROCM_ARCH: gfx950' \
    'AITER_CONFIG_GEMM_BF16: /tuning/bf16_tuned_gemms_250b_tp4.csv (250B TP4)' \
    'AITER_CONFIG_GEMM_A8W8: /tuning/a8w8_tuned_gemms_35b_fp8_tp1.csv (compiled into module_gemm_a8w8.so)' \
    'AITER_CONFIG_FMOE: AITER stock' \
    'VLLM_USE_V2_MODEL_RUNNER: unset (vLLM picks Model Runner V2)' \
    >> /app/versions.txt

# ---------------------------------------------------------------------------
# 5. Compile into the image:
#
#    module_fused_moe_router - otherwise it JITs on the first decode batch
#    (~35 s) inside the first request, and since that write lands in the
#    container's writable layer, not the image, every fresh container pays it
#    again. One .so holds the MXFP4 and FP8 kernels.
#
#    module_gemm_a8w8 - the base image ships it prebuilt from AITER's stock
#    a8w8 table, and an existing .so is not rebuilt when
#    AITER_CONFIG_GEMM_A8W8 changes. Rebuilt here from the table above; the
#    lookup-header count proves every tuned row made it in, since the codegen
#    silently drops rows for any other (gfx, cu_num).
#
#    Compiling needs no GPU: only the build runs here, never a kernel. See
#    build_kernel.py for the two device probes that have to be stubbed.
# ---------------------------------------------------------------------------
COPY build_kernel.py /tmp/build_kernel.py
RUN set -eu; \
    GPU_ARCHS=${SOLAR_ROCM_ARCH} python3 /tmp/build_kernel.py \
        module_fused_moe_router module_gemm_a8w8; \
    test -f ${SITE_PACKAGES}/aiter/jit/module_fused_moe_router.so; \
    test -f ${SITE_PACKAGES}/aiter/jit/module_gemm_a8w8.so; \
    want=$(python3 -c "import pandas as pd; print(len(pd.read_csv('${AITER_CONFIG_GEMM_A8W8}')))"); \
    got=$(grep -c '{"gfx950", 256, ' ${SITE_PACKAGES}/aiter/jit/build/module_gemm_a8w8/blob/gemm_a8w8_lookup.h || true); \
    [ "$got" = "$want" ] || { echo "module_gemm_a8w8: $got tuned shapes compiled in, want $want" >&2; exit 1; }; \
    echo "module_gemm_a8w8: $got tuned shapes compiled in"; \
    rm -f /tmp/build_kernel.py; \
    rm -rf ${SITE_PACKAGES}/aiter/jit/build/module_fused_moe_router \
           ${SITE_PACKAGES}/aiter/jit/build/module_gemm_a8w8
