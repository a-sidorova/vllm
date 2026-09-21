# Solar-Open2-250B (MXFP4) "New combined (8 Sep)" image.
#
# Base: stock vllm/vllm-openai-rocm:v0.25.1. AITER and flydsl are NOT replaced -
# the router is injected as a six-file overlay onto the AITER the image already
# ships (amd-aiter 0.1.16.post2 / flydsl 0.2.0, both untouched).
#
#   engine  asidorov/features/solar_model_optimizations_v3
#           8665505227ad4db9042ef55795e435a317ed8af6  (v0.25.1 + 20 commits)
#   router  ROCm/aiter#5150, sshlyapn/aiter
#           sshliapn/features/fused_router_v0.25_vllm_port @ 7fb161ea1
#
# Build:
#   ./fetch-sources.sh          # regenerates vllm-src/ and aiter-overlay/
#   docker build -t vllm-solar:0908-combined .

# Pinned by digest, not tag: a re-pushed v0.25.1 would otherwise change the
# image silently. This digest is v0.25.1 as of 2026-09-17 (b0f090658a8d).
FROM vllm/vllm-openai-rocm:v0.25.1@sha256:84459732ca98b40fe2f5338a3f050be6d522504e47a484a5180d58fb75956f86

ARG PYTHON_VERSION=3.12
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages

# MI355X only. A multi-arch .so takes ~9x longer to build; see §4 of
# AITER_FUSED_ROUTER_SETUP.md before redistributing this image.
#
# This must NOT be named PYTORCH_ROCM_ARCH: the base image sets that as an ENV
# (the full 9-arch list), and ENV wins over a same-named ARG during expansion,
# so the override would be silently ignored and you would get a 9-arch build.
ARG SOLAR_ROCM_ARCH=gfx950

# The source is exported with `git archive`, so there is no .git for
# setuptools_scm to read. v0.25.1 + 20 commits, hence 0.25.2.dev20.
ARG VLLM_VERSION=0.25.2.dev20+g8665505227

# ---------------------------------------------------------------------------
# 1. vLLM: rebuild _rocm_C for the wvSplitK out-of-bounds fix (a54bf82717,
#    csrc/rocm/skinny_gemms.cu). It is the only non-Python change on the
#    branch, and the only reason a compile is needed at all.
# ---------------------------------------------------------------------------
COPY vllm-src/ /vllm_src/
RUN set -eu; \
    cd /vllm_src; \
    PYTORCH_ROCM_ARCH=${SOLAR_ROCM_ARCH} MAX_JOBS=$(nproc) \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_VERSION} \
        python3 setup.py build_ext --inplace; \
    cp vllm/_rocm_C.abi3.so ${SITE_PACKAGES}/vllm/_rocm_C.abi3.so

# ---------------------------------------------------------------------------
# 2. vLLM Python drop-in. The source tree has no _rust_tool_parser.abi3.so
#    (no Rust toolchain in the image), so cp -r leaves the image's copy alone.
# ---------------------------------------------------------------------------
RUN set -eu; \
    cp -r /vllm_src/vllm/. ${SITE_PACKAGES}/vllm/; \
    find ${SITE_PACKAGES}/vllm -name '__pycache__' -type d -exec rm -rf {} +; \
    rm -rf /vllm_src

# ---------------------------------------------------------------------------
# 3. AITER fused router (aiter#5150), overlay-style. The branch sits on exactly
#    the revision the image ships (base 620287969) - fused_moe.py, __init__.py
#    and optCompilerConfig.json all md5-match the installed copies before the
#    overlay - so no AITER reinstall and no flydsl bump.
#
#    Kernels go to aiter_meta/, where AITER_CSRC_DIR points; the build registry
#    lives in the package tree. module_fused_moe_router JITs on first use.
# ---------------------------------------------------------------------------
COPY aiter-overlay/aiter/ ${SITE_PACKAGES}/aiter/
COPY aiter-overlay/csrc/kernels/fused_moe_router.cu \
     ${SITE_PACKAGES}/aiter_meta/csrc/kernels/
COPY aiter-overlay/csrc/py_itfs_cu/fused_moe_router_entry.cu \
     ${SITE_PACKAGES}/aiter_meta/csrc/py_itfs_cu/
RUN find ${SITE_PACKAGES}/aiter -name '__pycache__' -type d -exec rm -rf {} +

# ---------------------------------------------------------------------------
# 4. Fail the build if anything did not land.
#
#    Build-time checks only: `docker build` has no GPU, and importing
#    aiter.fused_moe calls get_gfx() -> rocminfo, which dies without one. The
#    import- and signature-level checks therefore live in verify.sh, which is
#    baked in and must be run once on a GPU host.
# ---------------------------------------------------------------------------
RUN set -eu; \
    test -f ${SITE_PACKAGES}/vllm/_rust_tool_parser.abi3.so; \
    test -f ${SITE_PACKAGES}/vllm/model_executor/models/solar_open2.py; \
    test -f ${SITE_PACKAGES}/aiter/ops/fused_moe_router.py; \
    test -f ${SITE_PACKAGES}/aiter_meta/csrc/kernels/fused_moe_router.cu; \
    test -f ${SITE_PACKAGES}/aiter_meta/csrc/py_itfs_cu/fused_moe_router_entry.cu; \
    python3 -c "import json,sys; d=json.load(open('${SITE_PACKAGES}/aiter/jit/optCompilerConfig.json')); sys.exit(0 if 'module_fused_moe_router' in d else 1)"; \
    python3 -c "import importlib.metadata as m; assert m.version('flydsl')=='0.2.0', m.version('flydsl')"; \
    python3 -c "import importlib.metadata as m; assert m.version('amd-aiter')=='0.1.16.post2', m.version('amd-aiter')"; \
    grep -q 'def fused_moe_router_config_supported' ${SITE_PACKAGES}/aiter/fused_moe.py; \
    grep -q 'shared_expert_weight' ${SITE_PACKAGES}/aiter/fused_moe.py; \
    grep -q 'AiterFusedRouterExperts' ${SITE_PACKAGES}/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py; \
    python3 -c "import vllm.model_executor.models.solar_open2, vllm.reasoning, vllm.tool_parsers"

COPY verify.sh /usr/local/bin/verify-fused-router
RUN chmod +x /usr/local/bin/verify-fused-router

# ---------------------------------------------------------------------------
# 4b. Tuned configs. The MXFP4 decode MoE one is the third of the three §2
#     changes; the bf16 GEMM one is what the §2 run command passes as
#     AITER_CONFIG_GEMM_BF16. Both env vars are set, so they apply by
#     default - point them elsewhere to override, no rebuild needed.
#
#     The fmoe file holds both shapes: 80 routed / topk 7 (§2) and the fused
#     81-expert / topk 8 (§4). They cannot collide - `expert` and `topk` are
#     both AITER index columns.
# ---------------------------------------------------------------------------
COPY tuning/fmoe_tuned_mxfp4_decode_solar.csv tuning/aiter_gemm_tuning.csv /tuning/
ENV AITER_CONFIG_FMOE=/tuning/fmoe_tuned_mxfp4_decode_solar.csv
ENV AITER_CONFIG_GEMM_BF16=/tuning/aiter_gemm_tuning.csv

# Both parse, carry the expected shapes, and have no duplicate index rows -
# a dup would make the lookup arbitrary. For fmoe the check is per partition,
# since the flydsl_fallback rows deliberately mirror the tuned index.
RUN python3 -c "\
import pandas as pd, sys; \
FIDX = ['gfx','cu_num','token','model_dim','inter_dim','expert','topk','act_type','dtype','q_dtype_a','q_dtype_w','q_type','use_g1u1','doweight_stage1']; \
d = pd.read_csv('/tuning/fmoe_tuned_mxfp4_decode_solar.csv'); \
assert set(FIDX) <= set(d.columns), 'fmoe: missing index columns'; \
shapes = set(map(tuple, d[['expert','topk']].drop_duplicates().values)); \
assert shapes == {(80,7),(81,8)}, shapes; \
tag = d['_tag'].fillna(''); \
[sys.exit(f'fmoe: duplicate index rows in {n}') for n, s in (('tuned', d[tag=='']), ('fallback', d[tag=='flydsl_fallback'])) if s.duplicated(subset=FIDX).any()]; \
print(f'fmoe tuning ok: {len(d)} rows, shapes {sorted(shapes)}'); \
GIDX = ['gfx','cu_num','M','N','K','bias','dtype','outdtype','scaleAB','bpreshuffle']; \
g = pd.read_csv('/tuning/aiter_gemm_tuning.csv'); \
stock = pd.read_csv('${SITE_PACKAGES}/aiter/configs/bf16_tuned_gemm.csv'); \
assert list(g.columns) == list(stock.columns), 'gemm: schema differs from aiter/configs'; \
assert not g.isna().any().any(), 'gemm: null cells'; \
assert not g.duplicated(subset=GIDX).any(), 'gemm: duplicate index rows'; \
assert ('gfx950' in set(g.gfx)) and (4096 in set(g.K)), 'gemm: no gfx950 K=4096 rows'; \
print(f'gemm tuning ok: {len(g)} rows, {g[[\"N\",\"K\"]].drop_duplicates().shape[0]} N/K pairs')"

# The router only engages when both are set; the oracle gates on the second.
ENV VLLM_ROCM_USE_AITER=1
ENV VLLM_ROCM_USE_AITER_FUSED_ROUTER=1

RUN printf '%s\n' \
    'engine: asidorov/features/solar_model_optimizations_v3 @ 8665505227' \
    'router: ROCm/aiter#5150 overlay on amd-aiter 0.1.16.post2 (flydsl 0.2.0)' \
    'PYTORCH_ROCM_ARCH: gfx950' \
    'AITER_CONFIG_FMOE: /tuning/fmoe_tuned_mxfp4_decode_solar.csv (80+81 expert)' \
    'AITER_CONFIG_GEMM_BF16: /tuning/aiter_gemm_tuning.csv' \
    >> /app/versions.txt

# ---------------------------------------------------------------------------
# 5. Compile the router kernel into the image.
#
#    Otherwise it JITs on the first decode batch (~34 s) inside the first
#    request - and since that write lands in the container's writable layer,
#    not the image, every fresh container pays it again.
#
#    Compiling needs no GPU: only the build runs here, never the kernel. See
#    build_kernel.py for the two device probes that have to be stubbed.
# ---------------------------------------------------------------------------
COPY build_kernel.py /tmp/build_kernel.py
RUN set -eu; \
    GPU_ARCHS=${SOLAR_ROCM_ARCH} python3 /tmp/build_kernel.py; \
    test -f ${SITE_PACKAGES}/aiter/jit/module_fused_moe_router.so; \
    rm -f /tmp/build_kernel.py; \
    rm -rf ${SITE_PACKAGES}/aiter/jit/build/module_fused_moe_router
