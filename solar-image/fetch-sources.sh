#!/usr/bin/env bash
# Regenerate the two source trees the Dockerfile consumes, from git pins.
#
#   vllm-src/       engine branch, exported with `git archive`
#   aiter-overlay/  the six router files, exported from the public aiter branch
#
# Both are pinned by commit and checked against recorded md5s, so a re-run on
# another host produces byte-identical inputs or fails loudly.
#
#   ./fetch-sources.sh && docker build -t vllm-solar:1001-fp8 .
set -euo pipefail
cd "$(dirname "$0")"

# --- pins ------------------------------------------------------------------
VLLM_REMOTE=https://github.com/UpstageAI/upstage-vllm-amd
VLLM_REF=asidorov/features/solar_35b_fp8_model_optimizations
# The FP8 fused-router engine commit. The branch tip may carry later changes to
# this directory only; the image is built from this commit.
VLLM_COMMIT=d8cde836bdf5dd787ff97fe6a30f49468d8aa29d

AITER_REMOTE=https://github.com/sshlyapn/aiter.git
AITER_REF=asidorov/features/fused_router_fp8
AITER_COMMIT=adcfd7630f0aa07358d35aa2528f506ae411321c

# The overlay, and the md5 each file must have at $AITER_COMMIT. __init__.py and
# optCompilerConfig.json are unchanged from the MXFP4-only image (7fb161ea1).
OVERLAY_FILES=(
  "aiter/fused_moe.py:66abdfd775aadbb45f212ed3a31c0d08"
  "aiter/__init__.py:c67a8d22d27beed0d671e4bbee71e71c"
  "aiter/ops/fused_moe_router.py:33d9b4933fee9c92685778242759d3cd"
  "aiter/jit/optCompilerConfig.json:d104a47acd966c3c78b984ba488c9fa5"
  "csrc/kernels/fused_moe_router.cu:84d05cbeaf8cef5c00e64c60064c8933"
  "csrc/py_itfs_cu/fused_moe_router_entry.cu:3d476801335301c2d5c2eec8a69d4e1e"
)

# --- engine ----------------------------------------------------------------
# This directory ships inside the engine repo on $VLLM_REF, and $VLLM_COMMIT is
# an ancestor of it, so a normal clone already has the source -- no network.
# The fetch is only a fallback for a shallow clone, and needs read access to
# the private remote.
echo "==> vllm-src/  $VLLM_COMMIT"
VLLM_GIT=${VLLM_GIT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
if ! git -C "$VLLM_GIT" cat-file -e "$VLLM_COMMIT^{commit}" 2>/dev/null; then
    echo "    $VLLM_COMMIT not in $VLLM_GIT (shallow clone?); fetching it"
    # Fetch the commit itself, not the ref: on a shallow clone fetching the ref
    # just re-gets the tip and leaves the ancestor missing. --unshallow is
    # needed because a shallow repo rejects a fetch that deepens it otherwise.
    depth_arg=()
    [ "$(git -C "$VLLM_GIT" rev-parse --is-shallow-repository)" = true ] &&
        depth_arg=(--unshallow)
    git -C "$VLLM_GIT" fetch -q "${depth_arg[@]}" "$VLLM_REMOTE" "$VLLM_COMMIT" 2>/dev/null ||
        git -C "$VLLM_GIT" fetch -q "${depth_arg[@]}" "$VLLM_REMOTE" "$VLLM_REF" || true
    git -C "$VLLM_GIT" cat-file -e "$VLLM_COMMIT^{commit}" 2>/dev/null || {
        echo "could not obtain $VLLM_COMMIT from $VLLM_REMOTE." >&2
        echo "Re-clone $VLLM_REMOTE at $VLLM_REF without --depth, or point" >&2
        echo "VLLM_GIT at a full checkout that has the commit." >&2
        exit 1
    }
fi
rm -rf vllm-src && mkdir vllm-src
git -C "$VLLM_GIT" archive "$VLLM_COMMIT" | tar -x -C vllm-src

# --- router overlay --------------------------------------------------------
echo "==> aiter-overlay/  $AITER_COMMIT"
AITER_TMP=$(mktemp -d)
trap 'rm -rf "$AITER_TMP"' EXIT
git -C "$AITER_TMP" init -q
# Fetch the pinned commit, not the branch tip, so a later push to $AITER_REF
# cannot change the overlay. GitHub serves reachable commits by full SHA.
git -C "$AITER_TMP" fetch -q --depth 1 "$AITER_REMOTE" "$AITER_COMMIT" 2>/dev/null ||
    git -C "$AITER_TMP" fetch -q "$AITER_REMOTE" "$AITER_REF"
git -C "$AITER_TMP" cat-file -e "$AITER_COMMIT^{commit}" 2>/dev/null || {
    echo "$AITER_COMMIT is not on $AITER_REMOTE $AITER_REF" >&2
    exit 1
}

fail=0
for entry in "${OVERLAY_FILES[@]}"; do
    f=${entry%:*}; want=${entry#*:}
    mkdir -p "aiter-overlay/$(dirname "$f")"
    git -C "$AITER_TMP" show "$AITER_COMMIT:$f" > "aiter-overlay/$f"
    got=$(md5sum "aiter-overlay/$f" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        printf '    %-45s ok\n' "$f"
    else
        printf '    %-45s MD5 MISMATCH want=%s got=%s\n' "$f" "$want" "$got"
        fail=1
    fi
done
[ "$fail" -eq 0 ] || {
    echo "overlay at $AITER_COMMIT does not match the recorded md5s; update the pins" >&2
    exit 1
}

# --- tuning ----------------------------------------------------------------
# These ship with the repo rather than being fetched, but they are build inputs
# like any other, so pin them too.
echo "==> tuning/"
TUNING_FILES=(
  "fmoe_tuned_mxfp4_decode_solar.csv:f0fa6ab4172f19e20d6db66e459893e5"
  "fmoe_tuned_35b_fp8_tp1.csv:f6d1636d2c402a63cb1a9ac422728524"
  "aiter_gemm_tuning.csv:ead0bff9f7f86962bc469fa9e75c1846"
  "a8w8_tuned_gemm_35b_fp8.csv:99d452b34d5bc8ea4816edc5cb1ac04a"
)
for entry in "${TUNING_FILES[@]}"; do
    f=${entry%:*}; want=${entry#*:}
    [ -f "tuning/$f" ] || { echo "    tuning/$f MISSING" >&2; exit 1; }
    got=$(md5sum "tuning/$f" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        printf '    %-45s ok\n' "$f"
    else
        printf '    %-45s MD5 MISMATCH want=%s got=%s\n' "$f" "$want" "$got" >&2
        exit 1
    fi
done

echo
echo "Sources ready. Next:  docker build -t vllm-solar:1001-fp8 ."
