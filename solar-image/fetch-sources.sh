#!/usr/bin/env bash
# Regenerate the two source trees the Dockerfile consumes, from git pins.
#
#   vllm-src/       engine branch, exported with `git archive`
#   aiter-overlay/  the nine router files, exported from the aiter router branch,
#                   plus stock.md5: what the five files the router modifies
#                   must hash to in the base image before they are replaced
#
# Both are pinned by commit and checked against recorded md5s, so a re-run on
# another host produces byte-identical inputs or fails loudly.
#
#   ./fetch-sources.sh && docker build -t vllm-solar:v031 .
#
# Sources are taken from local checkouts when they have the pinned commit
# (VLLM_GIT, default: the repo this directory lives in; AITER_GIT, default: an
# `aiter` checkout next to it), else fetched from the remotes.
set -euo pipefail
cd "$(dirname "$0")"

# --- pins ------------------------------------------------------------------
VLLM_REMOTE=https://github.com/UpstageAI/upstage-vllm-amd
VLLM_REF=asidorov/port/solar-v0.31.0
# The v0.31.0 port: Solar-Open2, the MXFP4 + FP8 fused router, DFlash, fused
# KDA under speculative decoding, VLLM_ROCM_AITER_UNIFIED_ATTN_BACKEND. The
# branch tip may carry later changes to this directory only; the image is built
# from this commit.
VLLM_COMMIT=c59be78a58fb2682a97beb0551b1b655bb1d0815

AITER_REMOTE=https://github.com/sshlyapn/aiter.git
AITER_REF=sshliapn/feature/fused_router_0.1.23
AITER_COMMIT=9689185d39545469e0f0ea83c4c342b8a94723b5
# v0.1.23, the AITER the base image ships (AITER_BRANCH in /app/versions.txt).
# The router branch is this plus four commits.
AITER_BASE=50da036acdedec2dd596f93188d6c615e2561672

# The overlay, and the md5 each file must have at $AITER_COMMIT: every file the
# router commits touch on top of v0.1.23, except tests and CI.
OVERLAY_FILES=(
  "aiter/fused_moe.py:9a894f77b047c2454fe110badbfc19da"
  "aiter/__init__.py:0d51cd0f9ee13174c19ba911c9252c2a"
  "aiter/ops/fused_moe_router.py:7e507b9e99ca4c3527b57a62af9ccec6"
  "aiter/jit/optCompilerConfig.json:3a2c1f80bb5f3832bd9944ce33e78024"
  "csrc/include/aiter_hip_common.h:44c35ddad906e4d50543d0ee2c32a819"
  "csrc/include/fused_moe_router.h:f13612d37206953b6698235d8e2fa9a9"
  "csrc/include/rocm_ops.hpp:fd1226385e80344a97a5770d68d42dc2"
  "csrc/kernels/fused_moe_router.cu:1d28617ff78d9c0e6e45704a459ad7e8"
  "csrc/pybind/fused_moe_router_pybind.cu:023dcc6c219d4ea9942edea0c6d6947b"
)

has_commit() { git -C "$1" cat-file -e "$2^{commit}" 2>/dev/null; }

# --- engine ----------------------------------------------------------------
# This directory ships inside the engine repo on $VLLM_REF, and $VLLM_COMMIT is
# an ancestor of it, so a normal clone already has the source -- no network.
# The fetch is only a fallback for a shallow clone, and needs read access to
# the private remote.
echo "==> vllm-src/  $VLLM_COMMIT"
VLLM_GIT=${VLLM_GIT:-$(git rev-parse --show-toplevel)}
if ! has_commit "$VLLM_GIT" "$VLLM_COMMIT"; then
    echo "    $VLLM_COMMIT not in $VLLM_GIT (shallow clone?); fetching it"
    # Fetch the commit itself, not the ref: on a shallow clone fetching the ref
    # just re-gets the tip and leaves the ancestor missing. --unshallow is
    # needed because a shallow repo rejects a fetch that deepens it otherwise.
    depth_arg=()
    [ "$(git -C "$VLLM_GIT" rev-parse --is-shallow-repository)" = true ] &&
        depth_arg=(--unshallow)
    git -C "$VLLM_GIT" fetch -q "${depth_arg[@]}" "$VLLM_REMOTE" "$VLLM_COMMIT" 2>/dev/null ||
        git -C "$VLLM_GIT" fetch -q "${depth_arg[@]}" "$VLLM_REMOTE" "$VLLM_REF" || true
    has_commit "$VLLM_GIT" "$VLLM_COMMIT" || {
        echo "could not obtain $VLLM_COMMIT from $VLLM_REMOTE." >&2
        echo "Re-clone $VLLM_REMOTE at $VLLM_REF without --depth, or point" >&2
        echo "VLLM_GIT at a full checkout that has the commit." >&2
        exit 1
    }
fi
rm -rf vllm-src && mkdir vllm-src
git -C "$VLLM_GIT" archive "$VLLM_COMMIT" | tar -x -C vllm-src

# --- router overlay --------------------------------------------------------
echo "==> aiter-overlay/  $AITER_COMMIT (on v0.1.23 $AITER_BASE)"
AITER_GIT=${AITER_GIT:-$VLLM_GIT/../aiter}
if [ -d "$AITER_GIT" ] && has_commit "$AITER_GIT" "$AITER_COMMIT" &&
        has_commit "$AITER_GIT" "$AITER_BASE"; then
    echo "    from local checkout $(cd "$AITER_GIT" && pwd)"
else
    echo "    fetching from $AITER_REMOTE"
    AITER_GIT=$(mktemp -d)
    trap 'rm -rf "$AITER_GIT"' EXIT
    git -C "$AITER_GIT" init -q
    # Fetch the pinned commit, not the branch tip, so a later push to
    # $AITER_REF cannot change the overlay. GitHub serves reachable commits by
    # full SHA. The depth has to reach $AITER_BASE for the ancestry check
    # below; the branch is only a few commits on top of it.
    git -C "$AITER_GIT" fetch -q --depth 64 "$AITER_REMOTE" "$AITER_COMMIT" 2>/dev/null ||
        git -C "$AITER_GIT" fetch -q "$AITER_REMOTE" "$AITER_REF"
    for c in "$AITER_COMMIT" "$AITER_BASE"; do
        has_commit "$AITER_GIT" "$c" || {
            echo "$c is not on $AITER_REMOTE $AITER_REF" >&2
            exit 1
        }
    done
fi

# The overlay is only safe on the exact AITER it was branched from: anything
# else the branch did not touch has to be byte-identical to the base image.
git -C "$AITER_GIT" merge-base --is-ancestor "$AITER_BASE" "$AITER_COMMIT" || {
    echo "$AITER_COMMIT is not based on v0.1.23 ($AITER_BASE)" >&2
    exit 1
}
want_files=$(printf '%s\n' "${OVERLAY_FILES[@]%:*}" | sort)
got_files=$(git -C "$AITER_GIT" diff --name-only "$AITER_BASE" "$AITER_COMMIT" -- aiter csrc | sort)
[ "$got_files" = "$want_files" ] || {
    echo "router commits touch a different file set than the overlay:" >&2
    diff <(echo "$want_files") <(echo "$got_files") >&2 || true
    exit 1
}

rm -rf aiter-overlay && mkdir aiter-overlay
fail=0
for entry in "${OVERLAY_FILES[@]}"; do
    f=${entry%:*}; want=${entry#*:}
    mkdir -p "aiter-overlay/$(dirname "$f")"
    git -C "$AITER_GIT" show "$AITER_COMMIT:$f" > "aiter-overlay/$f"
    got=$(md5sum "aiter-overlay/$f" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        printf '    %-45s ok\n' "$f"
    else
        printf '    %-45s MD5 MISMATCH want=%s got=%s\n' "$f" "$want" "$got"
        fail=1
    fi
    # Stock md5 at v0.1.23, under the path the file is installed at (csrc/
    # lives in aiter_meta/, a sibling of the package). New files have none.
    if git -C "$AITER_GIT" cat-file -e "$AITER_BASE:$f" 2>/dev/null; then
        dst=$f; [[ $f == csrc/* ]] && dst=aiter_meta/$f
        printf '%s  %s\n' "$(git -C "$AITER_GIT" show "$AITER_BASE:$f" | md5sum | cut -d' ' -f1)" \
            "$dst" >> aiter-overlay/stock.md5
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
  "bf16_tuned_gemms_250b_tp4.csv:0083abbe110254c422d49b3feadcd5ad"
  "a8w8_tuned_gemms_35b_fp8_tp1.csv:99d452b34d5bc8ea4816edc5cb1ac04a"
  "a4w4_tuned_fmoe_250b_tp4.csv:d934a6ac9e46821f3f9df573561162a6"
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
echo "Sources ready. Next:  docker build -t vllm-solar:v031 ."
