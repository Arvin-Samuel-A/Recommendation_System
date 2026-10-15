#!/usr/bin/env bash
# Fetch the flash-attention source (with submodules) and pack it for a machine without git.
#
# specs/ablation-run FR-3 / design D3. Runs on the Mac (or anywhere with git + network).
# FlashAttention-3 lives in the repository's hopper/ directory and needs the csrc/cutlass
# submodule, hence --recursive. The resolved commit is written to FA3_SOURCE_COMMIT inside
# the tarball and printed, so the H100 build is traceable to an exact revision.
#
# Usage: scripts/fetch_fa3_source.sh [REF] [OUT_DIR]
#   REF      branch or tag to clone (default: main). `git clone --branch` does not take a
#            bare commit SHA; pin a tag for reproducibility.
#   OUT_DIR  where flash-attention-src.tar.gz is written (default: dist)
# Environment: FA_REPO_URL overrides the repository URL.
set -euo pipefail

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

REF="${1:-main}"
OUT_DIR="${2:-dist}"
REPO_URL="${FA_REPO_URL:-https://github.com/Dao-AILab/flash-attention.git}"

command -v git >/dev/null 2>&1 || { echo "fetch_fa3_source: git is required" >&2; exit 2; }
command -v tar >/dev/null 2>&1 || { echo "fetch_fa3_source: tar is required" >&2; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[fetch_fa3_source $(date -u +%Y-%m-%dT%H:%M:%SZ)] cloning $REPO_URL at $REF (with submodules)"
git clone --depth 1 --branch "$REF" --recursive --shallow-submodules \
    "$REPO_URL" "$WORK/flash-attention"

COMMIT="$(git -C "$WORK/flash-attention" rev-parse HEAD)"
printf '%s\n' "$COMMIT" > "$WORK/flash-attention/FA3_SOURCE_COMMIT"
printf 'ref=%s\nrepo=%s\nfetched_at=%s\n' "$REF" "$REPO_URL" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WORK/flash-attention/FA3_SOURCE_COMMIT"

# The target machine has no git; the metadata only bloats the tarball.
find "$WORK/flash-attention" -name .git -prune -exec rm -rf {} +

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/flash-attention-src.tar.gz"
tar czf "$OUT" -C "$WORK" flash-attention

SIZE="$(du -h "$OUT" | cut -f1)"
echo "[fetch_fa3_source] wrote $OUT ($SIZE), flash-attention commit $COMMIT"
