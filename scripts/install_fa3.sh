#!/usr/bin/env bash
# Build and verify FlashAttention-3 on a Hopper machine (the shared H100), without git.
#
# specs/ablation-run FR-3 / design D3. Steps:
#   1. preflight (each failure is named and exits 2): python >= 3.10, torch imports, CUDA
#      available, compute capability 9.x, torch's CUDA major == nvcc's major, ninja and
#      packaging available, >= 20 GB free for the build directory. --dry-run stops here and
#      prints the build command it would run.
#   2. if `import flash_attn_interface` already works and --force is not given, skip to 5.
#   3. extract TARBALL (from scripts/fetch_fa3_source.sh) into the build directory.
#   4. build hopper/ against the INSTALLED torch (--no-build-isolation --no-deps: it can never
#      replace the machine's torch) with only the kernel variants this project needs.
#   5. run scripts/verify_attention_backend.py --backend flash_attention_3 --model t-base;
#      this script exits with its status.
#
# Exit status (review REV-006 -- each failure class has its own code):
#   0  FA3 built (or already installed) and verified
#   1  verification: FA3 runs but its output/gradients disagree with the SDPA reference
#   2  preflight or input failure (not Hopper, no CUDA, nvcc mismatch, no tarball, ...) or
#      verification: FA3 still unavailable after the build
#   3  extraction or build of hopper/ failed (see logs/fa3_build.log)
#
# The build takes long on a shared box: run it detached, e.g.
#   mkdir -p logs && setsid nohup bash scripts/install_fa3.sh dist/flash-attention-src.tar.gz \
#       > logs/fa3_install.log 2>&1 < /dev/null &
#
# Usage: scripts/install_fa3.sh TARBALL [--dry-run] [--force] [--max-jobs N] [--build-dir DIR]
#                                       [--log-dir DIR]
# Environment: PYTHON (default python3), FA3_KEEP_FP16=1 keeps the fp16 kernels.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
TARBALL=""
DRY_RUN=0
FORCE=0
MAX_JOBS_ARG=""
BUILD_DIR="${HOME}/fa3_build"
LOG_DIR="$ROOT/logs"
MIN_FREE_GB="${FA3_MIN_FREE_GB:-20}"  # env override for tests
EXIT_BUILD_FAILED=3

log() { echo "[install_fa3 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
fail() {  # fail <check name> <message>
    echo "[install_fa3] PREFLIGHT FAILED: $1 -- $2" >&2
    exit 2
}
build_failed() {  # build_failed <message>
    echo "[install_fa3] BUILD FAILED: $1 (exit $EXIT_BUILD_FAILED; see $LOG_DIR/fa3_build.log)" >&2
    exit "$EXIT_BUILD_FAILED"
}
usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --max-jobs) MAX_JOBS_ARG="${2:?--max-jobs needs a value}"; shift ;;
        --build-dir) BUILD_DIR="${2:?--build-dir needs a value}"; shift ;;
        --log-dir) LOG_DIR="${2:?--log-dir needs a value}"; shift ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 64 ;;
        *)
            if [ -z "$TARBALL" ]; then TARBALL="$1"; else
                echo "unexpected argument: $1" >&2; exit 64
            fi ;;
    esac
    shift
done
[ -n "$TARBALL" ] || { usage >&2; exit 64; }

# Kernel variants this project never uses. Kept: bf16, head dims 64/96/128 (head dim 80 at
# t-base rounds up to the 96 kernels), varlen and backward.
DISABLE_FLAGS=(FP8 PAGEDKV APPENDKV SOFTCAP LOCAL SPLIT SM80 HDIM192 HDIM256)
if [ "${FA3_KEEP_FP16:-0}" != "1" ]; then
    DISABLE_FLAGS+=(FP16)
fi

# ---------------------------------------------------------------- 1. preflight
log "preflight with PYTHON=$PY"
command -v "$PY" >/dev/null 2>&1 || fail "python" "'$PY' not found (set PYTHON=...)"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "python" "Python >= 3.10 required, found $("$PY" -V 2>&1)"
"$PY" -c 'import torch' 2>/dev/null \
    || fail "torch" "'import torch' failed for $PY"
TORCH_INFO="$("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
log "torch / CUDA: $TORCH_INFO"
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || fail "CUDA" "torch.cuda.is_available() is False -- FlashAttention-3 needs a Hopper CUDA GPU"
CAP="$("$PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
case "$CAP" in
    9.*) log "compute capability $CAP (Hopper)" ;;
    *) fail "Hopper" "compute capability $CAP is not 9.x; FlashAttention-3 runs on Hopper only (use --attention_backend sdpa)" ;;
esac
TORCH_CUDA_MAJOR="$("$PY" -c 'import torch; print((torch.version.cuda or "").split(".")[0])')"
NVCC=""
if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
elif [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    NVCC="${CUDA_HOME}/bin/nvcc"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC=/usr/local/cuda/bin/nvcc
fi
[ -n "$NVCC" ] || fail "nvcc" "nvcc not found on PATH, in \$CUDA_HOME/bin or /usr/local/cuda/bin"
NVCC_MAJOR="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -n1)"
[ "$NVCC_MAJOR" = "$TORCH_CUDA_MAJOR" ] \
    || fail "nvcc" "nvcc CUDA $NVCC_MAJOR does not match torch's CUDA $TORCH_CUDA_MAJOR ($NVCC)"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"
"$PY" -c 'import packaging' 2>/dev/null || fail "packaging" "'$PY -m pip install packaging'"
if ! command -v ninja >/dev/null 2>&1 && ! "$PY" -c 'import ninja' 2>/dev/null; then
    fail "ninja" "ninja not found ('$PY -m pip install ninja')"
fi
PROBE_DIR="$BUILD_DIR"
while [ ! -d "$PROBE_DIR" ]; do PROBE_DIR="$(dirname "$PROBE_DIR")"; done
FREE_GB="$(df -Pk "$PROBE_DIR" | awk 'NR==2 {print int($4 / 1048576)}')"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] \
    || fail "disk" "${FREE_GB} GB free under $PROBE_DIR, need >= ${MIN_FREE_GB} GB"

if [ -n "$MAX_JOBS_ARG" ]; then
    MAX_JOBS="$MAX_JOBS_ARG"
else
    NPROC="$(nproc 2>/dev/null || echo 2)"
    RAM_GB="$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 16)"
    MAX_JOBS=$(( NPROC / 2 ))
    [ $(( RAM_GB / 8 )) -lt "$MAX_JOBS" ] && MAX_JOBS=$(( RAM_GB / 8 ))
    [ "$MAX_JOBS" -ge 1 ] || MAX_JOBS=1
fi
export MAX_JOBS
for flag in "${DISABLE_FLAGS[@]}"; do
    export "FLASH_ATTENTION_DISABLE_${flag}=TRUE"
done
BUILD_CMD="nice -n 10 $PY -m pip install --no-build-isolation --no-deps -v ."
log "preflight passed (MAX_JOBS=$MAX_JOBS, disabled: ${DISABLE_FLAGS[*]})"

if [ "$DRY_RUN" = "1" ]; then
    log "dry run: would extract $TARBALL into $BUILD_DIR and run in flash-attention/hopper:"
    echo "  $BUILD_CMD"
    exit 0
fi

# ---------------------------------------------------------------- 2. already installed?
if [ "$FORCE" != "1" ] && "$PY" -c 'import flash_attn_interface' 2>/dev/null; then
    log "flash_attn_interface already imports; skipping the build (--force rebuilds)"
else
    # ------------------------------------------------------------ 3. extract
    [ -f "$TARBALL" ] || fail "tarball" "$TARBALL does not exist (scripts/fetch_fa3_source.sh)"
    mkdir -p "$BUILD_DIR"
    log "extracting $TARBALL into $BUILD_DIR"
    tar xzf "$TARBALL" -C "$BUILD_DIR" || build_failed "could not extract $TARBALL"
    SRC="$BUILD_DIR/flash-attention"
    [ -f "$SRC/FA3_SOURCE_COMMIT" ] && log "flash-attention commit: $(head -n1 "$SRC/FA3_SOURCE_COMMIT")"
    [ -f "$SRC/hopper/setup.py" ] || fail "source" "$SRC/hopper/setup.py not found"

    # A renamed flag would silently build every variant (hours longer); say so.
    for flag in "${DISABLE_FLAGS[@]}"; do
        grep -q "FLASH_ATTENTION_DISABLE_${flag}" "$SRC/hopper/setup.py" \
            || log "WARNING: FLASH_ATTENTION_DISABLE_${flag} does not occur in hopper/setup.py"
    done

    # ------------------------------------------------------------ 4. build
    mkdir -p "$LOG_DIR"
    log "building hopper/ (log: $LOG_DIR/fa3_build.log): $BUILD_CMD"
    if ! (cd "$SRC/hopper" && $BUILD_CMD) 2>&1 | tee "$LOG_DIR/fa3_build.log"; then
        build_failed "pip install of hopper/ failed"
    fi
    log "build finished"
fi

# ---------------------------------------------------------------- 5. verify
cd "$ROOT"
log "verifying: scripts/verify_attention_backend.py --backend flash_attention_3 --model t-base"
set +e
"$PY" scripts/verify_attention_backend.py --backend flash_attention_3 --model t-base \
    --report "$LOG_DIR/verify_fa3.json"
STATUS=$?
set -e
log "verification exit status $STATUS"
exit "$STATUS"
