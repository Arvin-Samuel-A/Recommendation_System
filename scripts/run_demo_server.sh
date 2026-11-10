#!/usr/bin/env bash
# Run the web demo (demo/server.py + demo/index.html) at http://127.0.0.1:8000.
#
# Like run_demo.sh, it always runs in the main checkout, where the processed data and the
# copied run12 checkpoints (outputs/aws_2026-10-17/) live. None of them is in git.
#
# Usage:
#   scripts/run_demo_server.sh                       # http://127.0.0.1:8000
#   scripts/run_demo_server.sh --port 8080 --host 0.0.0.0 --device mps
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here" && cd "$(git rev-parse --git-common-dir)/.." && pwd)"
cd "$root"

ckpt="outputs/aws_2026-10-17/run12/outputs/checkpoints/ablations/by_fingerprint/17bcde2f65e8/t_base/best.pt"
for f in demo/server.py demo/index.html "$ckpt" data/processed_users10pct/splits.pkl data/raw/movie.csv; do
  if [ ! -e "$f" ]; then
    echo "run_demo_server.sh: missing $root/$f" >&2
    echo "  (the checkpoints and data are not in git; copy outputs/aws_2026-10-17/ and data/ here)" >&2
    exit 1
  fi
done

exec uv run python demo/server.py "$@"
