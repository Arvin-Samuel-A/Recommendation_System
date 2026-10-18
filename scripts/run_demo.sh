#!/usr/bin/env bash
# Run the recommendation demo (scripts/demo_recommend.py), e.g. for the faculty review.
#
# Works from the main checkout or from any git worktree of it: it always runs in the main
# checkout, where the processed data (data/processed_users10pct, data/raw/movie.csv) and
# the copied run12 checkpoints (outputs/aws_2026-10-17/) live. None of them is in git.
#
# Usage:
#   scripts/run_demo.sh                 # three prepared examples, pausing between them
#   scripts/run_demo.sh --user 42       # any demo_recommend.py arguments instead
#   scripts/run_demo.sh --help
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here" && cd "$(git rev-parse --git-common-dir)/.." && pwd)"
cd "$root"

ckpt="outputs/aws_2026-10-17/run12/outputs/checkpoints/ablations/by_fingerprint/17bcde2f65e8/t_base/best.pt"
for f in scripts/demo_recommend.py "$ckpt" data/processed_users10pct/splits.pkl data/raw/movie.csv; do
  if [ ! -e "$f" ]; then
    echo "run_demo.sh: missing $root/$f" >&2
    echo "  (the checkpoints and data are not in git; copy outputs/aws_2026-10-17/ and data/ here)" >&2
    exit 1
  fi
done

demo() { uv run python scripts/demo_recommend.py "$@"; }

if [ $# -gt 0 ]; then
  demo "$@"
  exit
fi

pause() {
  if [ -t 0 ]; then
    read -r -p "Press Enter for the next example... " _
  fi
}

echo "Example 1: a real user; ✓ marks movies they really liked later (hidden from the model)"
demo --user 13408 --compare
pause

echo "Example 2: a new viewer described by five animated films they liked"
demo --liked "Toy Story (1995)" "Finding Nemo" "Shrek (2001)" "Monsters, Inc." "Incredibles"
pause

echo "Example 3: a randomly chosen user"
demo --seed 7
