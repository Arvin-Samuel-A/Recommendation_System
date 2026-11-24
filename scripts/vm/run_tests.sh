#!/bin/bash
# Run one checkout's test suite CPU-only: run_tests.sh DIR
cd "$1" || exit 1
export PYTHONPATH="$HOME/early_stop/testdeps:$PWD" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$HOME/run13/venv/bin/python" -m pytest -q -p no:cacheprovider -rfE tests > ../$(basename "$1").log 2>&1
