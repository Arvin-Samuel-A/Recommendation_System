#!/bin/bash
# Per-user test metrics for every distinct Beauty model (3 seeds), on GPU 0.
export CUDA_VISIBLE_DEVICES=0
PY=$HOME/run13/venv/bin/python
cd $HOME/beauty14
LIST=$($PY $HOME/early_stop/beauty_arm_list.py)
echo "arms: $(echo $LIST | wc -w)"
$PY $HOME/early_stop/per_user_eval.py $HOME/early_stop/per_user_beauty_all $LIST
echo DONE
