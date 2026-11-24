#!/usr/bin/env python3
"""early_stop_auto.py (patience 50) pointed at ~/beauty14_fixedw.

Its eval-only re-run goes through ~/beauty14_fixedw/launch/run_h100_gated.sh, which always
passes the fixed loss weights, so the re-run maps to the same fingerprint.
usage: early_stop_fixedw.py QUEUE SEED [SEED ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import early_stop as es  # noqa: E402

es.ROOT = "/root/beauty14_fixedw"
es.STOP_LOG = os.path.expanduser("~/early_stop/stopped_fixedw.jsonl")
import early_stop_auto as ea  # noqa: E402

ea.IDLE_POLLS = 240  # 4 h with no trainer (they may wait a while for GPU memory)
ea.main()
