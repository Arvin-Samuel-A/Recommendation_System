#!/usr/bin/env bash
cd /root/early_stop/gc
/root/run13/venv/bin/python /root/early_stop/gc_analysis.py \
 "Beauty init s42=beauty_init_s42.json" "Beauty init s43=beauty_init_s43.json" "Beauty init s44=beauty_init_s44.json" \
 "Beauty noGN best s42=beauty_arv_mmoe_best_s42.json" "Beauty noGN best s43=beauty_arv_mmoe_best_s43.json" \
 "Beauty noGN best s44=beauty_arv_mmoe_best_s44.json" \
 "Beauty GN best s42=beauty_arv_mmoe_gradnorm_best_s42.json:0.80,1.94,0.26" \
 "Beauty GN best s43=beauty_arv_mmoe_gradnorm_best_s43.json:0.77,2.05,0.18" \
 "Beauty GN best s44=beauty_arv_mmoe_gradnorm_best_s44.json:0.81,1.96,0.23" \
 "ML init s43=ml_arv_mmoe_gradnorm_init_s43.json" \
 "ML init s43 GN weights=ml_arv_mmoe_gradnorm_init_s43.json:0.74,2.25,0.0003"
