#!/usr/bin/env bash
# MovieLens (seed 43) through training: GradNorm run with its logged weights and with equal weights,
# plus the no-GradNorm run (equal weights).
cd /root/early_stop/gc
/root/run13/venv/bin/python /root/early_stop/gc_analysis.py \
 "ML init equal=ml_arv_mmoe_gradnorm_init_s43.json" \
 "ML init GN w=ml_arv_mmoe_gradnorm_init_s43.json:0.88,2.11,0.0067" \
 "ML GN ep0 GN w=ml_arv_mmoe_gradnorm_epoch_0_s43.json:0.88,2.11,0.0067" \
 "ML GN ep0 equal=ml_arv_mmoe_gradnorm_epoch_0_s43.json" \
 "ML GN ep4 GN w=ml_arv_mmoe_gradnorm_epoch_4_s43.json:0.81,2.19,0.0011" \
 "ML GN ep9 GN w=ml_arv_mmoe_gradnorm_epoch_9_s43.json:0.75,2.25,0.00066" \
 "ML GN best(ep19) GN w=ml_arv_mmoe_gradnorm_best_s43.json:0.72,2.28,0.00017" \
 "ML GN best(ep19) equal=ml_arv_mmoe_gradnorm_best_s43.json" \
 "ML noGN ep4 equal=ml_arv_mmoe_epoch_4_s43.json" \
 "ML noGN best equal=ml_arv_mmoe_best_s43.json"
