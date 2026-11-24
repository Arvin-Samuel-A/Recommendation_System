import json,glob,os,sys
root="/root/beauty14/outputs/amazon_beauty_2014/ablations"
SKIP={"label","study","model_name","config_diff","study_overrides","resolved_config","training_fingerprint","shared_backbone_with","best_val_ndcg","best_epoch","runtime","source_revision","metrics","num_eval_examples","test_eval","popularity_reference","eval_protocol","status","arm_key","code_revision","training_log_dir","router_parameters"}
for f in sorted(glob.glob(f"{root}/seed_*/*_t_base.json")):
    seed=f.split("/")[-2]
    for r in json.load(open(f)):
        m=r.get("metrics",{})
        g=lambda k:m.get(k,{}).get("mean") if isinstance(m.get(k),dict) else None
        n=g("ndcg"); rc=g("recall"); mr=g("mrr")
        extra=[k for k in r if k not in SKIP]
        print(f"{seed} {r['study']} {r['label']:26s} {r.get('status')} ep={r.get('best_epoch')} val={r.get('best_val_ndcg') or 0:.4f} "
              f"test_ndcg={n if n is None else round(n,4)} rec={rc if rc is None else round(rc,4)} mrr={mr if mr is None else round(mr,4)} extra={extra}")
