"""Time training steps and one validation pass of a study-X baseline: time_baseline.py ARM [PRESET]"""
import copy, os, sys, time
sys.path.insert(0, os.getcwd())
import torch
from configs.config import Config
from data.sequence_generator import load_train_dataset
from evaluation.metrics import load_full_ranking
from models.baselines import build_baseline
from training.ablation_runner import baseline_arm, resolve_dataset_fields
from training.baseline_train import PrefixFeed, evaluate_baseline, load_eval_split

arm = sys.argv[1]
base = Config()
if len(sys.argv) > 2:
    base.apply_dataset_preset(sys.argv[2])
cfg = copy.deepcopy(base)
for k, v in baseline_arm(arm, base).items():
    if k != "label":
        setattr(cfg, k, v)
resolve_dataset_fields(cfg)
dev = torch.device("cuda")
t0 = time.time()
train = load_train_dataset(cfg.processed_data_dir, cfg.max_seq_length)
model = build_baseline(cfg).to(dev)
feed = PrefixFeed(train.sequences, train.targets, cfg.num_items, dev, 42,
                  negatives=model.needs_negatives, same_target=model.needs_same_target)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"setup {time.time() - t0:.1f}s, {len(feed)} examples", flush=True)
order = feed.epoch_order()
model.train()
for n_steps in (20, 200):
    torch.cuda.synchronize(); t = time.time()
    for s in range(n_steps):
        ex = order[s * 256:(s + 1) * 256]
        b = feed.batch(ex)
        loss, _ = model.training_loss(b, feed.gen)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / n_steps
print(f"train {dt * 1000:.1f} ms/step -> {dt * len(feed) / 256 / 60:.1f} min/epoch", flush=True)
seqs, tg, us = load_eval_split(cfg.processed_data_dir, "val", cfg.max_seq_length)
rk = load_full_ranking(cfg.processed_data_dir, "val", tg, us)
for bs in (256, 1024):
    torch.cuda.synchronize(); t = time.time()
    m = evaluate_baseline(model, seqs, rk, 10, bs)
    torch.cuda.synchronize()
    print(f"val eval bs={bs}: {time.time() - t:.1f}s ndcg {m['ndcg']['mean']:.4f}", flush=True)
print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
