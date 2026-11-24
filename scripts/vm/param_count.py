"""Parameter count of a checkpoint, grouped by top-level module."""
import collections
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
groups = collections.Counter()
for sd_key in ("model_state_dict", "moe_state_dict"):
    for k, v in ck.get(sd_key, {}).items():
        if torch.is_tensor(v):
            prefix = "" if sd_key == "model_state_dict" else "moe."
            groups[prefix + k.split(".")[0]] += v.numel()
tot = sum(groups.values())
for g, n in groups.most_common():
    print(f"{g:28s} {n / 1e6:7.3f} M params  {n * 4 / 1e6:6.1f} MB fp32")
print(f"{'TOTAL':28s} {tot / 1e6:7.3f} M params  {tot * 4 / 1e6:6.1f} MB fp32  "
      f"{tot * 2 / 1e6:6.1f} MB fp16  {tot / 1e6:6.1f} MB int8")
print("checkpoint keys:", list(ck.keys()))
