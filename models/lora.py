from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank

        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank, dtype=self.weight.dtype, device=self.weight.device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features, dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + x @ self.lora_A @ self.lora_B


_MOE_PREFIXES = ("moe.router", "moe.experts", "moe.objective_embeddings", "moe.gates")


def inject_lora(model: nn.Module, rank: int) -> None:
    for block in model.blocks:
        attn = block.attn
        attn.q_proj = LoRALinear(attn.q_proj, rank)
        attn.v_proj = LoRALinear(attn.v_proj, rank)

    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks:
        attn = block.attn
        attn.q_proj.lora_A.requires_grad = True
        attn.q_proj.lora_B.requires_grad = True
        attn.v_proj.lora_A.requires_grad = True
        attn.v_proj.lora_B.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for k, v in model.state_dict().items():
        if "lora_" in k or ".lora_A" in k or ".lora_B" in k:
            if not any(k.startswith(prefix) for prefix in _MOE_PREFIXES):
                sd[k] = v
    return sd


def load_lora_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k in model_sd:
            model_sd[k] = v
    model.load_state_dict(model_sd)


def count_lora_parameters(model: nn.Module) -> int:
    total = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            total += param.numel()
    return total
