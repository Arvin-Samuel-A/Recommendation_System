# Personalization & Federated — Design

## Relevant Existing Architecture

Greenfield. Design follows Prompt_v6.2 Steps 10, 16–17.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `models/lora.py` | Create | LoRALinear, inject_lora, get_lora_state_dict, load_lora_state_dict, count_lora_parameters |
| `training/local_train.py` | Create | run_local_personalization, per-user train/eval |
| `training/federated.py` | Create | weighted_fedavg, federated simulation loop, communication accounting |

## Proposed Architecture

### LoRA Injection

```
For each attention layer in the Transformer:
  Replace Q projection: nn.Linear → LoRALinear(original, rank)
  Replace V projection: nn.Linear → LoRALinear(original, rank)

LoRALinear.forward(x):
  return F.linear(x, frozen_weight, bias) + x @ A @ B
```

After injection:
1. Freeze all parameters: `param.requires_grad = False` for all
2. Unfreeze LoRA: `param.requires_grad = True` for LoRA A and B only
3. Construct optimizer with LoRA parameters only

### Local Personalization Flow

```python
def run_local_personalization(config, backbone_path, ...):
    sampled_users = deterministic_sample(all_users, 1000, seed=config.seed)
    results = []

    for user_id in sampled_users:
        # Fresh model + fresh LoRA for every user
        model = load_model(backbone_path)
        inject_lora(model, config.lora_rank)
        freeze_backbone(model)
        assert count_trainable_backbone_params(model) == 0

        # Train LoRA
        optimizer = AdamW(get_lora_params(model), lr=lora_lr, weight_decay=0)
        for epoch in range(config.local_epochs):
            train_one_epoch(model, user_train_data, optimizer)

        # Evaluate with LoRA
        ndcg_with = evaluate_user(model, user_test_data, ...)

        # Evaluate without LoRA (same backbone, no adaptation)
        remove_lora(model)  # or load fresh backbone
        ndcg_without = evaluate_user(model, user_test_data, ...)

        results.append({user_id, ndcg_with, ndcg_without, gain, target_is_cold})

    save_report(results)
    generate_plots(results)
```

### Federated Simulation Flow

```python
def run_federated(config, backbone_path, ...):
    model = load_model(backbone_path)
    inject_lora(model, config.lora_rank)
    global_lora_state = get_lora_state_dict(model)
    frozen_checksum = compute_checksum(get_frozen_state(model))

    for round_idx in range(config.federated_rounds):
        assert compute_checksum(get_frozen_state(model)) == frozen_checksum

        clients = sample_clients(all_users, config.clients_per_round, round_idx)
        client_updates, client_counts = [], []

        for client_id in clients:
            client_model = load_model(backbone_path)
            inject_lora(client_model, config.lora_rank)
            load_lora_state_dict(client_model, global_lora_state)
            freeze_backbone(client_model)

            update, n = local_train_client(client_model, client_id, ...)

            # ASSERT: no MoE keys in update
            assert not any(k.startswith(("moe.router", "moe.experts",
                                          "moe.objective_embeddings", "moe.gates"))
                           for k in update.keys())

            client_updates.append(update)
            client_counts.append(n)

        global_lora_state = weighted_fedavg(client_updates, client_counts)

        # Evaluate global model with aggregated LoRA
        load_lora_state_dict(model, global_lora_state)
        val_metrics = evaluate(model, val_dataset, ...)
        log_round(round_idx, val_metrics, communication_cost)

    assert compute_checksum(get_frozen_state(model)) == frozen_checksum
```

### LoRA State Dict Filtering

```python
def get_lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items()
            if any(pattern in k for pattern in ('lora_', '.A', '.B'))
            and not k.startswith(('moe.router', 'moe.experts',
                                   'moe.objective_embeddings', 'moe.gates'))}
```

## Key Design Decisions

### D1: Fresh LoRA Per Personalization User
Each personalization user starts from the identical frozen checkpoint with a freshly initialized LoRA adapter. This ensures the personalization gain measurement reflects adaptation from a known starting point, not accumulated state.

### D2: Backbone Checksum Enforcement
A hash over all frozen parameters is computed before personalization/federated begins and verified before/after every operation. This catches any accidental backbone mutation.

### D3: LoRA Key Filtering by Name Pattern
State dict filtering uses both positive patterns (`lora_`, `.A`, `.B`) and negative patterns (MoE prefixes). The negative filter is a safety net — even if parameter naming is slightly different, MoE keys are explicitly excluded.

### D4: Communication Cost = LoRA Only
The efficiency claim (RQ4) depends on LoRA-only communication. Expert parameters, router parameters, and objective embeddings are global and never transmitted. The cost formula uses `lora_param_count * sizeof(bf16)`.

### D5: Weighted FedAvg Default
Weighting by sample count (vs simple mean) gives more influence to clients with more data, improving convergence. Simple mean is available as a fallback for ablation.

## Affected Components

- Depends on: pretrained checkpoint from `training/pretrain.py`, `models/transformer.py`, `models/moe.py`
- `models/lora.py` is also used by `training/ablation_runner.py` (Ablation E)
- `training/federated.py` is used by Ablation A (config 6)
- Communication cost feeds into Ablation H6 and core tradeoff plots

## Testing Strategy

- LoRA injection: verify Q and V are wrapped, other layers unchanged
- LoRA state dict: verify keys match pattern, no MoE keys present
- Parameter count: verify matches expected `2 * num_layers * (d_in * rank + rank * d_out)`
- Freeze verification: assert zero trainable backbone params after injection
- Checkpoint integrity: frozen checksum unchanged after personalization
- FedAvg: verify weighted average on known inputs matches manual computation
- Communication cost: verify formula matches logged values
