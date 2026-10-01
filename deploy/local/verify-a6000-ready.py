from pathlib import Path

import torch

from stochastic_bridge.config import load_config
from stochastic_bridge.model import StochasticImageBridge
from stochastic_bridge.prepared import PreparedBridgeDataset, ShardBatchSampler


root = Path("data/prepared_df2k_adaptive_microstep_v2")
datasets = {}
for split in ("train", "valid"):
    dataset = PreparedBridgeDataset(root / split)
    datasets[split] = dataset
    item = dataset[0]
    tensors = {
        key: tuple(value.shape)
        for key, value in item.items()
        if torch.is_tensor(value)
    }
    finite = {
        key: bool(torch.isfinite(value).all())
        for key, value in item.items()
        if torch.is_tensor(value) and value.is_floating_point()
    }
    print(
        f"{split}: records={len(dataset)} "
        f"format={dataset.manifest.get('format_version')} "
        f"tensors={tensors} finite={finite}"
    )

rank_indices = []
for rank in (0, 1):
    sampler = ShardBatchSampler(
        datasets["train"],
        batch_size=4,
        drop_last=True,
        seed=31,
        num_replicas=2,
        rank=rank,
    )
    batches = list(sampler)
    rank_indices.append({index for batch in batches for index in batch})
    print(f"rank {rank}: steps={len(batches)} records={len(rank_indices[-1])}")
assert rank_indices[0].isdisjoint(rank_indices[1])
assert len(rank_indices[0] | rank_indices[1]) == len(datasets["train"])
print("DDP sampler: ranks are disjoint and cover the full train dataset")

checkpoint = torch.load(
    "outputs/l40_adaptive_gate_fast_ustat_s4_epoch1_30/best.pt",
    map_location="cpu",
    weights_only=False,
)
config = load_config("configs/cloud_l40_adaptive_gate_fast_ustat.yaml")
model = StochasticImageBridge(**config.model.__dict__)
model.load_state_dict(checkpoint["model"])
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(
    "checkpoint:",
    {
        key: checkpoint.get(key)
        for key in ("epoch", "global_step", "best_val_loss")
    },
)
print("checkpoint keys:", sorted(checkpoint))
print(f"model state: strict load passed, parameters={parameter_count:,}")
print(
    "cuda:",
    torch.cuda.is_available(),
    torch.cuda.device_count(),
    [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
)
