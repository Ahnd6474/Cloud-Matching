import os

import torch
import torch.distributed as dist


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
dist.all_reduce(value)
torch.cuda.synchronize()
print(
    f"rank={dist.get_rank()} local_rank={local_rank} "
    f"device={torch.cuda.get_device_name(local_rank)} all_reduce={value.item():.1f}"
)

dist.destroy_process_group()
