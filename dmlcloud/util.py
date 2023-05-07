import os

import horovod.torch as hvd
import torch
import wandb


def hvd_allreduce(val, *args, **kwargs):
    tensor = torch.as_tensor(val)
    reduced = hvd.allreduce(tensor, *args, **kwargs)
    return reduced.cpu().numpy()


def set_wandb_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ['WANDB__SERVICE_WAIT'] = f'{seconds}'


def is_wandb_initialized():
    return wandb.run is not None
