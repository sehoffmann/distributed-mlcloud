from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..training.config import DefaultConfig


def default_scheduler(config, optimizer):
    return CosineAnnealingLR(
        optimizer, 
        T_max = config.epochs,
        eta_min=1e-7
    )

def default_optimizer(config, params, lr, **kwargs):
    return AdamW(
        params, 
        lr=lr,
        **kwargs    
    )

def default_loss(config):
    return CrossEntropyLoss()


def default_config(args, config_cls=DefaultConfig):
    cfg = config_cls()
    cfg.parse_args(args)
    cfg.optimizer_fn = default_optimizer
    cfg.scheduler_fn = default_scheduler
    cfg.loss_fn = default_loss
    return cfg