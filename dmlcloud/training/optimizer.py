import torch
from torch.optim import Adam, AdamW, SGD


def create_adam(config: dict):
    return Adam(
        params=config['params'],
        lr=config['lr'],
        betas=config['betas'],
        eps=config.get('eps', 1e-8),
        weight_decay=config['weight_decay'],
        amsgrad=config['amsgrad']
    )

def get_optimizer_cls(config: dict):
    if config['name'] == 'Adam':
        return Adam
    elif config['name'] == 'AdamW':
        return AdamW
    elif config['name'] == 'SGD':
        return SGD
    else:
        raise ValueError(f'Unknown optimizer name: {config["name"]}')
    

def get_optimizer(params, config: dict):
    optimizer_cls = get_optimizer_cls(config)
    return optimizer_cls(
        params=params,
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
    )