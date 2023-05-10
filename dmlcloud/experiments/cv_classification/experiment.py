import argparse

import horovod.torch as hvd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from ...training import ClassificationTrainer
from .tasks import TASKS
from .transform import create_transform


class CVClassificationTrainer(ClassificationTrainer):
    def create_model(self):
        create_fn = self.cfg.dct['model']['create_fn']
        kwargs = dict(self.cfg.dct['model'])
        kwargs.pop('create_fn')
        return create_fn(TASKS[self.cfg.task], **kwargs)

    def create_dataset(self):
        task = TASKS[self.cfg.task]
        
        train_transform = create_transform(task, self.cfg.train_transform_preset or 'collate')
        val_transform = create_transform(task, self.cfg.eval_transform_preset or 'collate')

        train = task.dataset_cls(root=self.cfg.data_dir / task.name, train=True, transform=train_transform, download=True)
        test = task.dataset_cls(root=self.cfg.data_dir / task.name, train=False, transform=val_transform, download=True)
        train_sampler = torch.utils.data.DistributedSampler(
            train, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True, seed=self.cfg.seed
        )
        val_sampler = torch.utils.data.DistributedSampler(
            test, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False, seed=self.cfg.seed
        )
        train_dl = torch.utils.data.DataLoader(train, batch_size=self.cfg.batch_size, sampler=train_sampler, num_workers=4)
        test_dl = torch.utils.data.DataLoader(test, batch_size=self.cfg.batch_size, sampler=val_sampler, num_workers=4)
        return train_dl, test_dl

    def create_optimizer(self, params, lr):
        return torch.optim.AdamW(params, lr=lr, weight_decay=self.cfg.weight_decay)

    def create_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=1e-7)
