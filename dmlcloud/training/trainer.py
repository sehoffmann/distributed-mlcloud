import logging
import random
import sys
from contextlib import nullcontext
from datetime import datetime, timedelta

import horovod.torch as hvd
import numpy as np
import torch
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR
from progress_table import ProgressTable

from ..util import is_wandb_initialized, set_wandb_startup_timeout
from .checkpoint import resume_project_dir
from .metrics import MetricSaver
from .util import (
    git_hash,
    log_config,
    log_delimiter,
    log_diagnostics,
    log_git,
    print_worker,
    setup_horovod,
    setup_logging,
)


class TrainerInterface:
    """
    These methods must be implemented for each experiment
    """

    def create_dataset(self):
        """
        Returns a tuple of (train_dl, val_dl).
        Will be available as self.train_dl and self.val_dl.
        These shall be iterators that yield batches.
        """
        raise NotImplementedError()

    def create_model(self):
        """
        Returns a torch.nn.Module.
        Will be available as self.model.
        If you need multiple networks, e.g. for GANs, wrap them in a nn.Module.
        """
        raise NotImplementedError()

    def create_loss(self):
        """
        Returns a loss function.
        Will be available as self.loss_fn.
        """
        raise NotImplementedError()

    def create_scheduler(self):
        """
        Returns a scheduler or None.
        """
        return None

    def create_optimizer(self, params, lr):
        """
        Returns an optimizer.
        Will be available as self.optimizer.
        """
        raise NotImplementedError()


    def metric_names(self):
        """
        Returns a list with custom metrics that are displayed during training.
        """
        return []

class BaseTrainer(TrainerInterface):
    def __init__(self, config):
        self.cfg = config
        self.base_dir = config.project_dir
        self.model_dir = None
        self.job_id = None
        self.is_resumed = False
        self.train_metrics = MetricSaver()
        self.val_metrics = MetricSaver()
        self.epoch = 1
        self.setup_all()

    def setup_all(self):
        setup_horovod()
        self.seed()
        self.setup_general()
        self.model_dir, self.job_id, self.is_resumed = resume_project_dir(self.base_dir, self.cfg)
        self.setup_wandb()
        self.setup_table()
        self.print_diagnositcs()

        self.setup_dataset()
        self.setup_model()
        self.setup_loss()
        self.setup_optimizer()
        self.load_checkpoint()

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def print_diagnositcs(self):
        log_delimiter()
        log_git()
        log_diagnostics(self.device)
        log_config(self.cfg)

    def setup_table(self):
        columns = ['Epoch', 'ETA', 'train/loss', 'val/loss']
        columns += self.metric_names()
        columns += ['ms/step']
        self.table = ProgressTable(columns=columns, print_row_on_update=False)

    def setup_general(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda', hvd.local_rank())
        else:
            self.device = torch.device('cpu')

        torch.set_num_threads(8)
        setup_logging()
        self.cfg.git_hash = git_hash()

    def seed(self):
        if self.cfg.seed is None:
            seed = int.from_bytes(random.randbytes(4), byteorder='little')
            self.cfg.seed = hvd.broadcast_object(seed)

        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)

    def setup_wandb(self):
        if hvd.rank() != 0:
            return

        set_wandb_startup_timeout(600)
        wandb.init(
            project=self.cfg.project_name,
            name=self.cfg.experiment_name,
            dir=self.model_dir,
            id=self.job_id,
            resume='must' if self.is_resumed else 'never',
            config=self.cfg.as_dictionary(),
        )

    def setup_dataset(self):
        hvd.barrier()
        if hvd.rank() == 0:
            logging.info('Creating dataset')
            self.train_dl, self.val_dl = self.create_dataset()
            hvd.barrier()
        else:
            hvd.barrier()  # wait until rank 0 has created the dataset (e.g. downloaded it)
            self.train_dl, self.val_dl = self.create_dataset()
            

    def setup_model(self):
        self.model = self.create_model().to(self.device)

    def setup_loss(self):
        self.loss_fn = self.create_loss()

    def setup_optimizer(self):
        lr = self.cfg.init_lr * (self.cfg.batch_size / 32.0)
        if self.cfg.adasum:
            lr_scaling = hvd.size()
        elif hvd.nccl_built():
            lr_scaling = hvd.local_size()
        else:
            lr_scaling = 1.0
        lr *= lr_scaling

        logging.info(f'LR: {self.cfg.init_lr:.1e},  Scaled: {lr:.1e}')

        optimizer = self.create_optimizer(self.model.parameters(), lr)
        self.optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=self.model.named_parameters(), op=hvd.Adasum if self.cfg.adasum else hvd.Average
        )

        schedulers = []
        if self.cfg.rampup_epochs:
            linear_warmup = LinearLR(
                self.optimizer, start_factor=1 / lr_scaling, end_factor=1.0, total_iters=self.cfg.rampup_epochs
            )
            schedulers.append(linear_warmup)

        user_scheduler = self.create_scheduler()
        if isinstance(user_scheduler, list):
            schedulers.extend(user_scheduler)
        elif user_scheduler is not None:
            schedulers.append(user_scheduler)

        self.scheduler = ChainedScheduler(schedulers)
        self.scaler = GradScaler(enabled=self.cfg.mixed)

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.train_metrics = MetricSaver(state_dict['train_metrics'])
        self.val_metrics = MetricSaver(state_dict['val_metrics'])
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.scaler.load_state_dict(state_dict['scaler_state'])

    def state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'train_metrics': self.train_metrics.epochs,
            'val_metrics': self.val_metrics.epochs,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
        }
        return state_dict

    def load_checkpoint(self):
        cp_path = self.model_dir / 'checkpoint.pt'
        if cp_path.exists():
            state_dict = torch.load(cp_path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.epoch += 1
            logging.info(f'Loaded checkpoint from {cp_path}')
            logging.info(
                f'Continuing training at epoch {self.epoch}, previous loss: {self.train_metrics.last["loss"]:.3f}'
            )
        elif self.is_resumed:
            logging.critical('No checkpoint found!')
            sys.exit(1)

    def save_checkpoint(self):
        if hvd.rank() != 0:
            return

        checkpoint_path = self.model_dir / 'checkpoint.pt'
        best_path = self.model_dir / 'best.pt'

        torch.save(self.state_dict(), checkpoint_path)
        if self.is_best_epoch():
            torch.save(self.state_dict(), best_path)
            if is_wandb_initialized():
                wandb.save(str(best_path), policy='now', base_path=str(self.model_dir))

        self.train_metrics.scalars_to_csv(self.model_dir / 'train_metrics.csv')
        self.val_metrics.scalars_to_csv(self.model_dir / 'val_metrics.csv')

    def is_best_epoch(self):
        return self.val_metrics.last['loss'] == min(self.val_metrics.get_metrics('loss'))

    def log_epoch(self):
        if hvd.rank() != 0:
            return

        self.table['train/loss'] = self.train_metrics.last['loss']
        self.table['val/loss'] = self.val_metrics.last['loss']
        self.table['ETA'] = str(self.train_metrics.last['eta'])
        self.table['ms/step'] = f'{self.train_metrics.last["ms_per_step"]:.1f}'

        for metric in self.metric_names():
            splits = metric.split('/', 1)
            if len(splits) == 2:
                current_metrics = self.train_metrics if splits[0] == 'train' else self.val_metrics
                self.table[metric] = current_metrics.last[splits[1]]
            else:
                self.table[metric] = self.train_metrics.last[metric]

        self.table.next_row()

        if is_wandb_initialized():
            self.log_wandb()

    def log_wandb(self):
        metrics = {}
        for key, value in self.train_metrics.scalar_metrics()[-1].items():
            metrics[f'train/{key}'] = value

        for key, value in self.val_metrics.scalar_metrics()[-1].items():
            metrics[f'val/{key}'] = value

        wandb.log(metrics)
        if self.is_best_epoch():
            wandb.run.summary['best/epoch'] = self.epoch
            for key, value in metrics.items():
                wandb.run.summary[f'best/{key}'] = value

    def forward_step(self, batch_idx, batch):
        raise NotImplementedError()

    def switch_mode(self, train=True):
        if train:
            self.model.train()
            self.current_metrics = self.train_metrics
            self.is_train = True
            self.is_eval = False
        else:
            self.model.eval()
            self.current_metrics = self.val_metrics
            self.is_train = False
            self.is_eval = True

    def pre_train(self):
        self.epoch_start_time = datetime.now()

    def post_train(self):
        self.epoch_train_end_time = datetime.now()

    def train_epoch(self, max_steps=None):
        self.pre_train()
        self.switch_mode(train=True)
        self.table['Epoch'] = self.epoch
        if hasattr(self.train_dl, 'sampler') and hasattr(self.train_dl.sampler, 'set_epoch'):
            self.train_dl.sampler.set_epoch(self.epoch)

        nan_ctx_manager = torch.autograd.detect_anomaly() if self.cfg.check_nans else nullcontext()
        for batch_idx, batch in enumerate(self.train_dl):
            if max_steps and batch_idx >= max_steps:
                break

            with nan_ctx_manager:
                # forward pass
                with autocast(enabled=self.cfg.mixed):
                    loss = self.forward_step(batch_idx, batch)
                # backward pass
                self.scaler.scale(loss).backward()  # scale loss and, in turn, gradients to prevent underflow

            self.optimizer.synchronize()  # make sure all async allreduces are done
            self.scaler.unscale_(self.optimizer)  # now, unscale gradients again

            if self.cfg.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradients)

            with self.optimizer.skip_synchronize():  # we already synchronized manually
                self.scaler.step(self.optimizer)
                self.scaler.update()  # adjust gradient scaling based on number of infs/nans
                self.optimizer.zero_grad()

            if not torch.isnan(loss):  # mixed-precision might produce nan steps
                self.log_metric('loss', loss)
                self.log_metric('n_steps', 1, hvd.Sum, allreduce=False)
                self.log_metric('n_nan', 0, hvd.Sum, allreduce=False)
            else:
                self.log_metric('n_nan', 1, hvd.Sum, allreduce=False)

        self.n_train_batches = batch_idx + 1

        self.log_metric('lr', self.scheduler.get_last_lr()[0], reduction=None)
        for k, v in self.scaler.state_dict().items():
            self.log_metric(f'scaler/{k}', v, reduction=None)

        if self.scheduler is not None:
            self.scheduler.step()

        self.post_train()

    def pre_eval(self):
        pass

    def post_eval(self):
        self.epoch_end_time = datetime.now()
        
        n_remaining = self.cfg.epochs - self.epoch - 1
        eta = n_remaining * (self.epoch_end_time  - self.start_time) / (self.epoch + 1)
        eta -= timedelta(microseconds=eta.microseconds)
        self.train_metrics.log_metric('eta', str(eta), reduction=None)

        per_step = (self.epoch_train_end_time  - self.epoch_start_time) / self.n_train_batches
        per_step = per_step.total_seconds() * 1000
        self.train_metrics.log_metric('ms_per_step', per_step, reduction=None)

        self.train_metrics.reduce()
        self.val_metrics.reduce()
        
        self.log_epoch()
        self.save_checkpoint()
        self.epoch += 1

    def evaluate_epoch(self):
        self.pre_eval()
        self.switch_mode(train=False)

        for batch_idx, batch in enumerate(self.val_dl):
            with torch.no_grad():
                loss = self.forward_step(batch_idx, batch).item()
                self.log_metric('loss', loss)

        self.post_eval()

    def pre_training(self):
        hvd.barrier()
        log_delimiter()
        print_worker('READY')
        hvd.barrier()
        self.start_time = datetime.now()
        logging.info('Starting training...')

    def post_training(self):
        self.table.close()
        logging.info('Training finished.')

    def train(self, max_steps=None):
        self.pre_training()
        while self.epoch <= self.cfg.epochs:
            self.train_epoch(max_steps)
            self.evaluate_epoch()
        self.post_training()

    def log_metric(self, name, value, reduction=hvd.Average, allreduce=True):
        self.current_metrics.log_metric(name, value, reduction, allreduce)