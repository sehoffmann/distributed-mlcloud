from argparse import BooleanOptionalAction

from .common import SubConfig, ConfigVar, ArgparseVar


class TrainingConfig(SubConfig):
    seed = ArgparseVar(type=int, help='The random seed')
    epochs = ArgparseVar(type=int, help='The number of epochs')
    batch_size = ArgparseVar(type=int, help='The batch size')
    base_lr = ArgparseVar(type=float, help='The learning rate per 32-er batch / GPU')
    scale_lr = ArgparseVar(type=bool, action=BooleanOptionalAction, help='Scale the learning rate by the numbers of 32-er batches / GPUs')
    rampup_epochs = ArgparseVar(type=int, help='The number of epochs to ramp up the learning rate')
    weight_decay = ArgparseVar(type=float, help='The weight decay')
    clip_gradients = ArgparseVar(type=float, help='The gradient clipping threshold')
    log_gradients = ArgparseVar(type=bool, action=BooleanOptionalAction, help='Log gradients during training')
    mixed = ArgparseVar(type=bool, action=BooleanOptionalAction, help='Use mixed precision training')
    adasum = ArgparseVar(type=bool, action=BooleanOptionalAction, help='Use adasum for distributed training')
    check_nans = ArgparseVar(type=bool, action=BooleanOptionalAction, help='Check for NaNs during training')

    def set_defaults(self):
        self.seed = None
        self.epochs = 10
        self.batch_size = 32
        self.base_lr = 1e-3
        self.scale_lr = True
        self.rampup_epochs = 5
        self.weight_decay = 1e-4
        self.clip_gradients = None
        self.log_gradients = True
        self.mixed = False
        self.adasum = False
        self.check_nans = False
