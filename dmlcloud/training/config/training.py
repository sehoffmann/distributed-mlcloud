from .common import SubConfig, ConfigVar, ArgparseVar


class TrainingConfig(SubConfig):
    seed = ArgparseVar(type=int, help='The random seed')
    epochs = ArgparseVar(type=int, help='The number of epochs')
    batch_size = ArgparseVar(type=int, help='The batch size')
    init_lr = ArgparseVar('--lr', type=float, help='The initial learning rate')
    rampup_epochs = ArgparseVar(type=int, help='The number of epochs to ramp up the learning rate')
    weight_decay = ArgparseVar(type=float, help='The weight decay')
    clip_gradients = ArgparseVar(type=float, help='The gradient clipping threshold')
    log_gradients = ArgparseVar(type=bool, help='Log gradients during training')
    mixed = ArgparseVar(type=bool, help='Use mixed precision training')
    adasum = ArgparseVar(type=bool, help='Use adasum for distributed training')
    check_nans = ArgparseVar(type=bool, help='Check for NaNs during training')

    def set_defaults(self):
        self.seed = None
        self.epochs = 10
        self.batch_size = 32
        self.init_lr = 1e-3
        self.rampup_epochs = 5
        self.weight_decay = 1e-4
        self.clip_gradients = None
        self.log_gradients = True
        self.mixed = False
        self.adasum = False
        self.check_nans = False
