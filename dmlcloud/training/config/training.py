from .common import SubConfig

class TrainingConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.seed = None
        self.epochs = 10
        self.batch_size = 32
        self.init_lr = 1e-3
        self.rampup_epochs = 5
        self.weight_decay = 1e-4
        self.clip_gradients = None
        self.mixed = False
        self.adasum = False
        self.check_nans = False

    def add_arguments(self, parser):
        parser.add_argument('--seed', type=int, default=None, help='The random seed')
        parser.add_argument('--epochs', type=int, default=None, help='The number of epochs to train')
        parser.add_argument('--batch-size', type=int, default=None, help='The batch size')
        parser.add_argument('--init-lr', type=float, default=None, help='The initial learning rate')
        parser.add_argument('--rampup-epochs', type=int, default=None, help='The number of epochs to rampup the learning rate')
        parser.add_argument('--weight-decay', type=float, default=None, help='The weight decay')
        parser.add_argument('--clip-gradients', type=float, default=None, help='The gradient clipping value')
        parser.add_argument('--mixed', action='store_true', help='Whether to use mixed precision training')
        parser.add_argument('--adasum', action='store_true', help='Whether to use adasum for distributed training')
        parser.add_argument('--check-nans', action='store_true', help='Whether to check for NaNs during training')


    def parse_args(self, args):
        if args.seed:
            self.seed = args.seed
        if args.epochs:
            self.epochs = args.epochs
        if args.batch_size:
            self.batch_size = args.batch_size
        if args.init_lr:
            self.init_lr = args.init_lr
        if args.rampup_epochs:
            self.rampup_epochs = args.rampup_epochs
        if args.weight_decay:
            self.weight_decay = args.weight_decay
        if args.clip_gradients:
            self.clip_gradients = args.clip_gradients
        self.mixed = args.mixed
        self.adasum = args.adasum
        self.check_nans = args.check_nans

    @property
    def seed(self):
        return self.dct['seed']
    
    @seed.setter
    def seed(self, value):
        self.dct['seed'] = value

    @property
    def epochs(self):
        return self.dct['epochs']
    
    @epochs.setter
    def epochs(self, value):
        self.dct['epochs'] = value

    @property
    def batch_size(self):
        return self.dct['batch_size']
    
    @batch_size.setter
    def batch_size(self, value):
        self.dct['batch_size'] = value

    @property
    def init_lr(self):
        return self.dct['init_lr']

    @init_lr.setter
    def init_lr(self, value):
        self.dct['init_lr'] = value
    
    @property
    def rampup_epochs(self):
        return self.dct['rampup_epochs']
    
    @rampup_epochs.setter
    def rampup_epochs(self, value):
        self.dct['rampup_epochs'] = value
    
    @property
    def weight_decay(self):
        return self.dct['weight_decay']

    @weight_decay.setter
    def weight_decay(self, value):
        self.dct['weight_decay'] = value

    @property
    def clip_gradients(self):
        return self.dct['clip_gradients']
    
    @clip_gradients.setter
    def clip_gradients(self, value):
        self.dct['clip_gradients'] = value

    @property
    def mixed(self):
        return self.dct['mixed']
    
    @mixed.setter
    def mixed(self, value):
        self.dct['mixed'] = value
    
    @property
    def adasum(self):
        return self.dct['adasum']

    @adasum.setter
    def adasum(self, value):
        self.dct['adasum'] = value

    @property
    def check_nans(self):
        return self.dct['check_nans']
    
    @check_nans.setter
    def check_nans(self, value):
        self.dct['check_nans'] = value