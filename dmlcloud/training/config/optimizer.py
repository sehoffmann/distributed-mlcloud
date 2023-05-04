from .common import SubConfig

class OptimizerConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.optimizer_fn = None

    @property
    def optimizer_fn(self):
        return self.dct['create_fn']
    
    @optimizer_fn.setter
    def optimizer_fn(self, value):
        self.dct['create_fn'] = value
    
    @property
    def optimizer_kwargs(self):
        kwargs = dict(self.dct)
        kwargs.pop('create_fn')
        return kwargs

    @optimizer_kwargs.setter
    def optimizer_kwargs(self, value):
        create_fn = self.dct['create_fn']
        self.dct = value
        self.dct['create_fn'] = create_fn