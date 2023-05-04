from .common import SubConfig

class LossConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.loss_fn = None

    @property
    def loss_fn(self):
        return self.dct['create_fn']
    
    @loss_fn.setter
    def loss_fn(self, value):
        self.dct['create_fn'] = value

    @property
    def loss_kwargs(self):
        kwargs = dict(self.dct)
        kwargs.pop('create_fn')
        return kwargs
    
    @loss_kwargs.setter
    def loss_kwargs(self, value):
        create_fn = self.dct['create_fn']
        self.dct = value
        self.dct['create_fn'] = create_fn