from .common import SubConfig

class ForwardConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.forward_fn = None

    @property
    def forward_fn(self):
        return self.dct['create_fn']

    @forward_fn.setter
    def forward_fn(self, value):
        self.dct['create_fn'] = value
    
    @property
    def forward_kwargs(self):
        kwargs = dict(self.dct)
        kwargs.pop('create_fn')
        return kwargs
    
    @forward_kwargs.setter
    def forward_kwargs(self, value):
        create_fn = self.dct['create_fn']
        self.dct = value
        self.dct['create_fn'] = create_fn