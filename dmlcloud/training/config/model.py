from .common import SubConfig

class ModelConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.model_fn = None

    @property
    def model_fn(self):
        return self.dct['create_fn']
    
    @model_fn.setter
    def model_fn(self, value):
        self.dct['create_fn'] = value

    @property
    def model_kwargs(self):
        kwargs = dict(self.dct)
        kwargs.pop('create_fn')
        return kwargs
    
    @model_kwargs.setter
    def model_kwargs(self, value):
        create_fn = self.dct['create_fn']
        self.dct = value
        self.dct['create_fn'] = create_fn