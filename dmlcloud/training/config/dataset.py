from .common import SubConfig

class DatasetConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.dataset_fn = None

    @property
    def dataset_fn(self):
        return self.dct['create_fn']
    
    @dataset_fn.setter
    def dataset_fn(self, value):
        self.dct['create_fn'] = value

    @property
    def dataset_kwargs(self):
        kwargs = dict(self.dct)
        kwargs.pop('create_fn')
        return kwargs

    @dataset_kwargs.setter
    def dataset_kwargs(self, value):
        create_fn = self.dct['create_fn']
        self.dct = value
        self.dct['create_fn'] = create_fn
