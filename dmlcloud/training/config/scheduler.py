from .common import SubConfig

class SchedulerConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.scheduler_fn = None

    @property
    def scheduler_fn(self):
        return self.dct['create_fn']
    
    @scheduler_fn.setter
    def scheduler_fn(self, value):
        self.dct['create_fn'] = value

    @property
    def scheduler_kwargs(self):
        kwargs = dict(self.dct)
        kwargs.pop('create_fn')
        return kwargs
    
    @scheduler_kwargs.setter
    def scheduler_kwargs(self, value):
        create_fn = self.dct['create_fn']
        self.dct = value
        self.dct['create_fn'] = create_fn