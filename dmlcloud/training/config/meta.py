from pathlib import Path

from .common import SubConfig

class MetaConfig(SubConfig):

    def __init__(self, dct):
        self.dct = dct
        self.project_dir = Path('./')
        self.model_dir = None
        self.id_prefix = ''
        self.project_name = None
        self.experiment_name = None

    def add_arguments(self, parser):
        parser.add_argument('--dir', type=Path, default=None, help='The project directory')
        parser.add_argument('--id-prefix', type=str, default=None, help='The id prefix for the experiment')
        parser.add_argument('--project-name', type=str, default=None, help='The wandb project name')
        parser.add_argument('--experiment-name', type=str, default=None, help='The wandb experiment name')

    def parse_args(self, args):
        if args.dir:
            self.project_dir = args.dir
        if args.id_prefix:
            self.id_prefix = args.id_prefix
        if args.project_name:
            self.project_name = args.project_name
        if args.experiment_name:
            self.experiment_name = args.experiment_name

    @property
    def project_dir(self):
        return self.dct['project_dir']

    @project_dir.setter
    def project_dir(self, value):
        self.dct['project_dir'] = value

    @property
    def model_dir(self):
        return self.dct['model_dir']
    
    @model_dir.setter
    def model_dir(self, value):
        self.dct['model_dir'] = value

    @property
    def id_prefix(self):
        return self.dct['id_prefix']
    
    @id_prefix.setter
    def id_prefix(self, value):
        self.dct['id_prefix'] = value

    @property
    def project_name(self):
        return self.dct['project_name']
    
    @project_name.setter
    def project_name(self, value):
        self.dct['project_name'] = value
    
    @property
    def experiment_name(self):
        return self.dct['experiment_name']
    
    @experiment_name.setter
    def experiment_name(self, value):
        self.dct['experiment_name'] = value