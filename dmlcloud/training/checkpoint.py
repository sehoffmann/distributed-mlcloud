import logging
import json
import sys
import os
from datetime import datetime
import horovod.torch as hvd
from wandb.sdk.lib.runid import generate_id


class ExtendedJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that serializes classes and functions as well (by their name).
    """

    def default(self, o):
        if isinstance(o, type):
            return f'<cls {o.__module__}.{o.__name__}>'
        elif callable(o):
            return f'<fn {o.__module__}.{o.__name__}>'
        
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def get_config_path(model_dir):
    return model_dir / 'config.json'


def get_checkpoint_path(model_dir):
    return model_dir / 'checkpoint.pth'


def get_slurm_id():
    return os.environ.get('SLURM_JOB_ID')


def find_old_checkpoint(base_dir, id_prefix):
    slurm_id = get_slurm_id()
    slurm_dir = next(iter(base_dir.glob(f'*-{id_prefix}{slurm_id}')), None)
    
    if get_config_path(base_dir).exists():
        model_dir = base_dir
        job_id = base_dir.stem.split('-',1)[0]
    elif slurm_id and slurm_dir is not None:
        model_dir = slurm_dir
        job_id = id_prefix + slurm_id
    else:
        job_id = None
        model_dir = None

    return model_dir, job_id


def create_project_dir(base_dir, config):
    slurm_id = get_slurm_id()
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = config.id_prefix + hvd.broadcast_object(slurm_id if slurm_id else generate_id(), name='job_id')
    model_dir = hvd.broadcast_object(base_dir / f'{date_str}-{job_id}', name='model_dir')

    if hvd.rank() == 0:
        os.makedirs(model_dir)
        with open(get_config_path(model_dir), 'w') as file:
            json.dump(config.as_dictionary(), file, cls=ExtendedJSONEncoder)
    return model_dir, job_id


def config_consistency_check(model_dir, config):
    config_path = get_config_path(model_dir)
    with open(config_path, 'r') as file:
        if file.read() != json.dumps(config.as_dictionary(), cls=ExtendedJSONEncoder):
            logging.critical('Config of resumed run does not match current config. Aborting...')
            sys.exit(1)