import argparse

from ..training import Trainer
from ..experiments import rotating_mnist

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Experiment to run')
    subparsers.required=True
    rotating_mnist.add_parser(subparsers)

    args = parser.parse_args()
    create_config_fn = args.create_config
    config = create_config_fn(args)
    return args, config


def main():
    args, config = parse_args()
    trainer = Trainer(config)    
    trainer.train()
    
    

if __name__ == '__main__':
    main()