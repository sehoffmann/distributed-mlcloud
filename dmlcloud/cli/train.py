import argparse

from ..experiments import cv_classification, mnist

EXPERIMENTS = [cv_classification, mnist]


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Experiment to run')
    subparsers.required = True

    for exp in EXPERIMENTS:
        exp.add_parser(subparsers)

    args = parser.parse_args()
    trainer = args.create_trainer(args)
    return trainer


def main():
    trainer = parse_args()
    trainer.train()


if __name__ == '__main__':
    main()
