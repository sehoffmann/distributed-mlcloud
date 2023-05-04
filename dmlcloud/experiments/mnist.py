import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms

from ..training.config import DefaultConfig
from .base_settings import default_config


def create_model(config):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def forward_step(trainer, model, batch):
    X,label = batch
    return trainer.loss_fn(model(X),label)


def create_dataset(config):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = MNIST(root='mnist', train=True, transform=transform, download=True)
    test = MNIST(root='mnist', train=False, transform=transform, download=True)
    train_dl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    return train_dl, test_dl


def create_rotating_mnist_config(args):
    cfg = default_config(args)
    cfg.model_fn = create_model
    cfg.dataset_fn = create_dataset
    cfg.forward_fn = forward_step
    return cfg


def add_parser(subparsers):
    parser = subparsers.add_parser('mnist', help='Train a model on rotating MNIST')
    parser.set_defaults(create_config=create_rotating_mnist_config)
    DefaultConfig().create_parser(parser)

