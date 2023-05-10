import torchvision.datasets as datasets

class Task:
    def __init__(self, name, dataset_cls, num_classes, input_channels, img_size, mean=None, std=None):
        self.name = name
        self.dataset_cls = dataset_cls
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def default_config(self, config):
        pass


class CIFAR10(Task):

    def __init__(self):
        super().__init__('cifar10', datasets.CIFAR10, 10, 3, 32, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

    def default_config(self, config):
        config.model_preset = 'cnn_minimal'
        config.train_transform_preset = 'flip_and_rotate'
        config.epochs = 30


class CIFAR100(Task):

    def __init__(self):
        super().__init__('cifar100', datasets.CIFAR100, 100, 3, 32, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))

    def default_config(self, config):
        config.model_preset = 'cnn'
        config.train_transform_preset = 'flip_and_rotate'
        config.epochs = 60


class ImageNet(Task):

    def __init__(self):
        super().__init__('imagenet', datasets.ImageNet, 1000, 3, 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def default_config(self, config):
        config.model_preset = 'resnet34'
        config.train_transform_preset = 'flip_and_rotate'
        config.epochs = 90


class EMNIST(Task):

    def __init__(self):
        super().__init__('emnist', datasets.EMNIST, 47, 1, 28, mean=(0.5,), std=(0.5,))

    def default_config(self, config):
        config.model_preset = 'cnn_minimal'
        config.train_transform_preset = 'flip_and_rotate'
        config.epochs = 30


class FashionMNIST(Task):

    def __init__(self):
        super().__init__('fashion_mnist', datasets.FashionMNIST, 10, 1, 28, mean=(0.5,), std=(0.5,))

    def default_config(self, config):
        config.model_preset = 'cnn_minimal'
        config.train_transform_preset = 'flip_and_rotate'
        config.epochs = 30


class SVHN(Task):

    def __init__(self):
        super().__init__('svhn', datasets.SVHN, 10, 3, 32, mean=(0.5,), std=(0.5,))

    def default_config(self, config):
        config.model_preset = 'cnn_minimal'
        config.train_transform_preset = 'flip_and_rotate'
        config.epochs = 30


_tasks = [
    CIFAR10(), 
    CIFAR100(), 
    ImageNet(), 
    EMNIST(), 
    FashionMNIST(), 
    SVHN()
]
TASKS = {task.name: task for task in _tasks}