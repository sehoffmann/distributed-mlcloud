from torchvision import transforms

TRANSFORMS = [
    'collate',
    'flip',
    'flip_and_rotate',
]

def create_transform(task, kind):
    if task.mean is not None and task.std is not None:
        mean = task.mean
        std = task.std
    else:
        mean = tuple([0.5 for _ in range(task.input_channels)])
        std = tuple([0.5 for _ in range(task.input_channels)])

    if kind == 'collate':
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(task.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif kind == 'flip':
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(task.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif kind == 'flip_and_rotate':
        return transforms.Compose(
            [
                transforms. RandomRotation(10),
                transforms.RandomResizedCrop(task.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError(f'Unknown transform {kind}')