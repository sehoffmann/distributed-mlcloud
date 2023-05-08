import csv

import torch
import horovod.torch as hvd

class Metric:

    def __init__(self, name, reduction = hvd.Average, allreduce=True):
        self.name = name
        self.reduction = reduction
        self.batch_values = []
        self.allreduce = allreduce

    def _reduce(self, value, dim=0):
        match self.reduction:
            case hvd.Average:
                return value.mean(dim=dim)
            case hvd.Sum:
                return value.sum(dim=dim)
            case hvd.Min:
                return value.min(dim=dim)[0]
            case hvd.Max:
                return value.max(dim=dim)[0]
            case hvd.Product:
                return value.prod(dim=dim)

    def add_batch_value(self, value):
        tensor = torch.as_tensor(value)
        tensor = self._reduce(tensor, dim=0)
        self.batch_values.append(tensor)

    def reduce(self):
        tensor = torch.stack(self.batch_values)
        tensor = self._reduce(tensor, dim=0)
        if self.allreduce:
            return hvd.allreduce(tensor, op=self.reduction, name=f'metric/{self.name}')
        else:
            return tensor

class MetricSaver:

    def __init__(self, epochs=None):
        self.epochs = epochs or []
        self.current_metrics = {}

    @property
    def last(self):
        return self.epochs[-1]

    def get_metrics(self, name):
        return [epoch[name] for epoch in self.epochs]

    def reduce(self):
        reduced = {}
        for name, metric in self.current_metrics.items():
            reduced[name] = metric.reduce()
        self.epochs.append(reduced)
        self.current_metrics = {}

    def log_metric(self, name, value, reduction=hvd.Average, allreduce=True):
        if name not in self.current_metrics:
            self.current_metrics[name] = Metric(name, reduction, allreduce)
        metric = self.current_metrics[name]
        metric.add_batch_value(value)

    def scalar_metrics(self, with_epoch=False):
        scalars = []
        for epoch, metrics in enumerate(self.epochs):
            dct = {}
            if with_epoch:
                dct['epoch'] = epoch+1

            for name, value in metrics.items():
                if value.dim() == 0:
                    dct[name] = value.item()
            scalars.append(dct)
            
        return scalars

    def scalars_to_csv(self, path):
        with open(path, 'w') as file:
            scalar_metrics = self.scalar_metrics(with_epoch=True)
            writer = csv.DictWriter(file, fieldnames=scalar_metrics[0].keys())
            writer.writeheader()
            writer.writerows(scalar_metrics)