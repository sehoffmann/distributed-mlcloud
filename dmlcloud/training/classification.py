from torch import nn

from .trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def forward_step(self, batch_idx, batch):
        X, label = (tensor.to(self.device, non_blocking=True) for tensor in batch)
        pred = self.model(X)

        acc = (pred.argmax(dim=1) == label).float().mean()
        self.log_metric('acc', acc)

        return self.loss_fn(pred, label)

    def create_loss(self):
        return nn.CrossEntropyLoss()

    def metric_names(self):
        return ['train/acc', 'val/acc']
