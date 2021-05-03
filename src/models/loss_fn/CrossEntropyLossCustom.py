import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossCustom(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight, reduction=reduction)
        self.weight = weight

    def forward(self, output, target, attention_mask, output_dim):
        active_loss = attention_mask.view(-1) == 1
        active_logits = output.view(-1, output_dim)[active_loss]
        active_labels = target.view(-1)[active_loss]

        ce_loss = F.cross_entropy(
            active_logits, active_labels, reduction=self.reduction, weight=self.weight
        )
        return ce_loss
