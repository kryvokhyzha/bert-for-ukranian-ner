from torch import nn


class AccuracyCallbackCustom(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, attention_mask):
        active_loss = attention_mask.view(-1) == 1
        output_tags = output.argmax(2).view(-1)[active_loss]
        active_labels = target.view(-1)[active_loss]

        correct = active_labels.eq(output_tags)
        return correct.sum() / correct.shape[0]
