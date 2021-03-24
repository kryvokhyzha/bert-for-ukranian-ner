import torch
import torch.nn as nn


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor,
    num_labels: int,
):
    lfn = nn.CrossEntropyLoss()
    active_loss = attention_mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss, target.view(-1), torch.tensor(lfn.ignore_index).type_as(target)
    )

    loss = lfn(active_logits, active_labels)
    return loss
