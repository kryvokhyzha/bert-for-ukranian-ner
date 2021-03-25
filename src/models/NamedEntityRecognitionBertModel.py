import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from models.utils.loss_fn import loss_fn


class NamedEntityRecognitionBertModel(nn.Module):
    def __init__(self, pretrained_model_name: str, num_tag: int):
        super().__init__()

        self.num_tag = num_tag

        config = AutoConfig.from_pretrained(pretrained_model_name)

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.droupout1 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(768, self.num_tag)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        target_tag: torch.Tensor,
    ):
        o1, _ = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        bo_tag = self.droupout1(o1)

        tag = self.linear1(bo_tag)

        loss = loss_fn(tag, target_tag, attention_mask, self.num_tag)

        return tag, loss
