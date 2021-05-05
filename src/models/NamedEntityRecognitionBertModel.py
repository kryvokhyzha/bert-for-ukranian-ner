from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class NamedEntityRecognitionBertModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        output_dim: int,
        lstm_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout_rate: float = 0.3,
        lstm_bidirectional_flag: bool = True,
        cnn_dropout_rate: float = 0.4,
        fc_dropout_rate: float = 0.4,
        use_lstm_flag: bool = False,
        use_cnn_flag: bool = False,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.use_lstm_flag = use_lstm_flag
        self.use_cnn_flag = use_cnn_flag

        config = AutoConfig.from_pretrained(pretrained_model_name)

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)

        if self.use_lstm_flag:
            self.lstm = nn.LSTM(
                self.model.config.hidden_size,
                lstm_dim,
                num_layers=lstm_num_layers,
                bidirectional=lstm_bidirectional_flag,
                batch_first=True,
                dropout=lstm_dropout_rate if lstm_num_layers > 1 else 0,
            )

        self.dropout = nn.Dropout(fc_dropout_rate)
        lstm_output_dim = lstm_dim * 2 if lstm_bidirectional_flag else lstm_dim
        lstm_output_dim = (
            lstm_output_dim * 2 if self.use_lstm_flag else self.model.config.hidden_size
        )

        self.fc = nn.Linear(lstm_output_dim, output_dim)

        if self.use_cnn_flag:
            self.cnn_list = list()
            for _ in range(1):
                self.cnn_list.append(
                    nn.Conv1d(
                        in_channels=lstm_output_dim,
                        out_channels=lstm_output_dim,
                        kernel_size=3,
                        padding=1,
                    )
                )
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(cnn_dropout_rate))
                self.cnn_list.append(nn.BatchNorm1d(lstm_output_dim))
            self.cnn = nn.Sequential(*self.cnn_list)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Dict,
    ):
        logits, _ = self.model(
            input_ids, attention_mask=attention_mask, return_dict=False
        )

        # print('o1', o1.size()) [32, 128, 768] == [batch size, sent len, emb dim]
        # o1 = o1.permute(1, 0, 2) # [sent len, batch size, emb dim]

        if self.use_lstm_flag:
            logits, (hidden, cell) = self.lstm(logits)
        if self.use_cnn_flag:
            logits = (
                self.cnn(logits.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
            )
            logits = logits.permute(1, 0, 2)
        logits = self.fc(logits)
        return logits

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
