from typing import Dict, Iterable, Union

import torch
import transformers
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class NamedEntityRecognitionDataset(Dataset):
    """
    Dataset for NER task.
    """

    def __init__(
        self,
        texts: Iterable[Iterable[str]],
        tags: Iterable[Iterable[str]],
        tokenizer: Union[str, transformers.tokenization_utils.PreTrainedTokenizer],
        max_seq_len: int = None,
        lazy_mode: bool = True,
    ):
        self.tags = tags
        self.texts = texts

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, transformers.tokenization_utils.PreTrainedTokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError(
                "You pass wrong type of tokenizer. It should be a model name or PreTrainedTokenizer."
            )

        self.max_seq_len = max_seq_len
        self.length = len(texts)

        if self.max_seq_len < 3:
            raise ValueError("Max sequence length should be greather than 2")

        if not lazy_mode:
            pbar = tqdm(self.length, desc="tokenizing texts")
            self.encoded = [self._getitem_lazy(idx) for idx in pbar]
            del self.texts
            del self.tags

        self._getitem_fn = self._getitem_lazy if lazy_mode else self._getitem_encoded

    def __len__(self) -> int:
        return self.length

    def _getitem_encoded(self, index: int) -> Dict[str, torch.Tensor]:
        return torch.tensor(self.encoded[index])

    def _getitem_lazy(self, index: int) -> Dict[str, torch.Tensor]:
        sentence = self.texts[index]
        tag = self.tags[index]

        encode_ids = []
        target_tag = []
        for i, word in enumerate(sentence):
            words_piece_ids = self.tokenizer.encode(
                word,
                max_length=self.max_seq_len,
                add_special_tokes=False,
            )
            encode_ids.extend(words_piece_ids)
            target_tag.extend([tag[i]] * len(words_piece_ids))

        encode_ids = [101] + encode_ids[: self.max_seq_len - 2] + [102]
        target_tag = [0] + target_tag[: self.max_seq_len - 2] + [0]

        attention_mask = [1] * len(encode_ids)
        token_type_ids = [0] * len(encode_ids)

        padding_len = self.max_seq_len - len(encode_ids)
        encode_ids = encode_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            'encode_ids': torch.tensor(encode_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target_tag': torch.tensor(encode_ids, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._getitem_fn(index)
