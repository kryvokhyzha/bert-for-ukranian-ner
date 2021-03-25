from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup

from data.dataset import NamedEntityRecognitionDataset
from models.NamedEntityRecognitionBertModel import NamedEntityRecognitionBertModel
from utils.helpers import get_config
from utils.runners import eval_fn, train_fn


PATH2ROOT = Path('..')
PATH2CONFIG = Path(PATH2ROOT / 'configs')
CONFIG = get_config(PATH2CONFIG / 'config.yml')
PATH2COURPUS = Path(PATH2ROOT / CONFIG['data']['path_to_corpus_folder'])
MODEL_NAME = 'youscan/ukr-roberta-base'


if __name__ == "__main__":
    data = joblib.load(PATH2ROOT / CONFIG['data']['path_to_preproc_data'])
    texts = data['text'].values.tolist()
    tags = data['tags'].values.tolist()

    num_tag = len(set([item for sublist in tags for item in sublist]))

    train_texts, val_texts, train_tags, val_tags = model_selection.train_test_split(
        texts,
        tags,
        random_state=CONFIG['general']['seed'],
        test_size=CONFIG['general']['test_size'],
    )

    train_dataset = NamedEntityRecognitionDataset(
        texts=train_texts,
        tags=train_tags,
        tokenizer=MODEL_NAME,
        max_seq_len=4,
        lazy_mode=True,
    )
    val_dataset = NamedEntityRecognitionDataset(
        texts=val_texts,
        tags=val_tags,
        tokenizer=MODEL_NAME,
        max_seq_len=4,
        lazy_mode=True,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG['training']['train_batch_size'], num_workers=1
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CONFIG['training']['valid_batch_size'], num_workers=1
    )

    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    model = NamedEntityRecognitionBertModel(
        pretrained_model_name=MODEL_NAME, num_tag=num_tag
    )
    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(
        len(train_texts)
        / CONFIG['training']['train_batch_size']
        * CONFIG['training']['num_epochs']
    )
    optimizer = AdamW(optimizer_parameters, lr=CONFIG['training']['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(CONFIG['training']['num_epochs']):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = eval_fn(val_data_loader, model, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), PATH2ROOT / CONFIG['data']['path_to_logdir'])
            best_loss = test_loss

    print('Best loss:', best_loss)
