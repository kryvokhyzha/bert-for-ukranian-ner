from pathlib import Path

import joblib
import torch
from catalyst.utils import prepare_cudnn, set_global_seed
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup

from data.dataset import NamedEntityRecognitionDataset
from models.loss_fn.FocalLossCustom import FocalLossCustom
from models.NamedEntityRecognitionBertModel import NamedEntityRecognitionBertModel
from utils.callbacks.AccuracyCallbackCustom import AccuracyCallbackCustom
from utils.helpers import get_config, remove_dir
from utils.runners.CustomRunner import CustomRunner


PATH2ROOT = Path('..')
PATH2CONFIG = Path(PATH2ROOT / 'configs')
CONFIG = get_config(PATH2CONFIG / 'config.yml')
PATH2CORPUS = Path(PATH2ROOT / CONFIG['data']['path_to_corpus_folder'])
MODEL_NAME = CONFIG['model']['model_name']


if __name__ == "__main__":
    data = joblib.load(PATH2ROOT / CONFIG['data']['path_to_preproc_data'])
    texts = data['text'].values.tolist()
    tags = data['tags'].values.tolist()

    tag_map = {'O': 0, 'LOC': 1, 'MISC': 2, 'ORG': 3, 'PERS': 4}

    for i in range(len(tags)):
        for j in range(len(tags[i])):
            tags[i][j] = tag_map[tags[i][j]]

    num_tag = len(tag_map.keys())

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
        max_seq_len=CONFIG['model']['max_seq_length'],
        lazy_mode=True,
    )

    val_dataset = NamedEntityRecognitionDataset(
        texts=val_texts,
        tags=val_tags,
        tokenizer=MODEL_NAME,
        max_seq_len=CONFIG['model']['max_seq_length'],
        lazy_mode=True,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['train_batch_size'],
        num_workers=CONFIG['training']['num_workers'],
        shuffle=True,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['valid_batch_size'],
        num_workers=CONFIG['training']['num_workers'],
    )

    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NamedEntityRecognitionBertModel(
        pretrained_model_name=MODEL_NAME,
        output_dim=num_tag,
        lstm_dim=CONFIG['model']['lstm_dim'],
        lstm_num_layers=CONFIG['model']['lstm_num_layers'],
        lstm_dropout_rate=CONFIG['model']['lstm_dropout_rate'],
        lstm_bidirectional_flag=bool(CONFIG['model']['lstm_bidirectional_flag']),
        cnn_dropout_rate=CONFIG['model']['cnn_dropout_rate'],
        fc_droupout_rate=CONFIG['model']['fc_droupout_rate'],
        use_cnn_flag=bool(CONFIG['model']['use_cnn_flag']),
    )
    model = model.to(device)

    model.freeze()
    param_optimizer = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    model.unfreeze()

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        'gamma',
        'beta',
        'final_layer_norm.weight',
    ]
    param_optimizer = [
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
    optimizer = AdamW(param_optimizer, lr=float(CONFIG['training']['learning_rate']))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['training']['train_batch_size'] * 2,
        num_training_steps=num_train_steps,
    )

    if bool(CONFIG['training']['is_deterministic']):
        set_global_seed(CONFIG["general"]["seed"])
        prepare_cudnn(deterministic=True)

    remove_dir(PATH2ROOT / CONFIG["training"]["log_dir"])
    (PATH2ROOT / CONFIG["training"]["log_dir"]).mkdir()
    (PATH2ROOT / CONFIG["training"]["log_dir"] / '.gitkeep').touch()

    loaders = {"train": train_data_loader, "valid": val_data_loader}

    model.freeze()

    runner = CustomRunner(
        custom_metrics={
            'accuracy': AccuracyCallbackCustom(),
        },
    )

    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=FocalLossCustom(gamma=CONFIG['training']['focal_loss_gamma']),
        loaders=loaders,
        num_epochs=CONFIG['training']['num_epochs'],
        logdir=PATH2ROOT / CONFIG["training"]["log_dir"],
        load_best_on_end=True,
        verbose=True,
        timeit=False,  # you can pass True to measure execution time of different parts of train process
    )
    model.unfreeze()

    torch.save(
        model.state_dict(), PATH2ROOT / CONFIG['data']['path_to_logdir'] / 'best.pth'
    )
