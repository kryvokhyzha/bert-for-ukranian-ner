import json
from pathlib import Path
from typing import Dict

import streamlit as st
import torch

from data.dataset import NamedEntityRecognitionDataset
from models.NamedEntityRecognitionBertModel import NamedEntityRecognitionBertModel
from utils.annotation import annotated_text
from utils.constants import TEXT_EXAMPLES
from utils.helpers import get_config


@st.cache
def get_configs():
    PATH2ROOT = Path('..')
    PATH2CONFIG = Path(PATH2ROOT / 'configs')

    return PATH2ROOT, PATH2CONFIG, get_config(PATH2CONFIG / 'config.yml')


@st.cache(allow_output_mutation=True)
def get_model(PATH2ROOT: Path, CONFIG: Dict, device, output_dim):
    model = NamedEntityRecognitionBertModel(
        pretrained_model_name=CONFIG['model']['model_name'],
        output_dim=output_dim,
        lstm_dim=CONFIG['model']['lstm_dim'],
        lstm_num_layers=CONFIG['model']['lstm_num_layers'],
        lstm_dropout_rate=CONFIG['model']['lstm_dropout_rate'],
        lstm_bidirectional_flag=bool(CONFIG['model']['lstm_bidirectional_flag']),
        cnn_dropout_rate=CONFIG['model']['cnn_dropout_rate'],
        fc_dropout_rate=CONFIG['model']['fc_dropout_rate'],
        use_lstm_flag=bool(CONFIG['model']['use_lstm_flag']),
        use_cnn_flag=bool(CONFIG['model']['use_cnn_flag']),
    )

    model.load_state_dict(
        torch.load(
            PATH2ROOT / CONFIG['data']['path_to_logdir'] / 'best.pth', map_location=device
        )
    )

    model.freeze()
    return model


def get_general_page():
    PATH2ROOT, PATH2CONFIG, CONFIG = get_configs()

    with open(PATH2CONFIG / 'target_mapper.json', 'r') as file:
        tag_map = json.load(file)

    inv_tag_map = {val[0]: (key, val[1]) for key, val in tag_map.items()}

    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if CONFIG['general']['device'] == 'auto'
        else torch.device(CONFIG['general']['device'])
    )
    model = get_model(PATH2ROOT, CONFIG, device, len(tag_map.keys()))

    st.title('Named Entity Recognition using BERT')
    st.subheader(
        'This simple application provide a way to visualize token classification task (NER)💡'
    )

    text_key = st.selectbox(
        'select some example or type your text', list(TEXT_EXAMPLES.keys())
    )

    raw_text = st.text_area("enter text here", value=TEXT_EXAMPLES[text_key])
    raw_text = raw_text.split()

    test_dataset = NamedEntityRecognitionDataset(
        texts=[raw_text],
        tags=[[0] * len(raw_text)],
        tokenizer=CONFIG['model']['model_name'],
        max_seq_len=CONFIG['model']['max_seq_length'],
        lazy_mode=True,
    )

    if st.button('Submit🔥'):
        model.eval()

        with torch.no_grad():
            for d in test_dataset:
                for k, v in d.items():
                    d[k] = v.unsqueeze(0).to(device)

                outputs = model(**d)

        ann_text = list(
            zip(
                [
                    test_dataset.tokenizer.decode(t)
                    for t in d['input_ids'][0][1 : d['attention_mask'].view(-1).sum() - 1]
                ],
                outputs.argmax(2)
                .cpu()
                .numpy()
                .reshape(-1)[1 : d['attention_mask'].view(-1).sum() - 1],
            )
        )

        annotated_text(
            *[
                text + " "
                if class_ == tag_map['O'][0]
                else (text + " ", inv_tag_map[class_][0], inv_tag_map[class_][1])
                for text, class_ in ann_text
            ]
        )
