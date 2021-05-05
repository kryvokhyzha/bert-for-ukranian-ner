import json
from pathlib import Path
from typing import Dict

import streamlit as st
import streamlit.components.v1
import torch
from htbuilder import HtmlElement, div, span, styles
from htbuilder.units import em, px, rem

from data.dataset import NamedEntityRecognitionDataset
from models.NamedEntityRecognitionBertModel import NamedEntityRecognitionBertModel
from utils.helpers import get_config


def annotation(body, label="", background="#ddd", color="#333", **style):
    """Build an HtmlElement span object with the given body and annotation label.
    The end result will look something like this:
        [body | label]
    Parameters
    ----------
    body : string
        The string to put in the "body" part of the annotation.
    label : string
        The string to put in the "label" part of the annotation.
    background : string
        The color to use for the background "chip" containing this annotation.
    color : string
        The color to use for the body and label text.
    **style : dict
        Any CSS you want to use to customize the containing "chip".
    Examples
    --------
    Produce a simple annotation with default colors:
    >>> annotation("apple", "fruit")
    Produce an annotation with custom colors:
    >>> annotation("apple", "fruit", background="#FF0", color="black")
    Produce an annotation with crazy CSS:
    >>> annotation("apple", "fruit", background="#FF0", border="1px dashed red")
    """

    if "font_family" not in style:
        style["font_family"] = "sans-serif"

    return span(
        style=styles(
            background=background,
            border_radius=rem(0.33),
            color=color,
            padding=(rem(0.17), rem(0.67)),
            display="inline-flex",
            justify_content="center",
            align_items="center",
            **style,
        )
    )(
        body,
        span(
            style=styles(
                color=color,
                font_size=em(0.67),
                opacity=0.5,
                padding_left=rem(0.5),
                text_transform="uppercase",
                margin_bottom=px(-2),
            )
        )(label),
    )


def annotated_text(*args, **kwargs):
    """Writes test with annotations into your Streamlit app.
    Parameters
    ----------
    *args : str, tuple or htbuilder.HtmlElement
        Arguments can be:
        - strings, to draw the string as-is on the screen.
        - tuples of the form (main_text, annotation_text, background, color) where
          background and foreground colors are optional and should be an CSS-valid string such as
          "#aabbcc" or "rgb(10, 20, 30)"
        - HtmlElement objects in case you want to customize the annotations further. In particular,
          you can import the `annotation()` function from this module to easily produce annotations
          whose CSS you can customize via keyword arguments.
    """
    out = div(
        style=styles(
            font_family="sans-serif",
            line_height="1.5",
            color="white",
            font_size=px(16),
        )
    )

    for arg in args:
        if isinstance(arg, str):
            out(arg)

        elif isinstance(arg, HtmlElement):
            out(arg)

        elif isinstance(arg, tuple):
            out(annotation(*arg))

        else:
            raise Exception("Oh noes!")

    streamlit.components.v1.html(str(out), **kwargs)


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
    st.subheader('some explanation...')

    raw_text = st.text_area("Enter Text Here")
    raw_text = raw_text.split()

    test_dataset = NamedEntityRecognitionDataset(
        texts=[raw_text],
        tags=[[0] * len(raw_text)],
        tokenizer=CONFIG['model']['model_name'],
        max_seq_len=CONFIG['model']['max_seq_length'],
        lazy_mode=True,
    )

    if st.button('Submit'):
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
        # st.write(ann_text)
