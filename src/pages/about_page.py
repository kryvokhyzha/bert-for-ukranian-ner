from pathlib import Path

import streamlit as st

from utils.helpers import get_config


def get_about_page():
    st.title('About')

    PATH2ROOT = Path('..')
    PATH2CONFIG = Path(PATH2ROOT / 'configs')
    CONFIG = get_config(PATH2CONFIG / 'config.yml')

    st.header('Model parameters')
    st.json(CONFIG)

    st.header('Loss function plot')
    st.markdown(
        '![](https://github.com/kryvokhyzha/bert-for-ukranian-ner/blob/main/imgs/tb_focal_loss.png?raw=true)'
    )

    st.header('Accuracy plot')
    st.markdown(
        '![](https://github.com/kryvokhyzha/bert-for-ukranian-ner/blob/main/imgs/tb_accuracy.png?raw=true)'
    )
