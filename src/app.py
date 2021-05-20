import streamlit as st

from pages import about_page, general_page, user_history_page
from pages.db import db_creation


if __name__ == '__main__':
    db_creation()

    options = {
        'General': general_page,
        'User History': user_history_page,
        'About': about_page,
    }

    st.sidebar.header("Please, choose page")
    page = st.sidebar.selectbox('', key='page_choice_box', options=list(options.keys()))

    options[page]()
