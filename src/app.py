import streamlit as st

from pages import general_page


if __name__ == '__main__':
    options = {
        'General': general_page,
    }

    st.sidebar.header("Please, choose page")
    page = st.sidebar.selectbox('', key='page_choice_box', options=list(options.keys()))

    options[page]()
