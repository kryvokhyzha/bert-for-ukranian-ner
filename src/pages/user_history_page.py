import pandas as pd
import streamlit as st

from pages.db import db_clean, db_select


def get_user_history_page():
    st.title('User history')

    rows = db_select()

    col = ('raw_text', 'tok_text', 'parameters', 'result', 'ins_dt')

    df = pd.DataFrame(rows, columns=col)

    columns = st.multiselect(
        label='What information do you want to display?', options=col, default=list(col)
    )

    if columns and df.shape[0]:
        st.write(df[columns])
        if st.button("Delete history"):
            db_clean()
    else:
        st.warning('User history is empty')
