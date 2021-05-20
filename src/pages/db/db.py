import datetime as dt
import sqlite3 as sql
from typing import Tuple


def db_creation():
    conn = sql.connect('pages/db/user_history.db')
    cur = conn.cursor()

    cur.executescript(
        '''
        CREATE TABLE IF NOT EXISTS UserHistory(
            id INTEGER PRIMARY KEY,
            raw_text TEXT NOT NULL,
            tok_text TEXT NOT NULL,
            parameters TEXT NOT NULL,
            result TEXT NOT NULL,
            ins_dt timestamp NOT NULL
        );'''
    )

    conn.commit()
    conn.close()


def db_clean():
    conn = sql.connect('pages/db/user_history.db')
    cur = conn.cursor()

    cur.executescript('''DELETE FROM UserHistory;''')

    conn.commit()
    conn.close()


def db_insert(raw_text: str, tok_text: str, parameters: str, result: str):
    conn = sql.connect('pages/db/user_history.db')
    cur = conn.cursor()

    user_history = (raw_text, tok_text, parameters, result, dt.datetime.now(tz=None))

    script = '''
        INSERT INTO UserHistory (raw_text, tok_text, parameters, result, ins_dt)
        VALUES(?, ?, ?, ?, ?);
    '''
    cur.execute(script, user_history)

    conn.commit()
    conn.close()


def db_select() -> Tuple:
    conn = sql.connect('pages/db/user_history.db')
    cur = conn.cursor()

    cur.execute(
        '''SELECT uh.raw_text, uh.tok_text, uh.parameters, uh.result, uh.ins_dt FROM UserHistory as uh;'''
    )
    rows = cur.fetchall()

    conn.commit()
    conn.close()

    return rows
