import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask import g

DATABASE = 'users.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # Access columns by name
    return db

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(query, args)
    conn.commit()
    cur.close()

def register_user(username, password):
    error = None
    if not username:
        error = 'Username is required.'
    elif not password:
        error = 'Password is required.'
    elif query_db('SELECT id FROM users WHERE username = ?', (username,), one=True) is not None:
        error = f'User {username} is already registered.'

    if error is None:
        hashed_password = generate_password_hash(password)
        execute_db('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    return error

def authenticate_user(username, password):
    user = query_db('SELECT * FROM users WHERE username = ?', (username,), one=True)
    if user is None:
        return None, 'Incorrect username.'
    elif not check_password_hash(user['password'], password):
        return None, 'Incorrect password.'
    return user, None