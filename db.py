# db.py
import psycopg2
from config import config

def get_connection():
    params = config()
    conn = psycopg2.connect(**params)
    return conn
if __name__ == "__main__":
    try:
        conn = get_connection()
        print("Connection successful")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")