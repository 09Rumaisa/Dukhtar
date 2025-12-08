# db.py
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection():
    """
    Get database connection.
    Priority:
    1. DATABASE_URL (production - full connection string)
    2. Individual POSTGRES_* variables (local development from .env)
    3. database.ini (fallback for legacy config)
    """
    
    # Check if DATABASE_URL is set (production environment like Render/Railway)
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        # Production: use DATABASE_URL
        print("✓ Using DATABASE_URL for connection")
        # Handle both postgresql:// and postgresql+asyncpg:// formats
        if 'postgresql+asyncpg://' in database_url:
            database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
        conn = psycopg2.connect(database_url)
    
    # Check if individual PostgreSQL variables are set (local .env)
    elif os.getenv('POSTGRES_USER'):
        print("✓ Using POSTGRES_* environment variables for connection")
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            database=os.getenv('POSTGRES_DB', 'Dukhtar'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
    
    else:
        # Fallback: use database.ini (legacy)
        print("⚠ Using database.ini for connection (consider moving to .env)")
        from config import config
        params = config()
        conn = psycopg2.connect(**params)
    
    return conn

if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("Testing Database Connection")
        print("="*60 + "\n")
        
        conn = get_connection()
        
        # Test the connection
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        
        print(f"✅ Connection successful!")
        print(f"📊 PostgreSQL version: {db_version[0]}\n")
        
        # Show current database
        cur.execute("SELECT current_database();")
        current_db = cur.fetchone()
        print(f"📁 Connected to database: {current_db[0]}\n")
        
        cur.close()
        conn.close()
        
        print("="*60)
        print("✅ Database connection test passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}\n")
        import traceback
        traceback.print_exc()