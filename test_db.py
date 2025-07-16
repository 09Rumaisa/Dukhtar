#!/usr/bin/env python3
"""
Test script to check database connection and tables for Dukhtar Pregnancy Tracker
"""

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test database connection and check tables"""
    try:
        # Get database connection parameters
        db_host = 'localhost'
        db_port = '5432'
        db_name = 'Dukhtar'  # Note: capital D
        db_user = 'postgres'
        db_password = 'anaya'
        
        print(f"Connecting to database: {db_name} on {db_host}:{db_port}")
        print(f"User: {db_user}")
        
        # Connect to database
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        
        print("✅ Database connection successful!")
        
        # Create cursor
        cur = conn.cursor()
        
        # Check if tables exist
        tables_to_check = [
            'users',
            'pregnancy_tracking',
            'health_metrics', 
            'appointments',
            'baby_milestones'
        ]
        
        print("\n📋 Checking tables:")
        for table in tables_to_check:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            
            exists = cur.fetchone()[0]
            status = "✅" if exists else "❌"
            print(f"  {status} {table}")
        
        # Check users table
        print("\n👥 Checking users table:")
        cur.execute("SELECT COUNT(*) FROM users;")
        user_count = cur.fetchone()[0]
        print(f"  Users count: {user_count}")
        
        if user_count > 0:
            cur.execute("SELECT user_id, name, email FROM users LIMIT 5;")
            users = cur.fetchall()
            print("  Sample users:")
            for user in users:
                print(f"    ID: {user[0]}, Name: {user[1]}, Email: {user[2]}")
        
        # Check baby_milestones table
        print("\n👶 Checking baby_milestones table:")
        cur.execute("SELECT COUNT(*) FROM baby_milestones;")
        milestone_count = cur.fetchone()[0]
        print(f"  Milestones count: {milestone_count}")
        
        if milestone_count > 0:
            cur.execute("SELECT week_number, milestone_title FROM baby_milestones ORDER BY week_number LIMIT 5;")
            milestones = cur.fetchall()
            print("  Sample milestones:")
            for milestone in milestones:
                print(f"    Week {milestone[0]}: {milestone[1]}")
        
        # Check pregnancy_tracking table
        print("\n🤰 Checking pregnancy_tracking table:")
        cur.execute("SELECT COUNT(*) FROM pregnancy_tracking;")
        tracking_count = cur.fetchone()[0]
        print(f"  Pregnancy tracking records: {tracking_count}")
        
        # Check foreign key constraints
        print("\n🔗 Checking foreign key constraints:")
        cur.execute("""
            SELECT 
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
            AND tc.table_name IN ('pregnancy_tracking', 'health_metrics', 'appointments');
        """)
        
        foreign_keys = cur.fetchall()
        print("  Foreign key constraints:")
        for fk in foreign_keys:
            print(f"    {fk[0]}.{fk[1]} -> {fk[2]}.{fk[3]}")
        
        # Close cursor and connection
        cur.close()
        conn.close()
        
        print("\n✅ Database test completed successfully!")
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_connection() 