#!/usr/bin/env python3
"""
Quick script to check pregnancy guide generation analytics
Run this anytime to see current stats
"""

from db import get_connection

def check_analytics():
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        print("\n" + "="*60)
        print("📊 PREGNANCY GUIDE GENERATION ANALYTICS")
        print("="*60 + "\n")
        
        # Get endpoint hit count
        cur.execute("""
            SELECT hit_count, last_hit 
            FROM endpoint_analytics 
            WHERE endpoint_name = 'pregnancy_guide_generation'
        """)
        result = cur.fetchone()
        
        if result:
            hit_count, last_hit = result
            print(f"🎯 Total Guides Generated: {hit_count}")
            print(f"⏰ Last Generated: {last_hit}")
        else:
            print("⚠️  No data found - endpoint not hit yet")
        
        print("\n" + "-"*60 + "\n")
        
        # Get additional stats
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(CASE WHEN language = 'urdu' THEN 1 END) as urdu,
                COUNT(CASE WHEN language = 'english' THEN 1 END) as english,
                AVG(pregnancy_week) as avg_week
            FROM pregnancy_guides
        """)
        
        stats = cur.fetchone()
        if stats:
            total, unique_users, urdu, english, avg_week = stats
            print(f"📝 Total Records in DB: {total}")
            print(f"👥 Unique Users: {unique_users}")
            print(f"🇵🇰 Urdu Guides: {urdu}")
            print(f"🇬🇧 English Guides: {english}")
            print(f"📅 Average Week: {round(avg_week, 1) if avg_week else 'N/A'}")
        
        print("\n" + "="*60 + "\n")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_analytics()
