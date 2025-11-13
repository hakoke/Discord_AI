"""
Database initialization script
Run this manually if needed, but bot.py will auto-initialize on startup
"""
import asyncio
import os
from database import Database

async def main():
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        return
    
    print("Initializing database...")
    db = Database(database_url)
    await db.initialize()
    print("Database initialized successfully!")
    
    # Show stats
    stats = await db.get_stats()
    print("\nDatabase Statistics:")
    print(f"Total Interactions: {stats['total_interactions']}")
    print(f"Unique Users: {stats['unique_users']}")
    print(f"Memory Records: {stats['memory_records']}")
    print(f"Consciousness Entries: {stats['consciousness_entries']}")
    print(f"Learned Behaviors: {stats['learned_behaviors']}")
    
    await db.close()

if __name__ == '__main__':
    asyncio.run(main())

