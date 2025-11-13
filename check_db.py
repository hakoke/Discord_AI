"""Quick script to check if database tables exist"""
import asyncio
import asyncpg
import os

async def check_tables():
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    
    # Check all tables
    tables = await conn.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    
    print(f"\nFound {len(tables)} tables:")
    for table in tables:
        print(f"  - {table['table_name']}")
        
        # Count rows
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['table_name']}")
        print(f"    ({count} rows)")
    
    await conn.close()

if __name__ == '__main__':
    asyncio.run(check_tables())

