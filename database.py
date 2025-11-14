import asyncpg
import os
from datetime import datetime
import json

class Database:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        self.pool = await asyncpg.create_pool(self.database_url)
        
        async with self.pool.acquire() as conn:
            # Interactions table - stores every conversation
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    guild_id TEXT,
                    channel_id TEXT,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    context JSONB,
                    has_images BOOLEAN DEFAULT FALSE,
                    has_documents BOOLEAN DEFAULT FALSE,
                    search_query TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            ''')
            await conn.execute('''
                ALTER TABLE interactions 
                ADD COLUMN IF NOT EXISTS has_documents BOOLEAN DEFAULT FALSE
            ''')
            await conn.execute('''
                ALTER TABLE interactions 
                ADD COLUMN IF NOT EXISTS channel_id TEXT
            ''')
            
            # User memory table - stores consciousness/personality per user
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    personality_profile JSONB,
                    preferences JSONB,
                    relationship_notes TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    first_interaction TIMESTAMP DEFAULT NOW(),
                    last_interaction TIMESTAMP DEFAULT NOW(),
                    memory_summary TEXT
                )
            ''')
            
            # Learned behaviors table - bot's evolving personality
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS learned_behaviors (
                    id SERIAL PRIMARY KEY,
                    behavior_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence_score FLOAT DEFAULT 0.5,
                    learned_from_user_id TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Consciousness stream - internal "thoughts" and realizations
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS consciousness_stream (
                    id SERIAL PRIMARY KEY,
                    thought_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    related_user_id TEXT,
                    related_interaction_id INTEGER,
                    emotional_state TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Server bans table - for managing server access
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS server_bans (
                    guild_id TEXT PRIMARY KEY,
                    banned_at TIMESTAMP DEFAULT NOW(),
                    ban_type TEXT NOT NULL CHECK (ban_type IN ('temporary', 'permanent')),
                    expires_at TIMESTAMP,
                    banned_by TEXT,
                    reason TEXT
                )
            ''')
            
            # Server structure table - stores channels, categories, etc.
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS server_structure (
                    guild_id TEXT PRIMARY KEY,
                    channels JSONB,
                    categories JSONB,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Create indexes for performance
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_consciousness_timestamp ON consciousness_stream(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_server_bans_expires ON server_bans(expires_at)')
            
            print("Database initialized successfully")
    
    async def store_interaction(self, user_id: str, username: str, guild_id: str, 
                                user_message: str, bot_response: str, context: str = None,
                                has_images: bool = False, has_documents: bool = False,
                                search_query: str = None, channel_id: str = None):
        """Store a conversation interaction"""
        async with self.pool.acquire() as conn:
            interaction_id = await conn.fetchval('''
                INSERT INTO interactions 
                (user_id, username, guild_id, channel_id, user_message, bot_response, context, has_images, has_documents, search_query)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            ''', user_id, username, guild_id, channel_id, user_message, bot_response, context, has_images, has_documents, search_query)
            
            return interaction_id
    
    async def get_user_interactions(self, user_id: str, limit: int = 20):
        """Get recent interactions for a user"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT user_message, bot_response, timestamp, has_images, has_documents, search_query
                FROM interactions
                WHERE user_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            ''', user_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_or_create_user_memory(self, user_id: str, username: str):
        """Get or create user memory record"""
        async with self.pool.acquire() as conn:
            # Try to get existing memory
            row = await conn.fetchrow('''
                SELECT * FROM user_memory WHERE user_id = $1
            ''', user_id)
            
            if row:
                return dict(row)
            
            # Create new memory record
            await conn.execute('''
                INSERT INTO user_memory (user_id, username, personality_profile, preferences)
                VALUES ($1, $2, $3, $4)
            ''', user_id, username, json.dumps({}), json.dumps({}))
            
            return {
                'user_id': user_id,
                'username': username,
                'personality_profile': {},
                'preferences': {},
                'relationship_notes': None,
                'interaction_count': 0,
                'memory_summary': None
            }
    
    async def update_user_memory(self, user_id: str, username: str, 
                                personality_profile: dict = None,
                                preferences: dict = None,
                                relationship_notes: str = None,
                                memory_summary: str = None):
        """Update user memory"""
        async with self.pool.acquire() as conn:
            update_parts = []
            params = []
            param_count = 1
            
            if personality_profile is not None:
                update_parts.append(f'personality_profile = ${param_count}')
                params.append(json.dumps(personality_profile))
                param_count += 1
            
            if preferences is not None:
                update_parts.append(f'preferences = ${param_count}')
                params.append(json.dumps(preferences))
                param_count += 1
            
            if relationship_notes is not None:
                update_parts.append(f'relationship_notes = ${param_count}')
                params.append(relationship_notes)
                param_count += 1
            
            if memory_summary is not None:
                update_parts.append(f'memory_summary = ${param_count}')
                params.append(memory_summary)
                param_count += 1
            
            update_parts.append(f'username = ${param_count}')
            params.append(username)
            param_count += 1
            
            update_parts.append('interaction_count = interaction_count + 1')
            update_parts.append('last_interaction = NOW()')
            
            params.append(user_id)
            
            query = f'''
                UPDATE user_memory 
                SET {', '.join(update_parts)}
                WHERE user_id = ${param_count}
            '''
            
            await conn.execute(query, *params)
    
    async def store_consciousness_thought(self, thought_type: str, content: str,
                                         related_user_id: str = None,
                                         related_interaction_id: int = None,
                                         emotional_state: str = None):
        """Store a consciousness stream thought"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO consciousness_stream 
                (thought_type, content, related_user_id, related_interaction_id, emotional_state)
                VALUES ($1, $2, $3, $4, $5)
            ''', thought_type, content, related_user_id, related_interaction_id, emotional_state)
    
    async def get_recent_consciousness(self, limit: int = 50):
        """Get recent consciousness stream"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM consciousness_stream
                ORDER BY timestamp DESC
                LIMIT $1
            ''', limit)
            
            return [dict(row) for row in rows]
    
    async def store_learned_behavior(self, behavior_type: str, description: str,
                                     confidence_score: float = 0.5,
                                     learned_from_user_id: str = None):
        """Store a learned behavior"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO learned_behaviors 
                (behavior_type, description, confidence_score, learned_from_user_id)
                VALUES ($1, $2, $3, $4)
            ''', behavior_type, description, confidence_score, learned_from_user_id)
    
    async def get_learned_behaviors(self, min_confidence: float = 0.5, limit: int = 100):
        """Get learned behaviors"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM learned_behaviors
                WHERE confidence_score >= $1
                ORDER BY confidence_score DESC, timestamp DESC
                LIMIT $2
            ''', min_confidence, limit)
            
            return [dict(row) for row in rows]
    
    async def clear_user_memory(self, user_id: str):
        """Clear all memory for a user"""
        async with self.pool.acquire() as conn:
            await conn.execute('DELETE FROM user_memory WHERE user_id = $1', user_id)
            await conn.execute('DELETE FROM interactions WHERE user_id = $1', user_id)
            await conn.execute('DELETE FROM consciousness_stream WHERE related_user_id = $1', user_id)
    
    async def get_stats(self):
        """Get database statistics"""
        async with self.pool.acquire() as conn:
            total_interactions = await conn.fetchval('SELECT COUNT(*) FROM interactions')
            unique_users = await conn.fetchval('SELECT COUNT(DISTINCT user_id) FROM interactions')
            memory_records = await conn.fetchval('SELECT COUNT(*) FROM user_memory')
            consciousness_entries = await conn.fetchval('SELECT COUNT(*) FROM consciousness_stream')
            learned_behaviors_count = await conn.fetchval('SELECT COUNT(*) FROM learned_behaviors')
            unique_servers = await conn.fetchval('SELECT COUNT(DISTINCT guild_id) FROM interactions WHERE guild_id IS NOT NULL')
            
            return {
                'total_interactions': total_interactions,
                'unique_users': unique_users,
                'memory_records': memory_records,
                'consciousness_entries': consciousness_entries,
                'learned_behaviors': learned_behaviors_count,
                'unique_servers': unique_servers
            }
    
    async def get_all_servers(self):
        """Get all servers (guilds) with stats"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT 
                    guild_id,
                    COUNT(*) as interaction_count,
                    COUNT(DISTINCT user_id) as unique_users,
                    MAX(timestamp) as last_activity,
                    MIN(timestamp) as first_activity
                FROM interactions
                WHERE guild_id IS NOT NULL
                GROUP BY guild_id
                ORDER BY interaction_count DESC
            ''')
            return [dict(row) for row in rows]
    
    async def get_server_stats(self, guild_id: str):
        """Get statistics for a specific server"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    MAX(timestamp) as last_activity,
                    MIN(timestamp) as first_activity,
                    SUM(CASE WHEN has_images THEN 1 ELSE 0 END) as image_interactions,
                    SUM(CASE WHEN has_documents THEN 1 ELSE 0 END) as document_interactions,
                    SUM(CASE WHEN search_query IS NOT NULL THEN 1 ELSE 0 END) as search_interactions
                FROM interactions
                WHERE guild_id = $1
            ''', guild_id)
            
            return dict(stats) if stats else None
    
    async def get_server_users(self, guild_id: str):
        """Get all users in a server with their memory"""
        async with self.pool.acquire() as conn:
            # Get users with interaction counts
            user_stats = await conn.fetch('''
                SELECT 
                    i.user_id,
                    i.username,
                    COUNT(*) as interaction_count,
                    MAX(i.timestamp) as last_interaction,
                    MIN(i.timestamp) as first_interaction,
                    SUM(CASE WHEN i.has_images THEN 1 ELSE 0 END) as image_count,
                    SUM(CASE WHEN i.has_documents THEN 1 ELSE 0 END) as document_count
                FROM interactions i
                WHERE i.guild_id = $1
                GROUP BY i.user_id, i.username
                ORDER BY interaction_count DESC
            ''', guild_id)
            
            # Get memory for each user
            users_with_memory = []
            for user in user_stats:
                user_dict = dict(user)
                memory = await conn.fetchrow('''
                    SELECT * FROM user_memory WHERE user_id = $1
                ''', user_dict['user_id'])
                
                if memory:
                    user_dict['memory'] = dict(memory)
                else:
                    user_dict['memory'] = None
                
                users_with_memory.append(user_dict)
            
            return users_with_memory
    
    async def get_server_interactions(self, guild_id: str, limit: int = 50):
        """Get recent interactions for a server"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM interactions
                WHERE guild_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            ''', guild_id, limit)
            return [dict(row) for row in rows]
    
    async def get_usage_over_time(self, guild_id: str = None, days: int = 30):
        """Get usage statistics over time for charts"""
        async with self.pool.acquire() as conn:
            if guild_id:
                rows = await conn.fetch(f'''
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as count
                    FROM interactions
                    WHERE guild_id = $1
                        AND timestamp >= NOW() - INTERVAL '{days} days'
                    GROUP BY DATE(timestamp)
                    ORDER BY date ASC
                ''', guild_id)
            else:
                rows = await conn.fetch(f'''
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as count
                    FROM interactions
                    WHERE timestamp >= NOW() - INTERVAL '{days} days'
                    GROUP BY DATE(timestamp)
                    ORDER BY date ASC
                ''')
            
            return [dict(row) for row in rows]
    
    async def get_top_users(self, guild_id: str = None, limit: int = 10):
        """Get top users by interaction count"""
        async with self.pool.acquire() as conn:
            if guild_id:
                rows = await conn.fetch('''
                    SELECT 
                        user_id,
                        username,
                        COUNT(*) as interaction_count
                    FROM interactions
                    WHERE guild_id = $1
                    GROUP BY user_id, username
                    ORDER BY interaction_count DESC
                    LIMIT $2
                ''', guild_id, limit)
            else:
                rows = await conn.fetch('''
                    SELECT 
                        user_id,
                        username,
                        COUNT(*) as interaction_count
                    FROM interactions
                    GROUP BY user_id, username
                    ORDER BY interaction_count DESC
                    LIMIT $1
                ''', limit)
            
            return [dict(row) for row in rows]
    
    async def ban_server(self, guild_id: str, ban_type: str, days: int = None, banned_by: str = None, reason: str = None):
        """Ban a server (temporary or permanent)"""
        async with self.pool.acquire() as conn:
            expires_at = None
            if ban_type == 'temporary' and days:
                from datetime import timedelta
                expires_at = datetime.now() + timedelta(days=days)
            
            await conn.execute('''
                INSERT INTO server_bans (guild_id, ban_type, expires_at, banned_by, reason)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (guild_id) 
                DO UPDATE SET 
                    ban_type = $2,
                    expires_at = $3,
                    banned_by = $4,
                    reason = $5,
                    banned_at = NOW()
            ''', guild_id, ban_type, expires_at, banned_by, reason)
    
    async def unban_server(self, guild_id: str):
        """Remove server ban"""
        async with self.pool.acquire() as conn:
            await conn.execute('DELETE FROM server_bans WHERE guild_id = $1', guild_id)
    
    async def check_server_ban(self, guild_id: str):
        """Check if server is banned, return ban info or None"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM server_bans 
                WHERE guild_id = $1
            ''', guild_id)
            
            if not row:
                return None
            
            ban_info = dict(row)
            
            # Check if temporary ban has expired
            if ban_info['ban_type'] == 'temporary' and ban_info['expires_at']:
                if ban_info['expires_at'] < datetime.now():
                    # Ban expired, remove it
                    await conn.execute('DELETE FROM server_bans WHERE guild_id = $1', guild_id)
                    return None
            
            return ban_info
    
    async def get_server_ban(self, guild_id: str):
        """Get server ban info (for dashboard)"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM server_bans 
                WHERE guild_id = $1
            ''', guild_id)
            
            if not row:
                return None
            
            ban_info = dict(row)
            
            # Check if temporary ban has expired
            if ban_info['ban_type'] == 'temporary' and ban_info['expires_at']:
                if ban_info['expires_at'] < datetime.now():
                    # Ban expired, remove it
                    await conn.execute('DELETE FROM server_bans WHERE guild_id = $1', guild_id)
                    return None
            
            return ban_info
    
    async def store_server_structure(self, guild_id: str, channels: list, categories: list):
        """Store server channel and category structure"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO server_structure (guild_id, channels, categories, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (guild_id) 
                DO UPDATE SET 
                    channels = $2,
                    categories = $3,
                    updated_at = NOW()
            ''', guild_id, json.dumps(channels), json.dumps(categories))
    
    async def get_server_structure(self, guild_id: str):
        """Get server structure (channels, categories)"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM server_structure 
                WHERE guild_id = $1
            ''', guild_id)
            
            if not row:
                return None
            
            structure = dict(row)
            if structure.get('channels'):
                structure['channels'] = json.loads(structure['channels']) if isinstance(structure['channels'], str) else structure['channels']
            if structure.get('categories'):
                structure['categories'] = json.loads(structure['categories']) if isinstance(structure['categories'], str) else structure['categories']
            
            return structure
    
    async def get_most_used_channel(self, guild_id: str):
        """Get the most used channel ID for a server based on interactions"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT channel_id, COUNT(*) as interaction_count
                FROM interactions
                WHERE guild_id = $1 AND channel_id IS NOT NULL
                GROUP BY channel_id
                ORDER BY interaction_count DESC
                LIMIT 1
            ''', guild_id)
            
            if row:
                return row['channel_id']
            return None
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()

