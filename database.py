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
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    context JSONB,
                    has_images BOOLEAN DEFAULT FALSE,
                    search_query TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
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
            
            # Create indexes for performance
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_consciousness_timestamp ON consciousness_stream(timestamp)')
            
            print("Database initialized successfully")
    
    async def store_interaction(self, user_id: str, username: str, guild_id: str, 
                                user_message: str, bot_response: str, context: str = None,
                                has_images: bool = False, search_query: str = None):
        """Store a conversation interaction"""
        async with self.pool.acquire() as conn:
            interaction_id = await conn.fetchval('''
                INSERT INTO interactions 
                (user_id, username, guild_id, user_message, bot_response, context, has_images, search_query)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            ''', user_id, username, guild_id, user_message, bot_response, context, has_images, search_query)
            
            return interaction_id
    
    async def get_user_interactions(self, user_id: str, limit: int = 20):
        """Get recent interactions for a user"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT user_message, bot_response, timestamp, has_images, search_query
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
            
            return {
                'total_interactions': total_interactions,
                'unique_users': unique_users,
                'memory_records': memory_records,
                'consciousness_entries': consciousness_entries,
                'learned_behaviors': learned_behaviors_count
            }
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()

