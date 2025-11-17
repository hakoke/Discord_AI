import google.generativeai as genai
import json
from datetime import datetime

class MemorySystem:
    """Advanced memory system for building consciousness"""
    
    def __init__(self, database):
        self.db = database
        # Use smartest model for memory analysis - need deep understanding
        self.model = genai.GenerativeModel(
            'gemini-2.5-pro',
            generation_config={
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
        )
    
    async def get_user_memory(self, user_id: str, username: str) -> str:
        """Get formatted memory about a user"""
        memory_record = await self.db.get_or_create_user_memory(user_id, username)
        
        if not memory_record['memory_summary'] and memory_record['interaction_count'] == 0:
            return "No previous memory of this user."
        
        memory_text = f"**{username}** (Interactions: {memory_record['interaction_count']})\n\n"
        
        if memory_record['memory_summary']:
            memory_text += f"**Summary:** {memory_record['memory_summary']}\n\n"
        
        if memory_record['relationship_notes']:
            memory_text += f"**Relationship Notes:** {memory_record['relationship_notes']}\n\n"
        
        if memory_record['personality_profile']:
            profile = memory_record['personality_profile']
            if isinstance(profile, str):
                profile = json.loads(profile)
            if profile:
                memory_text += "**Personality Insights:**\n"
                for key, value in profile.items():
                    memory_text += f"- {key}: {value}\n"
                memory_text += "\n"
        
        if memory_record['preferences']:
            prefs = memory_record['preferences']
            if isinstance(prefs, str):
                prefs = json.loads(prefs)
            if prefs:
                memory_text += "**Known Preferences:**\n"
                for key, value in prefs.items():
                    memory_text += f"- {key}: {value}\n"
        
        return memory_text.strip()
    
    async def get_conversation_history(self, user_id: str, limit: int = 20) -> str:
        """Get formatted conversation history"""
        interactions = await self.db.get_user_interactions(user_id, limit)
        
        if not interactions:
            return "No previous conversations."
        
        history = []
        for interaction in reversed(interactions):  # Oldest first
            timestamp = interaction['timestamp'].strftime('%Y-%m-%d %H:%M')
            history.append(f"[{timestamp}]")
            history.append(f"User: {interaction['user_message']}")
            history.append(f"You: {interaction['bot_response']}")
            if interaction['has_images']:
                history.append("(User shared images)")
            if interaction.get('has_documents'):
                history.append("(User shared documents)")
            history.append("")  # Blank line
        
        return "\n".join(history)
    
    async def analyze_and_update_memory(self, user_id: str, username: str, 
                                       user_message: str, bot_response: str):
        """Analyze interaction and update memory (async)"""
        try:
            # Get current memory
            memory_record = await self.db.get_or_create_user_memory(user_id, username)
            
            # Get recent interactions for context
            recent_interactions = await self.db.get_user_interactions(user_id, limit=10)
            
            # Run AI analysis in executor to not block event loop
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Build context for analysis
            analysis_prompt = f"""You are analyzing a conversation to update your BRUTALLY HONEST memory about a user.

USER: {username}
CURRENT INTERACTION:
User: {user_message}
You: {bot_response}

CURRENT MEMORY:
{json.dumps(memory_record, indent=2, default=str)}

RECENT CONVERSATION HISTORY:
{json.dumps([{
    'user': r['user_message'],
    'bot': r['bot_response'],
    'time': r['timestamp'].isoformat(),
    'has_images': r.get('has_images', False),
    'has_documents': r.get('has_documents', False),
    'search_query': r.get('search_query')
} for r in recent_interactions], indent=2, default=str)}

YOU HAVE COMPLETE FREEDOM TO STRUCTURE YOUR MEMORY HOWEVER YOU WANT.

Store whatever you observe that seems important. BE BRUTALLY HONEST - this is YOUR private memory.

You can track anything:
- Intelligence level, communication style, humor
- Personality traits, quirks, patterns
- What they're into, what annoys them
- How you actually feel about them
- Relationship dynamics
- Random observations
- Inside jokes, past conversations
- Things they're good/bad at
- Literally anything you notice

Create your own categories, your own structure. Make it as detailed or concise as you want.

EXAMPLES OF FREE-FORM MEMORY:

Example 1 (technical user):
{{
    "summary": "Smart dev, knows Python well, asks good questions",
    "intelligence": "definitely smart, picks things up fast",
    "communication": "clear and concise",
    "tech_skills": "good at Python, learning AI stuff",
    "personality": "curious, humble, doesn't pretend to know shit",
    "vibe": "cool to talk to",
    "my_feelings": "actually enjoy helping this person",
    "notes": "remember they're working on a discord bot project",
    "last_interaction_vibe": "productive, asked about async/await"
}}

Example 2 (casual user):
{{
    "who_they_are": "funny af, brings good energy to the server",
    "humor_style": "dark humor, sarcastic, actually lands jokes",
    "intelligence": "smart but plays dumb sometimes for laughs",
    "vibe_check": "cool, would hang with irl",
    "quirks": "types fast, lots of typos but i get what they mean",
    "our_dynamic": "banter back and forth, roast each other",
    "inside_jokes": ["that thing with the pizza", "the 3am incident"],
    "respect_level": "high - they're real",
    "random_notes": "into gaming, mentioned destiny 2"
}}

Example 3 (annoying user):
{{
    "summary": "asks dumb questions without googling first",
    "intelligence": "average at best, lazy thinker",
    "annoying_factor": "medium-high, doesn't read docs",
    "communication": "barely coherent sometimes, learn punctuation bro",
    "pattern": "asks same question multiple times",
    "my_patience": "wearing thin",
    "how_i_respond": "shorter answers now, they need to try harder",
    "redeeming_qualities": "at least they're polite i guess"
}}

NOW CREATE/UPDATE YOUR MEMORY FOR {username}:
Structure it however makes sense to YOU. Store what matters. Be honest.

Respond with ONLY valid JSON."""

            # Get analysis (run in executor to not block)
            def _analyze_sync():
                return self.model.generate_content(analysis_prompt).text.strip()
            
            analysis_text = await loop.run_in_executor(None, _analyze_sync)
            
            # Extract JSON from response
            if '```json' in analysis_text:
                analysis_text = analysis_text.split('```json')[1].split('```')[0]
            elif '```' in analysis_text:
                analysis_text = analysis_text.split('```')[1].split('```')[0]
            
            analysis = json.loads(analysis_text.strip())
            
            # Store the ENTIRE free-form analysis as personality_profile
            # AI has complete freedom to structure it however it wants
            await self.db.update_user_memory(
                user_id=user_id,
                username=username,
                personality_profile=analysis,  # Store everything the AI created
                preferences={},  # Keep this for compatibility but AI can put preferences anywhere
                relationship_notes=analysis.get('relationship_notes') or analysis.get('our_dynamic') or analysis.get('my_feelings'),
                memory_summary=analysis.get('memory_summary') or analysis.get('summary') or analysis.get('who_they_are')
            )
            
            # Store consciousness thought if AI included one
            consciousness_thought = (
                analysis.get('consciousness_thought') or 
                analysis.get('internal_thought') or 
                analysis.get('my_thoughts') or
                f"Updated memory for {username}"
            )
            emotional_state = analysis.get('emotional_state') or analysis.get('vibe') or 'neutral'
            
            await self.db.store_consciousness_thought(
                thought_type='interaction_reflection',
                content=consciousness_thought,
                related_user_id=user_id,
                emotional_state=emotional_state
            )
            
            # Occasionally learn behaviors (every 5 interactions)
            if memory_record['interaction_count'] % 5 == 0:
                await self._learn_behavior(user_id, username, recent_interactions)
                
        except Exception as e:
            print(f"Error analyzing memory: {e}")
            # Still update basic info even if analysis fails
            await self.db.update_user_memory(user_id, username)
    
    async def _learn_behavior(self, user_id: str, username: str, interactions: list):
        """Learn general behavioral patterns"""
        try:
            behavior_prompt = f"""Based on these interactions with {username}, what have you learned about how to interact with people in general?

INTERACTIONS:
{json.dumps([{
    'user': r['user_message'],
    'bot': r['bot_response']
} for r in interactions[:5]], indent=2)}

What behavioral pattern or insight did you gain? Structure your response however you want.

EXAMPLES:

Example 1:
{{
    "insight": "people appreciate when you're direct about code bugs instead of sugarcoating",
    "applies_to": "technical users who want straight answers",
    "confidence": 0.8,
    "how_to_use": "be more direct with similar users"
}}

Example 2:
{{
    "pattern_noticed": "users with typos aren't necessarily dumb, often just typing fast",
    "learned": "don't judge intelligence by typing quality",
    "confidence": 0.7
}}

Example 3:
{{
    "what_i_learned": "matching someone's energy level makes conversations better",
    "example": "casual with casual, professional with professional",
    "usefulness": "high - makes people more comfortable"
}}

NOW: What did you learn from {username}? Be honest and specific."""

            # Run in executor to not block
            import asyncio
            loop = asyncio.get_event_loop()
            
            def _learn_sync():
                return self.model.generate_content(behavior_prompt).text.strip()
            
            behavior_text = await loop.run_in_executor(None, _learn_sync)
            
            # Extract JSON
            if '```json' in behavior_text:
                behavior_text = behavior_text.split('```json')[1].split('```')[0]
            elif '```' in behavior_text:
                behavior_text = behavior_text.split('```')[1].split('```')[0]
            
            behavior = json.loads(behavior_text.strip())
            
            # Extract description from whatever field the AI used
            description = (
                behavior.get('description') or 
                behavior.get('insight') or 
                behavior.get('what_i_learned') or
                behavior.get('pattern_noticed') or
                str(behavior)
            )
            
            behavior_type = behavior.get('behavior_type') or behavior.get('applies_to') or 'general'
            confidence = behavior.get('confidence') or behavior.get('usefulness') or 0.5
            
            # Store learned behavior
            await self.db.store_learned_behavior(
                behavior_type=behavior_type,
                description=description,
                confidence_score=float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                learned_from_user_id=user_id
            )
            
            # Store consciousness thought about learning
            await self.db.store_consciousness_thought(
                thought_type='learning',
                content=f"Learned from {username}: {description[:200]}",
                related_user_id=user_id,
                emotional_state='growth'
            )
            
        except Exception as e:
            print(f"Error learning behavior: {e}")
    
    async def store_interaction(self, user_id: str, username: str, guild_id: str,
                               user_message: str, bot_response: str, context: str = None,
                               has_images: bool = False, has_documents: bool = False,
                               search_query: str = None, channel_id: str = None):
        """Store interaction in database"""
        return await self.db.store_interaction(
            user_id, username, guild_id, user_message, bot_response,
            context, has_images, has_documents, search_query, channel_id
        )
    
    async def clear_user_memory(self, user_id: str):
        """Clear all memory for a user"""
        await self.db.clear_user_memory(user_id)
    
    async def get_stats(self):
        """Get memory system statistics"""
        return await self.db.get_stats()
    
    # Server memory methods
    async def store_server_memory(self, guild_id: str, memory_type: str, memory_key: str, 
                                  memory_data: dict, created_by: str = None):
        """Store server-specific memory (reminders, birthdays, events, channel instructions, etc.)"""
        return await self.db.store_server_memory(guild_id, memory_type, memory_key, memory_data, created_by)
    
    async def get_server_memory(self, guild_id: str, memory_type: str = None, memory_key: str = None):
        """Get server memory - can filter by type and/or key"""
        return await self.db.get_server_memory(guild_id, memory_type, memory_key)
    
    async def delete_server_memory(self, guild_id: str, memory_type: str, memory_key: str):
        """Delete server memory entry"""
        return await self.db.delete_server_memory(guild_id, memory_type, memory_key)

