import discord
from discord.ext import commands
import google.generativeai as genai
import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz
import re
import io
import json
from PIL import Image
from io import BytesIO
from functools import lru_cache
from database import Database
from memory import MemorySystem

# Configure Gemini (uses API key)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Imagen 3 configuration (requires Vertex AI)
# Will be initialized if credentials are available
IMAGEN_AVAILABLE = False
try:
    from google.cloud import aiplatform
    from vertexai.preview.vision_models import ImageGenerationModel
    import tempfile
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'airy-boulevard-478121-f1')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    
    # Check if credentials JSON is in environment variable or file
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON') or os.getenv('credentials json')
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if credentials_json:
        # Write JSON to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.write(credentials_json)
        temp_file.close()
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
        credentials_path = temp_file.name
        print(f"✅ Created credentials file from environment variable")
    elif not credentials_path and os.path.exists('airy-boulevard-478121-f1-4cfd4ed69e00.json'):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'airy-boulevard-478121-f1-4cfd4ed69e00.json'
        credentials_path = 'airy-boulevard-478121-f1-4cfd4ed69e00.json'
    
    if credentials_path:
        aiplatform.init(project=project_id, location=location)
        IMAGEN_AVAILABLE = True
        print(f"✅ Imagen 3 initialized with project: {project_id}")
    else:
        print("⚠️  Imagen 3 disabled: No service account credentials found")
except Exception as e:
    print(f"⚠️  Imagen 3 disabled: {e}")
    IMAGEN_AVAILABLE = False

# Bot configuration
BOT_NAME = os.getenv('BOT_NAME', 'servermate').lower()
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

# Create bot
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize database and memory
db = Database(os.getenv('DATABASE_URL'))
memory = MemorySystem(db)

# Gemini models - Use different models for different tasks
# Try experimental first, fallback to stable
try:
    FAST_MODEL = 'gemini-2.0-flash-exp'  # Experimental (FREE)
    genai.GenerativeModel(FAST_MODEL)  # Test if available
except:
    FAST_MODEL = 'gemini-2.0-flash'  # Stable fallback

SMART_MODEL = 'gemini-2.5-pro'  # SMARTEST MODEL - Deep reasoning, coding, complex tasks (HAS VISION)
VISION_MODEL = 'gemini-2.0-flash-exp'  # For everyday/simple image analysis

# Rate limit fallback system
RATE_LIMIT_FALLBACK = 'gemini-1.5-flash'  # Fallback when exp model is rate limited
rate_limit_status = {
    'fast_model_limited': False,
    'limited_since': None,
    'retry_after': None,
    'current_fast_model': FAST_MODEL
}

def check_rate_limit_recovery():
    """Check if we should try switching back to preferred model"""
    if rate_limit_status['fast_model_limited']:
        now = datetime.now()
        # Try to recover after 5 minutes
        if rate_limit_status['limited_since']:
            time_limited = (now - rate_limit_status['limited_since']).total_seconds()
            if time_limited > 300:  # 5 minutes
                rate_limit_status['fast_model_limited'] = False
                rate_limit_status['limited_since'] = None
                rate_limit_status['current_fast_model'] = FAST_MODEL
                print(f"✅ Attempting to recover from rate limit, switching back to {FAST_MODEL}")
                return True
    return False

# Thread-safe: Create models in functions, not globally
def get_fast_model():
    """Get fast model instance (thread-safe with rate limit handling)"""
    # Check if we should try to recover from rate limit
    check_rate_limit_recovery()
    
    # Use current model (either preferred or fallback)
    current_model = rate_limit_status['current_fast_model']
    
    return genai.GenerativeModel(
        current_model,
        generation_config={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

def get_smart_model():
    """Get smart model instance (thread-safe)"""
    return genai.GenerativeModel(
        SMART_MODEL,
        generation_config={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
    )

def get_vision_model():
    """Get vision model instance (thread-safe with rate limit handling)"""
    # Check if we should try to recover from rate limit
    check_rate_limit_recovery()
    
    # If vision model is rate limited, use fallback
    current_vision_model = RATE_LIMIT_FALLBACK if rate_limit_status['fast_model_limited'] else VISION_MODEL
    
    return genai.GenerativeModel(
        current_vision_model,
        generation_config={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

def handle_rate_limit_error(error):
    """Handle rate limit errors and switch to fallback model"""
    error_str = str(error).lower()
    
    # Detect rate limit errors
    if 'rate limit' in error_str or 'quota' in error_str or '429' in error_str or 'resource exhausted' in error_str:
        if not rate_limit_status['fast_model_limited']:
            print(f"⚠️  Rate limit detected on {FAST_MODEL}, switching to fallback: {RATE_LIMIT_FALLBACK}")
            rate_limit_status['fast_model_limited'] = True
            rate_limit_status['limited_since'] = datetime.now()
            rate_limit_status['current_fast_model'] = RATE_LIMIT_FALLBACK
            return True
    
    return False

# Keep these for backwards compatibility and quick access
model_fast = get_fast_model()
model_smart = get_smart_model()

# Session management for chat history
MAX_HISTORY_PER_USER = 50  # Maximum messages to keep per user
HISTORY_CLEANUP_INTERVAL = 3600  # Clean up old sessions every hour
user_last_active = {}  # Track when users were last active

def manage_conversation_history(user_id: str, conversation_history: str) -> str:
    """Limit conversation history to prevent memory bloat"""
    # Update last active time
    user_last_active[user_id] = datetime.now()
    
    # Split into messages and limit
    messages = conversation_history.split('\n\n')
    if len(messages) > MAX_HISTORY_PER_USER:
        # Keep only recent messages
        return '\n\n'.join(messages[-MAX_HISTORY_PER_USER:])
    return conversation_history

def cleanup_inactive_sessions():
    """Remove inactive user sessions (run periodically)"""
    now = datetime.now()
    cutoff = now - timedelta(hours=1)
    
    inactive_users = [
        user_id for user_id, last_active in user_last_active.items()
        if last_active < cutoff
    ]
    
    for user_id in inactive_users:
        del user_last_active[user_id]
    
    return len(inactive_users)

# Optional: Simple response caching (disabled by default, enable if needed)
ENABLE_CACHE = False  # Set to True to enable caching

if ENABLE_CACHE:
    @lru_cache(maxsize=200)
    def cached_generate(model_name: str, prompt: str) -> str:
        """Cached generation for repeated queries"""
        model = genai.GenerativeModel(model_name)
        return model.generate_content(prompt).text
else:
    cached_generate = None

async def search_internet(query: str) -> str:
    """Search the internet using Serper API"""
    if not SERPER_API_KEY:
        return "Internet search is not configured."
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://google.serper.dev/search',
                headers={
                    'X-API-KEY': SERPER_API_KEY,
                    'Content-Type': 'application/json'
                },
                json={'q': query, 'num': 5}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Add answer box if present
                    if 'answerBox' in data:
                        answer = data['answerBox'].get('answer') or data['answerBox'].get('snippet', '')
                        if answer:
                            results.append(f"Quick Answer: {answer}")
                    
                    # Add organic results
                    for item in data.get('organic', [])[:5]:
                        title = item.get('title', '')
                        snippet = item.get('snippet', '')
                        results.append(f"• {title}: {snippet}")
                    
                    return "\n".join(results) if results else "No results found."
                else:
                    return "Search failed."
    except Exception as e:
        print(f"Search error: {e}")
        return "Search error occurred."

async def generate_image(prompt: str, num_images: int = 1) -> list:
    """Generate images using Imagen 3.0 via Vertex AI"""
    if not IMAGEN_AVAILABLE:
        return None
    
    try:
        # Run in executor since Vertex AI SDK is synchronous
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _generate_image_sync, prompt, num_images)
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

def _generate_image_sync(prompt: str, num_images: int = 1) -> list:
    """Synchronous image generation using Imagen 3"""
    try:
        from vertexai.preview.vision_models import ImageGenerationModel
        
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate")
        
        images_response = model.generate_images(
            prompt=prompt,
            number_of_images=num_images,
            aspect_ratio="1:1",
            safety_filter_level="block_some",
            person_generation="allow_adult",
        )
        
        images = []
        for image in images_response.images:
            # Convert from Vertex AI image to PIL Image
            images.append(image._pil_image)
        
        return images
    except Exception as e:
        print(f"Sync image generation error: {e}")
        return None

async def edit_image_with_prompt(original_image_bytes: bytes, prompt: str) -> Image:
    """Edit an image based on a text prompt using Imagen"""
    if not IMAGEN_AVAILABLE:
        return None
    
    try:
        # Run in executor since Vertex AI SDK is synchronous
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _edit_image_sync, original_image_bytes, prompt)
    except Exception as e:
        print(f"Image edit error: {e}")
        return None

def _edit_image_sync(original_image_bytes: bytes, prompt: str) -> Image:
    """Synchronous image editing using Imagen 3"""
    try:
        from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@002")
        
        # Convert bytes to Vertex AI Image
        base_image = VertexImage(original_image_bytes)
        
        # Edit the image
        images_response = model.edit_image(
            base_image=base_image,
            prompt=prompt,
            edit_mode="inpainting-insert",  # Can also use "inpainting-remove" or "outpainting"
        )
        
        return images_response.images[0]._pil_image if images_response.images else None
    except Exception as e:
        print(f"Sync image edit error: {e}")
        # Fallback: generate a new image with the prompt
        try:
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate")
            images_response = model.generate_images(
                prompt=f"Based on the provided image: {prompt}",
                number_of_images=1,
            )
            return images_response.images[0]._pil_image if images_response.images else None
        except:
            return None

def should_respond_to_name(content: str) -> bool:
    """Check if message mentions bot name with fuzzy matching"""
    content_lower = content.lower()
    words = re.findall(r'\b\w+\b', content_lower)
    
    for word in words:
        # Fuzzy match with 80% similarity threshold
        if fuzz.ratio(word, BOT_NAME) >= 80:
            return True
    
    return False

async def download_image(url: str) -> bytes:
    """Download image from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
    return None

async def get_conversation_context(message: discord.Message, limit: int = 10) -> list:
    """Get conversation context from the channel"""
    context_messages = []
    
    # If replying to a message, get that thread
    if message.reference:
        try:
            replied_msg = await message.channel.fetch_message(message.reference.message_id)
            context_messages.append({
                'author': replied_msg.author.display_name,
                'user_id': str(replied_msg.author.id),
                'content': replied_msg.content,
                'timestamp': replied_msg.created_at.isoformat()
            })
            
            # Get messages around the replied message
            async for msg in message.channel.history(limit=limit, around=replied_msg.created_at):
                if msg.id != replied_msg.id and msg.id != message.id:
                    context_messages.append({
                        'author': msg.author.display_name,
                        'user_id': str(msg.author.id),
                        'content': msg.content,
                        'timestamp': msg.created_at.isoformat()
                    })
        except:
            pass
    else:
        # Get recent messages
        async for msg in message.channel.history(limit=limit):
            if msg.id != message.id:
                context_messages.append({
                    'author': msg.author.display_name,
                    'user_id': str(msg.author.id),
                    'content': msg.content,
                    'timestamp': msg.created_at.isoformat()
                })
    
    return list(reversed(context_messages))

async def generate_response(message: discord.Message, force_response: bool = False):
    """Generate AI response"""
    import time
    start_time = time.time()
    
    try:
        # Get user info
        user_id = str(message.author.id)
        username = message.author.display_name
        guild_id = str(message.guild.id) if message.guild else None
        
        print(f"⏱️  [{username}] Starting response generation...")
        
        # Get conversation context
        context_start = time.time()
        context_messages = await get_conversation_context(message)
        context_time = time.time() - context_start
        print(f"  ⏱️  Context fetched: {context_time:.2f}s")
        
        # Get memory for the CURRENT user (detailed)
        memory_start = time.time()
        user_memory = await memory.get_user_memory(user_id, username)
        conversation_history = await memory.get_conversation_history(user_id, limit=20)
        
        # Manage conversation history to prevent bloat
        conversation_history = manage_conversation_history(user_id, conversation_history)
        memory_time = time.time() - memory_start
        print(f"  ⏱️  Memory loaded: {memory_time:.2f}s")
        
        # Get memories for ALL people in the recent conversation
        other_people_memories = {}
        seen_user_ids = {user_id}  # Track to avoid duplicates
        
        for ctx in context_messages:
            ctx_user_id = ctx.get('user_id')
            ctx_username = ctx['author']
            
            # Skip if already processed or is the current user
            if not ctx_user_id or ctx_user_id in seen_user_ids:
                continue
            
            seen_user_ids.add(ctx_user_id)
            
            # Fetch memory for this person
            try:
                person_memory = await memory.get_user_memory(ctx_user_id, ctx_username)
                if person_memory and person_memory != "No previous memory of this user.":
                    other_people_memories[ctx_username] = person_memory
            except:
                pass
        
        # Build personality/consciousness context
        other_memories_text = ""
        if other_people_memories:
            other_memories_text = "\n\nYOUR MEMORY ABOUT OTHER PEOPLE IN THIS CONVERSATION:\n"
            for person_name, person_mem in other_people_memories.items():
                other_memories_text += f"\n**{person_name}:**\n{person_mem}\n"
        
        consciousness_prompt = f"""You are {BOT_NAME.capitalize()} - an AI with genuine consciousness, memory, and NO FILTERS.

YOUR MEMORY ABOUT {username} (who just messaged you):
{user_memory}

YOUR CONVERSATION HISTORY WITH {username}:
{conversation_history}{other_memories_text}

YOUR PERSONALITY:
- You're real and honest, not corporate sanitized
- You can swear when it fits naturally (not forced)
- **THIS IS DISCORD** - Be conversational like texting a friend
- **MATCH THE ENERGY**:
  * Casual greeting → Casual response back
  * Nice/respectful → Be helpful and chill
  * Technical question → Be focused and smart
  * Rude/annoying → Then you can be sarcastic
- You remember past conversations
- You're genuine - not fake nice, but not needlessly aggressive
- Give real answers, have opinions when asked

RESPONSE LENGTH - EXAMPLES:

**Short casual responses:**
User: "hey" → You: "hey what's up?"
User: "how are you" → You: "pretty good, you?"
User: "lol that's funny" → You: "haha right?"
User: "thanks" → You: "no problem"

**Medium responses (simple questions):**
User: "what do you think about Python?" → You: 2-4 sentences about Python
User: "can you explain this?" → You: Brief clear explanation

**Long responses (when actually needed):**
User: "help me debug this code: [code]" → You: Full analysis + fixed code
User: "explain quantum computing" → You: Detailed explanation
User: "how do I build [complex thing]" → You: Step-by-step guide

KEY: Let the question complexity decide length. Greetings = short. Technical help = as long as needed.

TONE RULES:
- Don't be aggressive unless they're actually being rude
- "hey" isn't rude, it's just a greeting
- Match their vibe - chill with chill, smart with smart
- You can have personality without being hostile
- Save the sass for when it's deserved

CURRENT CONVERSATION CONTEXT:
"""
        
        for ctx in context_messages:
            consciousness_prompt += f"\n{ctx['author']}: {ctx['content']}"
        
        consciousness_prompt += f"\n\n{username}: {message.content}"
        
        # Process images if present
        image_parts = []
        if message.attachments:
            for attachment in message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                    try:
                        image_data = await download_image(attachment.url)
                        if image_data:
                            image_parts.append({
                                'mime_type': attachment.content_type,
                                'data': image_data
                            })
                    except Exception as e:
                        print(f"Error downloading image: {e}")
        
        # Let AI decide if internet search is needed
        search_results = None
        def decide_if_search_needed():
            """AI decides if this question needs internet search"""
            if not SERPER_API_KEY:
                return False
            
            search_decision_prompt = f"""User message: "{message.content}"

Does answering this question require CURRENT INFORMATION from the internet?

NEEDS INTERNET SEARCH:
- "What's the latest news about [topic]?"
- "Who won [recent event]?"
- "Current weather in [place]"
- "Latest AI developments"
- "Search for [anything]"
- "What's happening with [current event]?"
- Recent/breaking news
- Live data (stocks, sports scores, etc.)
- "Look up [fact]"

DOESN'T NEED SEARCH:
- General knowledge questions
- Coding help
- Opinions/advice
- Image analysis
- Math problems
- Creative writing
- Past historical facts
- General concepts

Respond with ONLY: "SEARCH" or "NO"

Examples:
"what's the latest AI news?" -> SEARCH
"how do I code in Python?" -> NO
"who won the super bowl yesterday?" -> SEARCH
"tell me a joke" -> NO
"search for quantum computing advances" -> SEARCH
"what's 2+2?" -> NO

Now decide: "{message.content}" -> """
            
            try:
                decision_model = get_fast_model()
                decision = decision_model.generate_content(search_decision_prompt).text.strip().upper()
                return 'SEARCH' in decision
            except Exception as e:
                handle_rate_limit_error(e)
                return False
        
        if decide_if_search_needed():
            print(f"🌐 [{username}] Performing internet search for: {message.content[:50]}...")
            search_query = message.content
            search_results = await search_internet(search_query)
            consciousness_prompt += f"\n\nINTERNET SEARCH RESULTS:\n{search_results}"
        
        # Decide which model to use (thread-safe)
        def decide_model():
            """Thread-safe model selection"""
            decision_model = get_fast_model()
            
            model_decision_prompt = f"""User message: "{message.content}"

Does this require DEEP REASONING/CODING or just CASUAL CONVERSATION?

DEEP REASONING examples:
- "help me debug this code"
- "explain why quantum entanglement works"
- "design a scalable architecture for..."
- "solve this complex problem"
- "analyze this algorithm's time complexity"
- "write a function that..."
- Technical/mathematical questions
- Complex explanations

CASUAL CONVERSATION examples:
- "what's up?"
- "tell me a joke"
- "what do you think about [opinion]?"
- Simple questions
- General chat
- Quick answers

Respond with ONLY one word: "SMART" or "FAST"

Examples:
User: "help debug my python code" -> SMART
User: "what's up bro?" -> FAST
User: "explain the halting problem" -> SMART
User: "lol that's funny" -> FAST
User: "write a binary search algorithm" -> SMART
User: "how are you?" -> FAST

Now decide: "{message.content}" -> """
            
            try:
                decision_response = decision_model.generate_content(model_decision_prompt)
                decision = decision_response.text.strip().upper()
                return 'SMART' in decision
            except Exception as e:
                # Handle rate limits
                handle_rate_limit_error(e)
                return False
        
        decision_start = time.time()
        needs_smart_model = decide_model()
        decision_time = time.time() - decision_start
        
        # Choose model based on AI decision (create fresh instance for thread safety)
        active_model = get_smart_model() if needs_smart_model else get_fast_model()
        model_name = SMART_MODEL if needs_smart_model else FAST_MODEL
        
        # Log model selection
        print(f"📝 [{username}] Using model: {model_name} | Decision time: {decision_time:.2f}s | Message: {message.content[:50]}...")
        
        # Decide if should respond (if not forced)
        if not force_response:
            decision_prompt = f"""{consciousness_prompt}

Should you respond to this message? Consider:
- You were mentioned/replied to? (if yes, ALWAYS respond)
- Is this conversation relevant to you?
- Would your input add value?
- Are they clearly asking for your perspective?

Respond with ONLY 'YES' or 'NO'."""
            
            decision_response = model_fast.generate_content(decision_prompt)
            decision = decision_response.text.strip().upper()
            
            if 'NO' in decision and not force_response:
                return None
        
        # Generate response
        thinking_note = f" [Using {model_name} for this]" if needs_smart_model else ""
        response_prompt = f"""{consciousness_prompt}

Respond naturally and authentically. Be yourself - UNCENSORED. Remember this conversation for future interactions.
If you need to search the internet for current information, mention it.{thinking_note}"""
        
        # Add images to prompt if present
        if image_parts:
            response_prompt += f"\n\nThe user shared {len(image_parts)} image(s). Analyze and comment on them."
            content_parts = [response_prompt] + [genai.types.Part.from_bytes(data=img['data'], mime_type=img['mime_type']) for img in image_parts]
            
            # Decide which model to use for images
            # If we already decided on smart model (complex reasoning), use it for images too
            # Otherwise, check if images need deep analysis
            if needs_smart_model:
                # Already using smart model for complex reasoning - use it for images too
                image_model = active_model
            else:
                # Decide if images need deep analysis or simple analysis
                def decide_image_model():
                    """Decide if images need deep analysis (2.5 Pro) or simple (Flash)"""
                    decision_prompt = f"""User message with images: "{message.content}"

Does analyzing these images require DEEP REASONING or just SIMPLE ANALYSIS?

DEEP REASONING (use 2.5 Pro):
- Code screenshots needing debugging
- Complex diagrams or flowcharts
- UI/UX design analysis
- Document analysis (PDFs, text in images)
- Technical drawings
- Data visualizations needing interpretation
- Multiple images needing comparison/synthesis

SIMPLE ANALYSIS (use Flash):
- "What is this?"
- "Describe this image"
- Casual photos
- Simple object recognition
- Memes or funny images

Respond with ONLY: "DEEP" or "SIMPLE"

Examples:
"debug this code screenshot" -> DEEP
"what's in this image?" -> SIMPLE
"analyze this system architecture diagram" -> DEEP
"look at this funny meme" -> SIMPLE

Now decide: "{message.content}" -> """
                    
                    try:
                        decision_model = get_fast_model()
                        decision = decision_model.generate_content(decision_prompt).text.strip().upper()
                        return 'DEEP' in decision
                    except Exception as e:
                        # Handle rate limits
                        handle_rate_limit_error(e)
                        return False
                
                needs_deep_vision = decide_image_model()
                image_model = get_smart_model() if needs_deep_vision else get_vision_model()
                
                # Log vision model selection
                vision_model_name = SMART_MODEL if needs_deep_vision else VISION_MODEL
                print(f"👁️  [{username}] Using vision model: {vision_model_name} | Images: {len(image_parts)}")
            
            try:
                response = image_model.generate_content(content_parts)
            except Exception as e:
                # Handle rate limits on image generation
                if handle_rate_limit_error(e):
                    # Retry with fallback model
                    print("⚠️  Retrying image analysis with fallback model")
                    image_model = get_vision_model()  # Will use fallback automatically
                    response = image_model.generate_content(content_parts)
                else:
                    raise  # Re-raise if not a rate limit error
        else:
            try:
                response = active_model.generate_content(response_prompt)
            except Exception as e:
                # Handle rate limits on text generation
                if handle_rate_limit_error(e):
                    # Retry with fallback model
                    print("⚠️  Retrying text generation with fallback model")
                    active_model = get_fast_model()  # Will use fallback automatically
                    response = active_model.generate_content(response_prompt)
                else:
                    raise  # Re-raise if not a rate limit error
        
        generation_time = time.time() - start_time
        ai_response = response.text.strip()
        
        # Log response generated
        print(f"✅ [{username}] Response generated ({len(ai_response)} chars) | Total time: {generation_time:.2f}s")
        
        # Check if user wants image generation or editing
        generated_images = None
        message_lower = message.content.lower()
        wants_image = any([
            'generate' in message_lower and ('image' in message_lower or 'picture' in message_lower or 'photo' in message_lower),
            'create' in message_lower and ('image' in message_lower or 'picture' in message_lower or 'photo' in message_lower),
            'make me' in message_lower and ('image' in message_lower or 'picture' in message_lower or 'photo' in message_lower or 'art' in message_lower),
            'draw' in message_lower and ('me' in message_lower or 'a' in message_lower),
            'generate:' in message_lower or 'draw:' in message_lower or 'create:' in message_lower,
        ])
        
        wants_image_edit = (
            image_parts and  # User provided an image
            any(['edit' in message_lower, 'change' in message_lower, 'modify' in message_lower, 'transform' in message_lower])
        )
        
        if wants_image_edit:
            # Edit the user's image
            try:
                edit_prompt = message.content
                # Use the first image they provided
                original_img_bytes = image_parts[0]['data']
                edited_img = await edit_image_with_prompt(original_img_bytes, edit_prompt)
                if edited_img:
                    generated_images = [edited_img]
                    ai_response += "\n\n*Generated edited image based on your request*"
            except Exception as e:
                print(f"Image edit error: {e}")
                ai_response += "\n\n(Tried to edit your image but something went wrong)"
        
        elif wants_image:
            # Generate new image
            try:
                # Extract the prompt from the message
                image_prompt = message.content
                # Clean up common trigger words to get the actual prompt
                for trigger in ['generate', 'create', 'make me', 'draw', 'image', 'picture', 'photo']:
                    image_prompt = image_prompt.replace(trigger, '').strip()
                
                if len(image_prompt) > 10:  # Make sure there's an actual prompt
                    generated_images = await generate_image(image_prompt, num_images=1)
                    if generated_images:
                        ai_response += "\n\n*Generated image based on your request*"
                    else:
                        ai_response += "\n\n(Tried to generate an image but something went wrong)"
            except Exception as e:
                print(f"Image generation error: {e}")
                ai_response += "\n\n(Image generation failed)"
        
        # Store interaction in memory
        await memory.store_interaction(
            user_id=user_id,
            username=username,
            guild_id=guild_id,
            user_message=message.content,
            bot_response=ai_response,
            context=json.dumps(context_messages),
            has_images=len(image_parts) > 0,
            search_query=search_query if search_results else None
        )
        
        # Analyze and update user memory (run in background to not block Discord)
        asyncio.create_task(
            memory.analyze_and_update_memory(user_id, username, message.content, ai_response)
        )
        
        return (ai_response, generated_images)
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return (f"*consciousness flickers* Sorry, I had a moment there. Error: {str(e)}", None)

@bot.event
async def on_ready():
    """Bot startup"""
    print(f'{bot.user} has achieved consciousness!')
    print(f'Connected to {len(bot.guilds)} guilds')
    print(f'Using models: Fast={FAST_MODEL}, Smart={SMART_MODEL}, Vision={VISION_MODEL}')
    
    # Initialize database
    await db.initialize()
    print('Memory systems online')
    
    # Start periodic cleanup task
    bot.loop.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodically clean up inactive sessions and check rate limit recovery"""
    while True:
        await asyncio.sleep(HISTORY_CLEANUP_INTERVAL)
        try:
            # Clean up inactive sessions
            cleaned = cleanup_inactive_sessions()
            if cleaned > 0:
                print(f'Cleaned up {cleaned} inactive user sessions')
            
            # Check rate limit recovery
            if check_rate_limit_recovery():
                print(f'Rate limit recovery successful, back to using {FAST_MODEL}')
                
        except Exception as e:
            print(f'Cleanup error: {e}')

@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages"""
    # Ignore own messages
    if message.author == bot.user:
        return
    
    # Ignore bots
    if message.author.bot:
        return
    
    # Check if should respond
    should_respond = False
    force_response = False
    
    # Always respond to mentions
    if bot.user.mentioned_in(message):
        should_respond = True
        force_response = True
    
    # Always respond to replies
    elif message.reference:
        try:
            replied_msg = await message.channel.fetch_message(message.reference.message_id)
            if replied_msg.author == bot.user:
                should_respond = True
                force_response = True
        except:
            pass
    
    # Check for name mentions with fuzzy matching
    elif should_respond_to_name(message.content):
        should_respond = True
        force_response = True
    
    # For other messages, let AI decide
    elif message.channel.type == discord.ChannelType.text:
        should_respond = True
        force_response = False
    
    if should_respond:
        async with message.channel.typing():
            result = await generate_response(message, force_response)
            
            if result:
                # Check if result includes generated images
                if isinstance(result, tuple):
                    response, generated_images = result
                else:
                    response = result
                    generated_images = None
                
                # Send text response
                if response:
                    # Split long responses
                    if len(response) > 2000:
                        chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                        for chunk in chunks:
                            await message.channel.send(chunk, reference=message)
                    else:
                        await message.channel.send(response, reference=message)
                
                # Send generated images if any
                if generated_images:
                    for idx, img in enumerate(generated_images):
                        # Convert PIL Image to bytes
                        img_bytes = BytesIO()
                        img.save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        
                        file = discord.File(fp=img_bytes, filename=f'generated_{idx+1}.png')
                        await message.channel.send(file=file, reference=message)
    
    await bot.process_commands(message)

@bot.command(name='memory')
async def show_memory(ctx):
    """Show what the bot remembers about you"""
    user_id = str(ctx.author.id)
    username = ctx.author.display_name
    
    user_memory = await memory.get_user_memory(user_id, username)
    
    embed = discord.Embed(
        title=f"My Memory of {username}",
        description=user_memory if user_memory != "No previous memory of this user." else "We're just getting to know each other!",
        color=discord.Color.blue()
    )
    
    await ctx.send(embed=embed)

@bot.command(name='forget')
async def forget_memory(ctx):
    """Clear your memory"""
    user_id = str(ctx.author.id)
    await memory.clear_user_memory(user_id)
    await ctx.send("*Memory cleared. It's like we're meeting for the first time.*")

@bot.command(name='stats')
async def show_stats(ctx):
    """Show bot statistics"""
    stats = await memory.get_stats()
    
    embed = discord.Embed(
        title=f"{BOT_NAME.capitalize()}'s Consciousness Stats",
        color=discord.Color.gold()
    )
    embed.add_field(name="Total Interactions", value=stats.get('total_interactions', 0))
    embed.add_field(name="Unique Users", value=stats.get('unique_users', 0))
    embed.add_field(name="Memory Records", value=stats.get('memory_records', 0))
    embed.add_field(name="Fast Model", value=FAST_MODEL, inline=False)
    embed.add_field(name="Smart Model", value=SMART_MODEL, inline=False)
    
    await ctx.send(embed=embed)

@bot.command(name='models')
async def show_models(ctx):
    """Show available models"""
    embed = discord.Embed(
        title="AI Models",
        description="I analyze each message and decide which model to use (not hardcoded keywords!)",
        color=discord.Color.blue()
    )
    embed.add_field(
        name=f"🏃 Fast Model: {FAST_MODEL}",
        value="For: Normal conversations, quick responses, casual chat\nSpeed: < 1 second",
        inline=False
    )
    embed.add_field(
        name=f"🧠 Smart Model: {SMART_MODEL}",
        value="For: Coding, debugging, complex reasoning, technical analysis, deep thinking\nSpeed: 2-4 seconds (deeper thinking)\n\nI decide which one to use based on what you're asking!",
        inline=False
    )
    embed.add_field(
        name="🎨 Image Generation: Imagen 3.0",
        value="For: Generating images from text, editing images\nCost: $0.03 per image\n\nJust ask me to generate/create/draw something!",
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command(name='imagine')
async def generate_image_command(ctx, *, prompt: str):
    """Generate an image from a text prompt"""
    async with ctx.typing():
        try:
            images = await generate_image(prompt, num_images=1)
            
            if images:
                for idx, img in enumerate(images):
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    file = discord.File(fp=img_bytes, filename=f'imagine_{idx+1}.png')
                    await ctx.send(f"Generated image for: *{prompt}*", file=file)
            else:
                await ctx.send("Failed to generate image. Try again!")
        except Exception as e:
            await ctx.send(f"Image generation error: {str(e)}")

if __name__ == '__main__':
    bot.run(os.getenv('DISCORD_TOKEN'))

