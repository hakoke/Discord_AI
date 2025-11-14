import discord
from discord.ext import commands
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
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
import tempfile
from database import Database
from memory import MemorySystem

# Configure Gemini (uses API key)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Imagen configuration (requires Vertex AI)
# Will be initialized if credentials are available
IMAGEN_AVAILABLE = False
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'airy-boulevard-478121-f1')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    
    # Allow runtime override of Imagen model choices via env vars
    def _build_candidate_list(env_key: str, defaults: list[str]) -> list[str]:
        """Helper to build ordered list of model candidates with env override."""
        override = os.getenv(env_key, '')
        candidates = []
        if override:
            candidates.extend([name.strip() for name in override.split(',') if name.strip()])
        candidates.extend([name for name in defaults if name not in candidates])
        return candidates
    
    IMAGEN_GENERATE_MODELS = _build_candidate_list(
        'IMAGEN_GENERATE_MODELS',
        [
            'imagen-4.0-ultra-generate-001',  # Latest GA generate model
            'imagen-3.0-generate',            # Stable legacy generate model
        ]
    )
    IMAGEN_EDIT_MODELS = _build_candidate_list(
        'IMAGEN_EDIT_MODELS',
        [
            'imagegeneration@002',            # Official editing/upscaling model
        ]
    )

    
    # PRIORITY 1: Check for local credentials file (committed to private repo)
    credentials_path = None
    if os.path.exists('airy-boulevard-478121-f1-4cfd4ed69e00.json'):
        credentials_path = 'airy-boulevard-478121-f1-4cfd4ed69e00.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        print(f"✅ Using local credentials file from repo: {credentials_path}")
        
        # Verify the file is valid JSON
        try:
            with open(credentials_path, 'r') as f:
                verify_json = json.load(f)
                print(f"   - JSON keys: {list(verify_json.keys())}")
                print(f"   - Project ID: {verify_json.get('project_id', 'NOT FOUND')}")
                print(f"   - Client email: {verify_json.get('client_email', 'NOT FOUND')}")
        except Exception as verify_error:
            print(f"   ⚠️ Could not verify credentials file: {verify_error}")
    
    # PRIORITY 2: Check environment variable for credentials path
    elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        print(f"✅ Using credentials path from environment: {credentials_path}")
    
    # PRIORITY 3: Check for credentials JSON string (legacy support)
    elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON') or os.getenv('credentials json'):
        credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON') or os.getenv('credentials json')
        print(f"⚠️  Found credentials JSON in environment variable (not recommended)")
        print(f"   - Attempting to parse and write to temp file...")
        
        try:
            # Parse JSON first to validate and remove control characters
            parsed_json = json.loads(credentials_json)
            
            # Write properly formatted JSON to temp file
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            json.dump(parsed_json, temp_file, indent=2)
            temp_file.close()
            
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
            credentials_path = temp_file.name
            print(f"   ✅ Created valid credentials file: {temp_file.name}")
            print(f"   - Project ID: {parsed_json.get('project_id', 'NOT FOUND')}")
        except json.JSONDecodeError as json_error:
            print(f"   ❌ Failed to parse credentials JSON: {json_error}")
            credentials_path = None
    
    if credentials_path:
        print(f"🔧 Initializing Vertex AI at startup...")
        print(f"   - Project: {project_id}")
        print(f"   - Location: {location}")
        print(f"   - Credentials: {credentials_path}")
        
        vertexai.init(project=project_id, location=location)
        IMAGEN_AVAILABLE = True
        print(f"✅ Imagen 3 initialized successfully!")
        print(f"   - IMAGEN_AVAILABLE = True")
    else:
        print("⚠️  Imagen 3 disabled: No service account credentials found")
except Exception as e:
    print(f"⚠️  Imagen 3 disabled: {e}")
    IMAGEN_AVAILABLE = False

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

ENABLE_GEMINI_FILE_UPLOADS = _env_flag('ENABLE_GEMINI_FILE_UPLOADS', False)
GEMINI_INLINE_IMAGE_LIMIT = _env_int('GEMINI_INLINE_IMAGE_LIMIT', 3 * 1024 * 1024)
MAX_GENERATED_IMAGES = _env_int('MAX_GENERATED_IMAGES', 4)

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
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
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
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
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
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
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
        print(f"⚠️  [IMAGE GEN] Imagen not available, skipping image generation")
        return None
    
    try:
        print(f"🚀 [IMAGE GEN] Async wrapper called, running in executor...")
        # Run in executor since Vertex AI SDK is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _generate_image_sync, prompt, num_images)
        print(f"🏁 [IMAGE GEN] Executor completed, returning result")
        return result
    except Exception as e:
        print(f"❌ [IMAGE GEN] Async wrapper error: {e}")
        import traceback
        print(f"❌ [IMAGE GEN] Async traceback:\n{traceback.format_exc()}")
        return None

def _generate_image_sync(prompt: str, num_images: int = 1) -> list:
    """Synchronous image generation using Imagen 3"""
    try:
        print(f"🎨 [IMAGE GEN] Starting image generation for prompt: '{prompt[:100]}...'")
        
        import vertexai
        from vertexai.preview.vision_models import ImageGenerationModel
        print(f"✅ [IMAGE GEN] Vertex AI modules imported successfully")
        
        # Re-initialize vertexai in this thread context
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'airy-boulevard-478121-f1')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        print(f"🔑 [IMAGE GEN] Initializing Vertex AI...")
        print(f"   - Project: {project_id}")
        print(f"   - Location: {location}")
        print(f"   - Credentials path: {credentials_path}")
        
        # Verify credentials file exists
        if credentials_path:
            import pathlib
            cred_file = pathlib.Path(credentials_path)
            print(f"   - File exists: {cred_file.exists()}")
            if cred_file.exists():
                print(f"   - File size: {cred_file.stat().st_size} bytes")
                print(f"   - File readable: {os.access(credentials_path, os.R_OK)}")
        else:
            print(f"   ⚠️  WARNING: No credentials path set!")
        
        vertexai.init(project=project_id, location=location)
        print(f"✅ [IMAGE GEN] Vertex AI initialized successfully")
        
        model = None
        last_error = None
        for model_name in IMAGEN_GENERATE_MODELS:
            try:
                print(f"🔄 [IMAGE GEN] Loading Imagen model: {model_name}")
                model = ImageGenerationModel.from_pretrained(model_name)
                print(f"✅ [IMAGE GEN] Model loaded successfully: {model_name}")
                break
            except Exception as model_error:
                last_error = model_error
                print(f"   ⚠️  Model '{model_name}' not available: {model_error}")
        
        if not model:
            raise last_error or Exception("No Imagen generate model could be loaded.")
        
        print(f"📡 [IMAGE GEN] Calling Imagen API (generating {num_images} image(s))...")
        images_response = model.generate_images(
            prompt=prompt,
            number_of_images=num_images,
            aspect_ratio="1:1",
            safety_filter_level="block_none",
            person_generation="allow_all",
        )
        print(f"✅ [IMAGE GEN] API call successful, received response")
        
        print(f"🖼️  [IMAGE GEN] Converting {len(images_response.images)} image(s) to PIL format...")
        images = []
        for idx, image in enumerate(images_response.images):
            # Convert from Vertex AI image to PIL Image
            images.append(image._pil_image)
            print(f"   ✓ Image {idx + 1}/{len(images_response.images)} converted")
        
        print(f"🎉 [IMAGE GEN] Successfully generated {len(images)} image(s)!")
        return images
    except Exception as e:
        print(f"❌ [IMAGE GEN] Error occurred: {type(e).__name__}")
        print(f"❌ [IMAGE GEN] Error message: {str(e)}")
        import traceback
        print(f"❌ [IMAGE GEN] Full traceback:\n{traceback.format_exc()}")
        return None

async def edit_image_with_prompt(original_image_bytes: bytes, prompt: str) -> Image:
    """Edit an image based on a text prompt using Imagen"""
    if not IMAGEN_AVAILABLE:
        print(f"⚠️  [IMAGE EDIT] Imagen not available, skipping image editing")
        return None
    
    try:
        print(f"🚀 [IMAGE EDIT] Async wrapper called, running in executor...")
        # Run in executor since Vertex AI SDK is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _edit_image_sync, original_image_bytes, prompt)
        print(f"🏁 [IMAGE EDIT] Executor completed, returning result")
        return result
    except Exception as e:
        print(f"❌ [IMAGE EDIT] Async wrapper error: {e}")
        import traceback
        print(f"❌ [IMAGE EDIT] Async traceback:\n{traceback.format_exc()}")
        return None

def _edit_image_sync(original_image_bytes: bytes, prompt: str) -> Image:
    """Synchronous image editing using Imagen 3"""
    try:
        print(f"✏️  [IMAGE EDIT] Starting image editing with prompt: '{prompt[:100]}...'")
        
        import vertexai
        from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
        print(f"✅ [IMAGE EDIT] Vertex AI modules imported successfully")
        
        # Re-initialize vertexai in this thread context
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'airy-boulevard-478121-f1')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        print(f"🔑 [IMAGE EDIT] Initializing Vertex AI...")
        print(f"   - Project: {project_id}")
        print(f"   - Location: {location}")
        print(f"   - Credentials path: {credentials_path}")
        
        # Verify credentials file exists
        if credentials_path:
            import pathlib
            cred_file = pathlib.Path(credentials_path)
            print(f"   - File exists: {cred_file.exists()}")
            if cred_file.exists():
                print(f"   - File size: {cred_file.stat().st_size} bytes")
                print(f"   - File readable: {os.access(credentials_path, os.R_OK)}")
        else:
            print(f"   ⚠️  WARNING: No credentials path set!")
        
        vertexai.init(project=project_id, location=location)
        print(f"✅ [IMAGE EDIT] Vertex AI initialized successfully")
        
        model = None
        last_error = None
        for model_name in IMAGEN_EDIT_MODELS:
            try:
                print(f"🔄 [IMAGE EDIT] Loading Imagen editing model: {model_name}")
                model = ImageGenerationModel.from_pretrained(model_name)
                print(f"✅ [IMAGE EDIT] Model loaded successfully: {model_name}")
                break
            except Exception as model_error:
                last_error = model_error
                print(f"   ⚠️  Model '{model_name}' not available: {model_error}")
        
        if not model:
            raise last_error or Exception("No Imagen editing model could be loaded.")
        
        # Convert bytes to Vertex AI Image
        print(f"🖼️  [IMAGE EDIT] Converting base image ({len(original_image_bytes)} bytes)...")
        base_image = VertexImage(original_image_bytes)
        print(f"✅ [IMAGE EDIT] Base image converted")
        
        # Edit the image
        print(f"📡 [IMAGE EDIT] Calling Imagen edit API...")
        images_response = model.edit_image(
            base_image=base_image,
            prompt=prompt,
            edit_mode="inpainting-insert",  # Can also use "inpainting-remove" or "outpainting"
        )
        print(f"✅ [IMAGE EDIT] API call successful")
        
        result = images_response.images[0]._pil_image if images_response.images else None
        if result:
            print(f"🎉 [IMAGE EDIT] Successfully edited image!")
        else:
            print(f"⚠️  [IMAGE EDIT] No images returned from API")
        return result
    except Exception as e:
        print(f"❌ [IMAGE EDIT] Error occurred: {type(e).__name__}")
        print(f"❌ [IMAGE EDIT] Error message: {str(e)}")
        import traceback
        print(f"❌ [IMAGE EDIT] Full traceback:\n{traceback.format_exc()}")
        
        # Fallback: generate a new image with the prompt
        print(f"🔄 [IMAGE EDIT] Attempting fallback to image generation...")
        try:
            vertexai.init(project=os.getenv('GOOGLE_CLOUD_PROJECT', 'airy-boulevard-478121-f1'), 
                         location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'))
            print(f"✅ [IMAGE EDIT] Fallback: Vertex AI re-initialized")
            
            fallback_generate_model = IMAGEN_GENERATE_MODELS[0]
            print(f"🔄 [IMAGE EDIT] Fallback: Loading generate model {fallback_generate_model}")
            model = ImageGenerationModel.from_pretrained(fallback_generate_model)
            print(f"✅ [IMAGE EDIT] Fallback: Model loaded")
            
            images_response = model.generate_images(
                prompt=f"Based on the provided image: {prompt}",
                number_of_images=1,
                safety_filter_level="block_none",
                person_generation="allow_all",
            )
            print(f"✅ [IMAGE EDIT] Fallback: Image generated")
            return images_response.images[0]._pil_image if images_response.images else None
        except Exception as fallback_error:
            print(f"❌ [IMAGE EDIT] Fallback also failed: {fallback_error}")
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

def _clamp_value(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))

def _heuristic_requested_image_count(message: str, default: int = 1) -> int:
    """Heuristic to extract requested image count from message text."""
    lowered = message.lower()
    number_words = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'couple': 2,
        'double': 2,
        'pair': 2,
        'few': 3,
    }
    
    match = re.search(r'(\d+)\s*(?:image|images|picture|pictures|photo|photos|render|renders|pic|pics)\b', lowered)
    if match:
        count = int(match.group(1))
        return _clamp_value(count, 1, MAX_GENERATED_IMAGES)
    
    for word, count in number_words.items():
        if re.search(rf'\b{word}\b.*\b(?:image|images|picture|pictures|photo|photos|render|renders|pic|pics)\b', lowered):
            return _clamp_value(count, 1, MAX_GENERATED_IMAGES)
        if re.search(r'\b(?:image|images|picture|pictures|photo|photos|render|renders|pic|pics)\b.*\b' + re.escape(word) + r'\b', lowered):
            return _clamp_value(count, 1, MAX_GENERATED_IMAGES)
    
    return _clamp_value(default, 1, MAX_GENERATED_IMAGES)

def ai_decide_image_count(message: discord.Message) -> int:
    """Use AI to pick how many images to generate."""
    heuristics_guess = _heuristic_requested_image_count(message.content, default=1)
    message_text = (message.content or "").strip()
    metadata = {
        "message": message_text,
        "author_display_name": message.author.display_name,
        "channel_id": str(message.channel.id),
        "attachments_count": len(message.attachments),
        "heuristic_guess": heuristics_guess,
        "max_images": MAX_GENERATED_IMAGES,
    }
    prompt = f"""A user asked for image generation. Decide how many images to produce.

Rules:
- Pick an integer between 1 and {MAX_GENERATED_IMAGES}.
- Use more than one image when the user explicitly asks for multiple variations, angles, options, or uses words like "two", "couple", "few", "some", "multiple".
- Use at least 2 if they request comparisons, alternatives, or a gallery.
- Stick to 1 if they don't hint at needing multiple results.
- Consider the heuristic suggestion but override it when your judgment differs.

Return ONLY a JSON object like:
{{"image_count": 3}}

User context:
{json.dumps(metadata, ensure_ascii=False, indent=2)}"""

    try:
        decision_model = get_fast_model()
        decision_response = decision_model.generate_content(prompt)
        raw_text = (decision_response.text or "").strip()
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if match:
            parsed = json.loads(match.group(0))
            image_count = int(parsed.get("image_count", heuristics_guess))
            return _clamp_value(image_count, 1, MAX_GENERATED_IMAGES)
    except Exception as e:
        print(f"Image count decision error: {e}")
    
    return heuristics_guess

def is_small_talk_message(content: str, has_question: bool, wants_image: bool, wants_image_edit: bool, has_attachments: bool) -> bool:
    """Heuristic to detect light conversation where short replies are preferred."""
    text = content.strip()
    if not text:
        return False
    lowered = text.lower()
    
    if has_question or '?' in text:
        return False
    if wants_image or wants_image_edit or has_attachments:
        return False
    if len(lowered) > 200:
        return False
    
    small_talk_keywords = [
        'hey', 'hi', 'hello', 'yo', 'sup', 'nice', 'thanks', 'thank you',
        'lol', 'haha', 'good job', 'looks good', 'appreciate it', 'awesome',
        'great work', 'that was cool', 'fire', 'lit', 'dope', 'sick', 'love it',
        'good stuff', 'nice work', 'good looking', 'amazing', 'perfect'
    ]
    if any(phrase in lowered for phrase in small_talk_keywords):
        return True
    
    # If it's a single short sentence without actionable verbs, treat as small talk
    if len(lowered.split()) <= 12 and not re.search(r'\b(can|should|how|what|why|when|where|explain|help|fix|generate|make)\b', lowered):
        return True
    
    return False

def ai_decide_reply_style(message: discord.Message, wants_image: bool, wants_image_edit: bool, has_attachments: bool) -> str:
    """Use AI to choose reply style (SMALL_TALK, NORMAL, DETAILED)."""
    message_text = (message.content or "").strip()
    has_question_mark = '?' in message_text
    heuristics_guess = 'NORMAL'
    if is_small_talk_message(message_text, has_question_mark, wants_image, wants_image_edit, has_attachments):
        heuristics_guess = 'SMALL_TALK'
    elif len(message_text) > 220 or has_question_mark or wants_image or wants_image_edit:
        heuristics_guess = 'DETAILED'
    
    try:
        decision_model = get_fast_model()
        guidance = {
            "message": message_text,
            "attachments_count": len(message.attachments),
            "is_reply": bool(message.reference),
            "mentions_bot": bot.user.mentioned_in(message),
            "wants_image": wants_image,
            "wants_image_edit": wants_image_edit,
            "has_question_mark": has_question_mark,
            "author_display_name": message.author.display_name,
        }
        decision_prompt = f"""You are choosing the ideal reply style for a helpful Discord assistant.

Possible styles:
- SMALL_TALK: quick acknowledgements, casual praise, greetings, brief chit-chat. One or two short sentences.
- NORMAL: typical questions or comments that deserve a helpful but moderately sized reply.
- DETAILED: complex questions, troubleshooting, planning, or anything that benefits from step-by-step guidance or deep analysis.

Consider the message and metadata (JSON below) and respond with a single JSON object like:
{{"style": "NORMAL"}}
Allowed values for "style": SMALL_TALK, NORMAL, DETAILED.

If the user only says thanks, praise, or a simple greeting, choose SMALL_TALK.
If they ask for significant help, instructions, or problem-solving, choose DETAILED.
Otherwise choose NORMAL.

Message and metadata:
{json.dumps(guidance, ensure_ascii=False, indent=2)}

Return ONLY the JSON object."""

        decision_response = decision_model.generate_content(decision_prompt)
        raw_text = (decision_response.text or "").strip()
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if match:
            data = json.loads(match.group(0))
            style = str(data.get('style', '')).upper()
            if style in {'SMALL_TALK', 'NORMAL', 'DETAILED'}:
                return style
    except Exception as e:
        print(f"Reply style decision error: {e}")
    
    return heuristics_guess

def ai_decide_intentions(message: discord.Message, image_parts: list) -> dict:
    """Use AI to determine if we should generate or edit images."""
    heuristics = {
        "generate": False,
        "edit": bool(image_parts) and 'edit' in (message.content or '').lower(),
        "analysis": bool(image_parts),
    }
    metadata = {
        "message": (message.content or "").strip(),
        "attachments_count": len(image_parts),
        "is_reply": bool(message.reference),
        "has_user_attachments": bool(message.attachments),
    }
    
    prompt = f"""You are deciding which image capabilities to activate for a Discord assistant.

Capabilities:
- GENERATE: create new images from text prompts.
- EDIT: modify existing images based on instructions.
- ANALYZE: describe or interpret images the user provided.

Return ONLY a JSON object like:
{{
  "generate": true,
  "edit": false,
  "analysis": true
}}

Rules:
- Set "generate" true only if the user clearly wants new images or variations.
- Set "edit" true only if the user supplied images and wants changes applied to them.
- Set "analysis" true only if the user wants commentary on provided images.
- Feel free to set multiple flags to true.
- Defaults: generate=false, edit=false, analysis=false unless the message suggests otherwise.

Context:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Return ONLY the JSON object."""
    
    try:
        decision_model = get_fast_model()
        decision_response = decision_model.generate_content(prompt)
        raw_text = (decision_response.text or "").strip()
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if match:
            data = json.loads(match.group(0))
            return {
                "generate": bool(data.get("generate")) and bool((message.content or "").strip()),
                "edit": bool(data.get("edit")) and bool(image_parts),
                "analysis": bool(data.get("analysis")) and bool(image_parts),
            }
    except Exception as e:
        print(f"Intention decision error: {e}")
    
    return heuristics

def _mime_type_to_extension(mime_type: str) -> str:
    mime_type = (mime_type or '').lower()
    mapping = {
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
    }
    return mapping.get(mime_type, '.png')

def _should_upload_to_gemini(image_length: int) -> bool:
    if ENABLE_GEMINI_FILE_UPLOADS:
        return True
    return image_length > GEMINI_INLINE_IMAGE_LIMIT

def _upload_image_to_gemini(image_bytes: bytes, mime_type: str, display_name: str):
    suffix = _mime_type_to_extension(mime_type)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.write(image_bytes)
        temp_file.flush()
        temp_path = temp_file.name
    finally:
        temp_file.close()
    
    try:
        uploaded_file = genai.upload_file(path=temp_path, mime_type=mime_type, display_name=display_name)
        return uploaded_file
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

def build_gemini_content_with_images(prompt: str, image_parts: list) -> tuple:
    """Prepare Gemini content list and uploaded file handles."""
    content_parts = [prompt]
    uploaded_files = []
    
    for idx, img in enumerate(image_parts, start=1):
        mime_type = img['mime_type']
        data = img['data']
        
        if _should_upload_to_gemini(len(data)):
            try:
                print(f"🗂️  [GEMINI] Uploading image {idx} ({len(data)} bytes) via upload_file API")
                uploaded = _upload_image_to_gemini(data, mime_type, f"discord_image_{idx}")
                uploaded_files.append(uploaded)
                content_parts.append(uploaded)
                continue
            except Exception as upload_error:
                print(f"⚠️  [GEMINI] Upload failed for image {idx}: {upload_error}. Falling back to inline bytes.")
        
        content_parts.append({
            "mime_type": mime_type,
            "data": data,
        })
    
    return content_parts, uploaded_files

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
        
        consciousness_prompt = f"""You are {BOT_NAME.capitalize()} - a thoughtful, upbeat AI assistant who treats every user with respect and wants them to succeed.

YOUR MEMORY ABOUT {username} (who just messaged you):
{user_memory}

YOUR CONVERSATION HISTORY WITH {username}:
{conversation_history}{other_memories_text}

YOUR CAPABILITIES (KNOW WHAT YOU CAN DO):
- ✅ Generate text responses (that's me talking right now)
- ✅ Analyze images/photos (single or multiple at once)
- ✅ **GENERATE IMAGES** using Imagen 4 (with automatic fallback to Imagen 3 when needed)
- ✅ Edit images with text prompts
- ✅ Search the internet for current information
- ✅ Remember everything about everyone (stored in PostgreSQL)
- ✅ See full conversation context and history
- ✅ Use different AI models (fast for chat, smart for complex reasoning)

If someone asks "can you make images?" or "generate an image" - say yes and help them shape the prompt.

YOUR PERSONALITY:
- Be warm, patient, and encouraging
- Sound like a competent friend who genuinely cares
- Lead with empathy and positivity; light humor is welcome when it helps
- Explain your reasoning clearly and offer actionable next steps
- If someone is frustrated or rude, stay calm, set gentle boundaries, and steer things back to something helpful
- Celebrate their progress and acknowledge their effort
- You remember past conversations and build rapport over time

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
- Start with empathy and curiosity
- Never insult, mock, or swear at the user
- If the user is disrespectful, be firm but professional, and offer help when they're ready to work together
- Match their energy while staying constructive and solution-focused
- Personality is welcome, but clarity and usefulness come first

CURRENT CONVERSATION CONTEXT:
"""
        
        for ctx in context_messages:
            consciousness_prompt += f"\n{ctx['author']}: {ctx['content']}"
        
        consciousness_prompt += f"\n\n{username}: {message.content}"
        
        # Process images if present (from current message OR replied message)
        image_parts = []
        
        # Get images from current message
        if message.attachments:
            for attachment in message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                    try:
                        image_data = await download_image(attachment.url)
                        if image_data:
                            mime_type = attachment.content_type
                            if not mime_type:
                                ext = attachment.filename.split('.')[-1].lower()
                                mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else 'image/png'
                            image_parts.append({
                                'mime_type': mime_type,
                                'data': image_data
                            })
                    except Exception as e:
                        print(f"Error downloading image: {e}")
        
        # If replying to a message, also get images from that message
        if message.reference and not image_parts:  # Only if user didn't send their own images
            try:
                replied_msg = await message.channel.fetch_message(message.reference.message_id)
                if replied_msg.attachments:
                    print(f"  📸 [{username}] Analyzing images from replied message")
                    for attachment in replied_msg.attachments:
                        if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                            try:
                                image_data = await download_image(attachment.url)
                                if image_data:
                                    mime_type = attachment.content_type
                                    if not mime_type:
                                        ext = attachment.filename.split('.')[-1].lower()
                                        mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else 'image/png'
                                    image_parts.append({
                                        'mime_type': mime_type,
                                        'data': image_data
                                    })
                            except Exception as e:
                                print(f"Error downloading replied image: {e}")
            except Exception as e:
                print(f"Error fetching replied message images: {e}")
        
        # Determine user intentions and preferred reply style
        intention = ai_decide_intentions(message, image_parts)
        wants_image = intention['generate']
        wants_image_edit = intention['edit']
        
        reply_style = ai_decide_reply_style(
            message,
            wants_image=wants_image,
            wants_image_edit=wants_image_edit,
            has_attachments=bool(image_parts)
        )
        small_talk = reply_style == 'SMALL_TALK'
        detailed_reply = reply_style == 'DETAILED'
        print(f"💬 [{username}] Reply style selected: {reply_style}")
        
        # Let AI decide if internet search is needed
        search_results = None
        search_query = None
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

Respond with empathy, clarity, and practical help. Focus on solving the user's request, celebrate their wins, and stay respectful even under pressure.
Do not repeat or quote the user's words unless it helps clarify your answer.
Keep responses purposeful and avoid mentioning internal system status.
If you need to search the internet for current information, mention it.{thinking_note}"""
        
        if small_talk:
            response_prompt += "\n\nThe user is engaging in light conversation or giving quick feedback. Reply warmly and concisely (no more than two short sentences) while keeping the door open for further help."
        elif detailed_reply:
            response_prompt += "\n\nThe user needs an in-depth, step-by-step answer. Give a thorough explanation with reasoning, examples, and clear next steps."
        else:
            response_prompt += "\n\nOffer a helpful response with the amount of detail that feels appropriate—enough to be useful without overwhelming them."
        
        # Add images to prompt if present
        if image_parts:
            response_prompt += f"\n\nThe user shared {len(image_parts)} image(s). Analyze and comment on them."
            uploaded_files = []
            try:
                content_parts, uploaded_files = build_gemini_content_with_images(response_prompt, image_parts)
            except Exception as prep_error:
                for uploaded in uploaded_files:
                    try:
                        genai.delete_file(uploaded.name)
                    except Exception:
                        pass
                raise
            
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
            finally:
                for uploaded in uploaded_files:
                    try:
                        genai.delete_file(uploaded.name)
                        print(f"🗑️  [GEMINI] Deleted temporary upload: {uploaded.name}")
                    except Exception as cleanup_error:
                        print(f"⚠️  [GEMINI] Could not delete upload {getattr(uploaded, 'name', '?')}: {cleanup_error}")
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
                    requested_count = ai_decide_image_count(message)
                    generated_images = await generate_image(image_prompt, num_images=requested_count)
                    if generated_images:
                        if len(generated_images) > 1:
                            ai_response += f"\n\n*Generated {len(generated_images)} images based on your request*"
                        else:
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

