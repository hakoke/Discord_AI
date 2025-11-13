# Complete Google Gemini API Setup Documentation

## 🎯 Overview

This document covers **complete setup and usage** of Google Gemini models for:
- **Text Generation** (conversations, reasoning, coding)
- **Vision** (image analysis, multi-image understanding)
- **Multi-modal** (text + images combined)

**All information verified against official Google AI documentation.**

---

## 📦 Installation

```bash
pip install google-generativeai==0.8.3
pip install Pillow==10.1.0  # For image handling
```

---

## 🔑 Authentication

Gemini API uses a **simple API key** (no service account needed).

### Get Your API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Select your Google Cloud project (or create a new one)
4. Copy the API key

### Configure in Python

```python
import google.generativeai as genai
import os

# Method 1: Direct
genai.configure(api_key="YOUR_API_KEY_HERE")

# Method 2: Environment variable (recommended)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
```

### Environment Variable Setup

```bash
# Linux/Mac
export GEMINI_API_KEY="your_api_key_here"

# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# .env file
GEMINI_API_KEY=your_api_key_here
```

---

## 📊 Available Gemini Models

### Model Comparison Table

| Model Name | Purpose | Speed | Intelligence | Cost | Context Window |
|------------|---------|-------|--------------|------|----------------|
| `gemini-2.0-flash-exp` | Fast conversations, vision | ⚡⚡⚡ Very Fast | ⭐⭐⭐ Good | 💰 FREE (exp) | 1M tokens |
| `gemini-2.5-pro` | Deep reasoning, coding | ⚡ Slower | ⭐⭐⭐⭐⭐ Best | 💰💰 $1.25/1M | 2M tokens |
| `gemini-1.5-pro` | General purpose | ⚡⚡ Medium | ⭐⭐⭐⭐ Great | 💰💰💰 $3.50/1M | 2M tokens |
| `gemini-1.5-flash` | Balanced | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💰 $0.35/1M | 1M tokens |

### When to Use Each Model

**Use `gemini-2.0-flash-exp` for:**
- Normal conversations
- Quick responses
- Chat applications
- Simple Q&A
- Image analysis (supports vision)
- 90% of use cases

**Use `gemini-2.5-pro` for:**
- Complex coding tasks
- Deep reasoning problems
- Technical analysis
- Mathematical proofs
- System design
- When quality matters most

**Use `gemini-1.5-pro` for:**
- Production applications needing stability
- Long document analysis
- Multi-step reasoning
- When experimental models aren't suitable

**Use `gemini-1.5-flash` for:**
- Cost-conscious applications
- High-volume requests
- When you need speed + moderate intelligence

---

## 🚀 Basic Text Generation

### Simple Generation

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# Create model
model = genai.GenerativeModel('gemini-2.5-pro')

# Generate text
response = model.generate_content('Explain quantum computing in simple terms')

print(response.text)
```

### With Error Handling

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash-exp')

try:
    response = model.generate_content('What is AI?')
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
```

### Streaming Responses

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.5-pro')

# Stream the response word by word
response = model.generate_content(
    'Write a long essay about artificial intelligence',
    stream=True
)

for chunk in response:
    print(chunk.text, end='', flush=True)
```

---

## ⚙️ Model Configuration

### Generation Config

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config={
        "temperature": 1.0,      # Creativity (0.0 = deterministic, 2.0 = very creative)
        "top_p": 0.95,           # Nucleus sampling (0.0-1.0)
        "top_k": 64,             # Top-k sampling (1-100)
        "max_output_tokens": 8192,  # Maximum response length
        "stop_sequences": ["END"],   # Stop generation at these strings
    }
)

response = model.generate_content('Write a creative story')
print(response.text)
```

### Parameter Explanation

**Temperature (0.0 - 2.0):**
- `0.0` - Deterministic, always same output
- `0.7` - Balanced (recommended for most tasks)
- `1.0` - Creative (good for stories, art)
- `1.5+` - Very random (good for brainstorming)

**Top P (0.0 - 1.0):**
- Controls diversity via nucleus sampling
- `0.95` - Good default (recommended)
- Lower = more focused responses
- Higher = more diverse responses

**Top K (1 - 100):**
- Limits vocabulary to top K tokens
- `40` - Good default
- Lower = more predictable
- Higher = more varied vocabulary

**Max Output Tokens:**
- Maximum length of response
- `8192` - Maximum for most models
- `2048` - Good for shorter responses

---

## 🎨 Advanced Text Generation

### System Instructions (Personality)

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    'gemini-2.5-pro',
    system_instruction="""You are a helpful coding assistant.
    
    Your personality:
    - Direct and concise
    - Focus on best practices
    - Provide working code examples
    - Explain your reasoning
    
    You always:
    - Write clean, documented code
    - Point out potential issues
    - Suggest optimizations
    """
)

response = model.generate_content('How do I sort a list in Python?')
print(response.text)
```

### Conversation History (Chat)

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Start a chat session
chat = model.start_chat(history=[])

# Send messages
response1 = chat.send_message('Hello! My name is Alice.')
print(f"Bot: {response1.text}")

response2 = chat.send_message('What is my name?')
print(f"Bot: {response2.text}")  # Will remember "Alice"

response3 = chat.send_message('Tell me about Python programming')
print(f"Bot: {response3.text}")

# View conversation history
print("\nFull conversation:")
for message in chat.history:
    print(f"{message.role}: {message.parts[0].text}")
```

### Pre-seeded Chat History

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.5-pro')

# Start chat with existing history
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": ["I'm working on a Discord bot"]
    },
    {
        "role": "model",
        "parts": ["That's great! I can help you with Discord bot development. What would you like to know?"]
    },
    {
        "role": "user",
        "parts": ["I need help with async/await"]
    }
])

# Continue the conversation
response = chat.send_message('How do I handle async functions in discord.py?')
print(response.text)
```

---

## 👁️ Vision Capabilities

### Single Image Analysis

```python
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="YOUR_API_KEY")

# Only certain models support vision
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Load image
image = Image.open('photo.jpg')

# Analyze image
response = model.generate_content([
    'What is in this image? Describe it in detail.',
    image
])

print(response.text)
```

### Multiple Images

```python
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Load multiple images
image1 = Image.open('photo1.jpg')
image2 = Image.open('photo2.jpg')
image3 = Image.open('photo3.jpg')

# Analyze all together
response = model.generate_content([
    'Compare these three images. What are the similarities and differences?',
    image1,
    image2,
    image3
])

print(response.text)
```

### Image from Bytes

```python
import google.generativeai as genai
from PIL import Image
from io import BytesIO

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Load image from bytes (e.g., from Discord attachment)
async def analyze_discord_image(attachment_bytes):
    image = Image.open(BytesIO(attachment_bytes))
    
    response = model.generate_content([
        'Describe this image',
        image
    ])
    
    return response.text
```

### Image from URL

```python
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Download image from URL
image_url = "https://example.com/image.jpg"
response_img = requests.get(image_url)
image = Image.open(BytesIO(response_img.content))

# Analyze
response = model.generate_content([
    'What is this?',
    image
])

print(response.text)
```

### Vision Models Support

**✅ Models with Vision:**
- `gemini-2.0-flash-exp`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

**❌ No Vision Support:**
- `gemini-2.5-pro` (text only, best for reasoning/coding)

---

## 🔄 Async Support for Discord.py

### Async Wrapper Method

```python
import google.generativeai as genai
import asyncio

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.5-pro')

async def generate_async(prompt: str) -> str:
    """Async wrapper for Gemini"""
    loop = asyncio.get_event_loop()
    
    # Run synchronous generate_content in executor
    response = await loop.run_in_executor(
        None,
        model.generate_content,
        prompt
    )
    
    return response.text

# Usage in Discord bot
async def on_message(message):
    if message.content.startswith('!ask'):
        question = message.content[5:]
        answer = await generate_async(question)
        await message.channel.send(answer)
```

### Full Discord.py Integration

```python
import discord
from discord.ext import commands
import google.generativeai as genai
import asyncio
import os

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Models
fast_model = genai.GenerativeModel('gemini-2.0-flash-exp')
smart_model = genai.GenerativeModel('gemini-2.5-pro')

# Discord bot
bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())

async def ask_gemini(model, prompt: str) -> str:
    """Async Gemini wrapper"""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        model.generate_content,
        prompt
    )
    return response.text

@bot.command()
async def ask(ctx, *, question: str):
    """Ask Gemini a question (fast model)"""
    async with ctx.typing():
        try:
            answer = await ask_gemini(fast_model, question)
            
            # Split long responses
            if len(answer) > 2000:
                chunks = [answer[i:i+2000] for i in range(0, len(answer), 2000)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(answer)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")

@bot.command()
async def think(ctx, *, question: str):
    """Ask Gemini a complex question (smart model)"""
    async with ctx.typing():
        try:
            answer = await ask_gemini(smart_model, question)
            await ctx.send(answer)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")

bot.run(os.getenv('DISCORD_TOKEN'))
```

---

## 🎯 Model Selection Strategy

### Smart Detection (AI Decides)

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

fast_model = genai.GenerativeModel('gemini-2.0-flash-exp')
smart_model = genai.GenerativeModel('gemini-2.5-pro')

def choose_model(user_message: str):
    """Let AI decide which model to use"""
    decision_prompt = f"""User message: "{user_message}"
    
Does this require DEEP REASONING/CODING or just CASUAL CONVERSATION?

DEEP REASONING examples:
- "help me debug this code"
- "explain why quantum entanglement works"
- "solve this complex math problem"
- Technical/mathematical questions

CASUAL CONVERSATION examples:
- "what's up?"
- "tell me a joke"
- "what do you think about [topic]?"
- Simple questions

Respond with ONLY one word: "SMART" or "FAST"
"""
    
    # Use fast model to decide
    decision = fast_model.generate_content(decision_prompt)
    
    if 'SMART' in decision.text.upper():
        return smart_model
    else:
        return fast_model

# Usage
user_input = "help me debug this python code"
selected_model = choose_model(user_input)

response = selected_model.generate_content(user_input)
print(f"Used model: {selected_model.model_name}")
print(response.text)
```

### Keyword-Based Detection

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

fast_model = genai.GenerativeModel('gemini-2.0-flash-exp')
smart_model = genai.GenerativeModel('gemini-2.5-pro')

def choose_model_keywords(message: str):
    """Choose model based on keywords"""
    message_lower = message.lower()
    
    # Smart model keywords
    smart_keywords = [
        'code', 'debug', 'program', 'function', 'algorithm',
        'bug', 'error', 'explain why', 'reasoning', 'logic',
        'proof', 'calculate', 'solve', 'theorem', 'mathematical',
        'architecture', 'system design', 'optimization'
    ]
    
    # Check if any smart keyword is present
    if any(keyword in message_lower for keyword in smart_keywords):
        return smart_model
    else:
        return fast_model

# Usage
message = "How do I optimize this SQL query?"
model = choose_model_keywords(message)
response = model.generate_content(message)
print(response.text)
```

---

## 🛡️ Safety Settings

### Configure Safety Filters

```python
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# This model will be more permissive
response = model.generate_content('Your prompt here')
print(response.text)
```

### Safety Threshold Options

- `BLOCK_NONE` - No filtering
- `BLOCK_ONLY_HIGH` - Block only high-probability harmful content
- `BLOCK_MEDIUM_AND_ABOVE` - Block medium and high (default)
- `BLOCK_LOW_AND_ABOVE` - Most restrictive

### Handle Blocked Responses

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash-exp')

try:
    response = model.generate_content('Your prompt')
    
    # Check if response was blocked
    if response.prompt_feedback.block_reason:
        print(f"Blocked: {response.prompt_feedback.block_reason}")
    else:
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")
```

---

## 🌍 Environment Setup

### For Development

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
```

```python
# Load with python-dotenv
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
```

### For Production (Railway)

Add environment variable in Railway dashboard:
```
GEMINI_API_KEY=your_api_key
```

No code changes needed - `os.getenv()` works automatically.

---

## 💰 Pricing & Quotas

### Current Pricing (as of 2025)

| Model | Input | Output | Free Tier |
|-------|-------|--------|-----------|
| `gemini-2.0-flash-exp` | FREE | FREE | Unlimited (experimental) |
| `gemini-2.5-pro` | $1.25/1M tokens | $5.00/1M tokens | First 50 requests/day free |
| `gemini-1.5-pro` | $3.50/1M tokens | $10.50/1M tokens | First 50 requests/day free |
| `gemini-1.5-flash` | $0.35/1M tokens | $1.05/1M tokens | First 1500 requests/day free |

### Token Estimation

Rough estimates:
- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words
- 1000 tokens ≈ 750 words
- 1M tokens ≈ 750,000 words

Example:
- "Hello, how are you?" = ~5 tokens
- A paragraph (100 words) = ~133 tokens
- A page of text (500 words) = ~666 tokens

### Rate Limits

**Free tier:**
- 15 requests per minute
- 1,500 requests per day (for most models)

**Paid tier:**
- 2,000 requests per minute
- No daily limit

---

## 🔍 Debugging & Error Handling

### Common Errors

```python
import google.generativeai as genai
from google.generativeai.types import BlockedPromptException

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.5-pro')

try:
    response = model.generate_content('Your prompt')
    print(response.text)
    
except BlockedPromptException as e:
    print(f"Prompt was blocked: {e}")
    
except ValueError as e:
    print(f"Invalid input: {e}")
    
except Exception as e:
    print(f"General error: {e}")
```

### Response Validation

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash-exp')

response = model.generate_content('Tell me about AI')

# Check if response has content
if response.candidates:
    candidate = response.candidates[0]
    
    # Check finish reason
    if candidate.finish_reason.name == "STOP":
        print("Complete response:")
        print(response.text)
    elif candidate.finish_reason.name == "MAX_TOKENS":
        print("Response was truncated (hit token limit)")
        print(response.text)
    elif candidate.finish_reason.name == "SAFETY":
        print("Response blocked by safety filters")
else:
    print("No response generated")
```

---

## 📚 Complete Example: Discord AI Bot

```python
import discord
from discord.ext import commands
import google.generativeai as genai
import asyncio
import os
from PIL import Image
from io import BytesIO

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Models
fast_model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    generation_config={
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    },
    system_instruction="""You are a helpful, friendly AI assistant.
    Be conversational and natural. Remember context from the conversation."""
)

smart_model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    }
)

# Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Store chat sessions per user
chat_sessions = {}

async def ask_gemini_async(model, prompt):
    """Async wrapper for Gemini"""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)
    return response.text

@bot.event
async def on_ready():
    print(f'{bot.user} is online!')

@bot.command()
async def chat(ctx, *, message: str):
    """Chat with AI (maintains conversation history)"""
    user_id = str(ctx.author.id)
    
    # Get or create chat session
    if user_id not in chat_sessions:
        chat_sessions[user_id] = fast_model.start_chat(history=[])
    
    async with ctx.typing():
        try:
            chat = chat_sessions[user_id]
            response = await asyncio.get_event_loop().run_in_executor(
                None, chat.send_message, message
            )
            
            await ctx.send(response.text)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")

@bot.command()
async def ask(ctx, *, question: str):
    """Ask AI a question (no history)"""
    async with ctx.typing():
        try:
            answer = await ask_gemini_async(fast_model, question)
            
            if len(answer) > 2000:
                chunks = [answer[i:i+2000] for i in range(0, len(answer), 2000)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(answer)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")

@bot.command()
async def code(ctx, *, question: str):
    """Ask AI about coding (uses smart model)"""
    async with ctx.typing():
        try:
            answer = await ask_gemini_async(smart_model, question)
            await ctx.send(answer)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")

@bot.command()
async def analyze(ctx):
    """Analyze an image (reply to a message with image)"""
    if not ctx.message.reference:
        await ctx.send("Reply to a message with an image!")
        return
    
    # Get referenced message
    ref_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
    
    if not ref_message.attachments:
        await ctx.send("No image found in that message!")
        return
    
    async with ctx.typing():
        try:
            # Download image
            attachment = ref_message.attachments[0]
            image_bytes = await attachment.read()
            image = Image.open(BytesIO(image_bytes))
            
            # Analyze with vision model
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                fast_model.generate_content,
                ['Describe this image in detail', image]
            )
            
            await ctx.send(response.text)
        except Exception as e:
            await ctx.send(f"Error analyzing image: {str(e)}")

@bot.command()
async def reset(ctx):
    """Reset your chat history"""
    user_id = str(ctx.author.id)
    if user_id in chat_sessions:
        del chat_sessions[user_id]
        await ctx.send("Chat history cleared!")
    else:
        await ctx.send("You don't have any chat history.")

bot.run(os.getenv('DISCORD_TOKEN'))
```

---

## ✅ Best Practices

### 1. **Use Appropriate Models**
- Fast model for 90% of requests
- Smart model only when needed
- Don't waste smart model on simple tasks

### 2. **Implement Rate Limiting**
```python
from datetime import datetime, timedelta

user_cooldowns = {}

async def check_cooldown(user_id, cooldown_seconds=3):
    now = datetime.now()
    if user_id in user_cooldowns:
        if now - user_cooldowns[user_id] < timedelta(seconds=cooldown_seconds):
            return False
    user_cooldowns[user_id] = now
    return True
```

### 3. **Handle Long Responses**
```python
def split_message(text, max_length=2000):
    """Split long messages for Discord"""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]
```

### 4. **Cache Responses**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_response(prompt):
    return model.generate_content(prompt).text
```

### 5. **Monitor Costs**
- Log token usage
- Set spending limits in Google Cloud Console
- Use fast model by default

---

## 🔗 Official Resources

- **API Documentation**: https://ai.google.dev/docs
- **API Key Management**: https://aistudio.google.com/app/apikey
- **Pricing**: https://ai.google.dev/pricing
- **Model Cards**: https://ai.google.dev/gemini-api/docs/models
- **Safety Settings**: https://ai.google.dev/gemini-api/docs/safety-settings

---

## 🎯 Quick Reference

### Model Names
```python
'gemini-2.0-flash-exp'  # Fast, vision, FREE
'gemini-2.5-pro'        # Smartest, no vision, paid
'gemini-1.5-pro'        # Stable, vision, paid
'gemini-1.5-flash'      # Balanced, vision, cheap
```

### Basic Setup
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content('Hello!')
```

### With Images
```python
from PIL import Image
image = Image.open('photo.jpg')
response = model.generate_content(['Describe this', image])
```

### Chat
```python
chat = model.start_chat(history=[])
response = chat.send_message('Hello!')
```

---

**Last Updated**: 2025  
**Source**: Official Google AI Studio Documentation  
**Verified**: All model names and features confirmed

