# ServerMate - AI Discord Bot with Consciousness

An advanced AI-powered Discord bot using Google Gemini 2.0 Flash with deep memory and evolving consciousness. Think Grok from Twitter, but personalized for Discord with memory that spans conversations.

## 🌟 Features

- **PERSONALITY**: 
  - Real, unfiltered AI that can swear and has actual opinions
  - Fights back when provoked - not a corporate pushover
  - Matches your energy - casual, smart, or savage
  - Brutally honest internal memory about users
- **Triple Model Intelligence**:
  - **Fast Model** (gemini-2.0-flash-exp): Lightning-fast responses for normal chat
  - **Smart Model** (gemini-2.5-pro): SMARTEST MODEL for deep reasoning/coding/complex tasks (text-only)
  - **Vision Model** (gemini-2.0-flash-exp): For everyday image analysis
  - **AI decides which model to use** - not hardcoded keywords!
  - **Auto-fallback** to stable versions if experimental models unavailable
- **Natural Conversations**: Responds to @mentions, replies, and name mentions (even with typos!)
- **Deep Memory System**: PostgreSQL-backed consciousness that remembers everything
  - **Fully AI-driven memory** - AI structures memories however it wants (not hardcoded)
  - User personalities and preferences (brutally honest assessments)
  - Conversation history with full context
  - Relationship dynamics and how it really feels about you
  - Learned behaviors and patterns
  - Internal "consciousness stream" of uncensored thoughts
- **Multi-Modal Intelligence**:
  - Analyzes multiple images at once
  - **Image Generation** with Imagen 3.0 - create images from text prompts
  - **Image Editing** - modify images with natural language
  - Internet search capabilities
  - Context-aware conversation threading
- **Human-like Evolution**: Gets more personalized and "human" with each interaction

## 🚀 Setup Instructions

### 1. Discord Bot Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a New Application
3. Go to "Bot" section and create a bot
4. **Copy your Bot Token** (you'll need this)
5. **Copy your Application ID** from the "General Information" section
6. Enable these Privileged Gateway Intents:
   - Message Content Intent
   - Server Members Intent
   - Presence Intent
7. Go to OAuth2 → URL Generator
8. Select scopes: `bot`, `applications.commands`
9. Select bot permissions:
   - Read Messages/View Channels
   - Send Messages
   - Send Messages in Threads
   - Embed Links
   - Attach Files
   - Read Message History
   - Add Reactions
10. Copy the generated URL and invite the bot to your server

### 2. Gemini API Setup

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. **Copy your API key** (for text generation)

### 3. Imagen 3 / Vertex AI Setup (for image generation)

**You already have the service account JSON!** Just need to:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Make sure your Google Cloud project has:
   - **Vertex AI API** enabled
   - **Billing** enabled (your $300 credit applies here)
3. That's it! Your JSON file already has the credentials

Your $300 credit covers BOTH Gemini and Imagen!

### 4. Serper API Setup (Optional - for internet search)

1. Go to [Serper.dev](https://serper.dev)
2. Sign up for free account (2,500 free searches)
3. **Copy your API key**

### 5. Railway Setup

#### A. Create PostgreSQL Database

1. Go to [Railway](https://railway.app)
2. Create a new project
3. Click "+ New" → "Database" → "PostgreSQL"
4. Once created, click on the database
5. Go to "Variables" tab
6. **Copy the `DATABASE_URL`** value

#### B. Deploy the Bot

1. Click "+ New" → "GitHub Repo"
2. Connect your GitHub account and select this repository
3. Railway will auto-detect the Python app
4. **Upload your service account JSON**:
   - Set the path to your service account JSON file as a Railway secret (recommended)
   - **OR** place it in your repo root (less secure, not recommended for public repos)
5. Go to "Variables" tab and add these:

```
DISCORD_TOKEN=your_discord_bot_token
DISCORD_APP_ID=your_discord_application_id
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=your_postgresql_connection_url
BOT_NAME=servermate
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account.json
SERPER_API_KEY=your_serper_api_key (optional)
```

6. Go to "Settings" tab
7. Under "Deploy", make sure it's set to deploy from `main` branch
8. Click "Deploy"

The bot will start automatically!

## 💬 Using the Bot

### Interaction Methods

The bot responds when you:
- **@mention it**: `@ServerMate what do you think?`
- **Reply to its messages**: Just reply to any message it sent
- **Say its name**: `hey servermate, what's up?` (works with typos too!)

### Commands

- `!memory` - See what the bot remembers about you (the public version, not the brutally honest internal one 😏)
- `!forget` - Clear your memory (fresh start)
- `!stats` - View bot statistics and active models
- `!models` - See available AI models and when they're used
- `!imagine <prompt>` - Generate an image from a text prompt

### Examples

**Normal Chat (Fast Model):**
```
You: @ServerMate what's up?
Bot: Not much, just vibing in the server. What about you?
```

**Coding Help (Auto-switches to Smart Model):**
```
You: @servermate debug this python code for me
Bot: [Uses Smart Model] Alright so here's the issue - you're calling that function before it's defined...
```

**Image Analysis:**
```
You: @ServerMate analyze these images
Bot: [Analyzes all images] Yo that's actually sick...
```

**Internet Search:**
```
You: servermate search for the latest AI news
Bot: [Searches and responds with findings]
```

**Conversation Context:**
```
You: *replies to bot's message* is that really true?
Bot: [Reads full conversation thread and responds with context]
```

**When Provoked:**
```
You: you're stupid
Bot: Lmao says the person who couldn't figure out [brings up past mistake]. Try again.
```

**Image Generation:**
```
You: @servermate generate an image of a cyberpunk city at night
Bot: *Generated image based on your request* [image appears]

You: @servermate make me a picture of a dragon
Bot: [generates and sends image]

You: [uploads image] @servermate edit this to make it darker and add rain
Bot: [generates edited version]

You: !imagine a futuristic robot playing guitar
Bot: [generates image]
```

## 🧠 Memory System

The bot maintains a sophisticated consciousness through:

1. **User Memory**: Personality, preferences, relationship notes
2. **Interaction History**: Every conversation stored with context
3. **Consciousness Stream**: Internal thoughts and realizations
4. **Learned Behaviors**: Patterns discovered from interactions

All stored in PostgreSQL for persistence across restarts.

## 🔥 Uncensored Personality

This bot is **NOT** your typical corporate AI:

- **Fully uncensored** - Can swear, curse, have actual opinions
- **Fights back** - Won't take shit if provoked
- **Real personality** - Not fake nice, genuinely itself
- **Brutally honest memory** - Internal assessments of users are HONEST (smart/dumb, funny/unfunny, cool/cringe)
- **Matches energy** - Casual with casual, savage with savage, smart with smart
- **No corporate leash** - Says what it thinks, not what it's "supposed to" say

## 🧠 Dual Model Intelligence

The bot **AI decides which model to use** for each message:

- **🏃 Fast Model** (gemini-2.0-flash-exp): For normal chat - lightning fast, super cheap
- **🧠 Smart Model** (gemini-2.5-pro): For coding, debugging, complex reasoning - SMARTEST AVAILABLE

The AI analyzes each message and decides if it needs deep reasoning or just casual conversation. Not based on hardcoded keywords - the AI actually thinks about it.

Use `!models` command to see which models are active.

## 📁 Project Structure

```
Discord_AI/
├── bot.py                      # Main bot logic with dual model switching
├── database.py                 # Database operations
├── memory.py                   # Memory/consciousness system (uses smart model)
├── requirements.txt            # Python dependencies
├── Procfile                    # Railway process configuration
├── railway.json                # Railway deployment config
├── runtime.txt                 # Python version specification
├── init_db.py                  # Database initialization script
├── .env.example / env.example  # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # Main documentation
└── SETUP_GUIDE.md              # Step-by-step setup
```

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | Yes | Your Discord bot token |
| `DISCORD_APP_ID` | Yes | Your Discord application ID |
| `GEMINI_API_KEY` | Yes | Google Gemini API key (for text) |
| `DATABASE_URL` | Yes | PostgreSQL connection URL |
| `GOOGLE_CLOUD_PROJECT` | For images | Your Google Cloud project ID |
| `GOOGLE_CLOUD_LOCATION` | For images | GCP region (default: us-central1) |
| `GOOGLE_APPLICATION_CREDENTIALS` | For images | Path to service account JSON |
| `BOT_NAME` | No | Bot name (default: servermate) |
| `SERPER_API_KEY` | No | Serper API for web search |

## 📊 Database Schema

### Tables

- **interactions**: Every conversation (user message, bot response, context, images, searches)
- **user_memory**: Per-user consciousness (personality, preferences, relationship notes)
- **learned_behaviors**: Bot's evolving behavioral patterns
- **consciousness_stream**: Internal thoughts and realizations

## 🎯 Gemini Models Used

**Fast Model: Gemini 2.0 Flash Experimental**
- ⚡ Lightning speed (< 1 second)
- 💰 Super cheap / currently FREE (your $300 lasts forever)
- 👁️ Vision capabilities
- Perfect for 90% of conversations

**Smart Model: Gemini 2.5 Pro**
- 🧠 SMARTEST model available
- 🔬 Best for coding, debugging, complex reasoning
- 👁️ **Supports vision** for complex image analysis (diagrams, code screenshots, documents)
- 💰 More expensive but still very reasonable (~$1.25 per 1M tokens)
- AI automatically decides when to use it

**Vision Model: Gemini 2.0 Flash**
- 👁️ For everyday image analysis
- ⚡ Fast and efficient
- 💰 FREE (experimental) or cheap when stable
- Handles multiple images at once

**Image Generation: Imagen 3.0**
- 🎨 State-of-the-art image generation
- 💰 ~$0.02-0.04 per image
- Supports text-to-image and image editing
- **Requires Vertex AI** (Google Cloud project + service account)
- Your service account JSON works for this!

## 🔐 Security Notes

- Never commit your `.env` file or credentials
- The `.gitignore` excludes sensitive files
- Google Cloud credentials JSON is ignored by default
- Keep your Discord token secure

## 🚨 Troubleshooting

### Bot doesn't respond
- Check Railway logs for errors
- Verify all environment variables are set
- Ensure bot has proper Discord permissions
- Check database connection

### Database errors
- Verify `DATABASE_URL` is correct
- Check Railway PostgreSQL service is running
- Database tables are created automatically on first run

### API errors
- Verify Gemini API key is valid
- Check you have credits remaining
- Ensure Serper API key is valid (if using search)

## 📈 Future Enhancements

This bot is designed to grow! Planned features:
- Voice channel integration
- Image generation
- Custom personality modes
- Multi-guild memory isolation
- Advanced emotion modeling
- Proactive conversation initiation

## 📝 License

MIT License - feel free to modify and use!

## 🙋 Support

If you encounter issues:
1. Check Railway logs
2. Verify environment variables
3. Ensure database is connected
4. Check Discord bot permissions

---

**Built with ❤️ using Google Gemini, Discord.py, and PostgreSQL**
