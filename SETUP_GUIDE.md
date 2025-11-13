# 🚀 Quick Setup Guide

## What You Need From Discord

### Step 1: Create Discord Bot
1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Give it a name (e.g., "ServerMate")
4. Go to "Bot" tab → "Add Bot"

### Step 2: Get Your Tokens
- **Bot Token**: Bot tab → Click "Reset Token" → Copy it
- **Application ID**: General Information tab → Copy "Application ID"

### Step 3: Enable Intents
In the Bot tab, scroll down to "Privileged Gateway Intents" and enable:
- ✅ Presence Intent
- ✅ Server Members Intent
- ✅ Message Content Intent

### Step 4: Invite Bot to Server
1. Go to OAuth2 → URL Generator
2. Select scopes: `bot` and `applications.commands`
3. Select permissions:
   - Read Messages/View Channels
   - Send Messages
   - Send Messages in Threads
   - Embed Links
   - Attach Files
   - Read Message History
   - Add Reactions
4. Copy the URL at the bottom
5. Open it in browser and invite bot to your server

---

## What You Need From Google

### Get Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

---

## What You Need From Serper (Optional - for web search)

1. Go to https://serper.dev
2. Sign up (free account gives 2,500 searches)
3. Copy your API key from dashboard

---

## Railway Deployment

### Step 1: Create PostgreSQL Database
1. Go to https://railway.app
2. Create new project
3. Click "+ New" → "Database" → "PostgreSQL"
4. Once created, click the database
5. Go to "Variables" tab
6. Copy the `DATABASE_URL`

### Step 2: Deploy Bot
1. Push your code to GitHub
2. In Railway, click "+ New" → "GitHub Repo"
3. Select your repository
4. Go to "Variables" tab and add:

```
DISCORD_TOKEN=<your bot token>
DISCORD_APP_ID=<your application id>
GEMINI_API_KEY=<your gemini key>
DATABASE_URL=<your postgresql url from step 1>
BOT_NAME=servermate
SERPER_API_KEY=<your serper key> (optional)
```

5. Railway will auto-deploy!

---

## Environment Variables Checklist

Copy this and fill it in:

```
✅ DISCORD_TOKEN = 
✅ DISCORD_APP_ID = 
✅ GEMINI_API_KEY = 
✅ DATABASE_URL = 
✅ BOT_NAME = servermate
⬜ SERPER_API_KEY = (optional)
```

---

## Testing the Bot

Once deployed:

1. Go to your Discord server
2. Type: `@ServerMate hello!`
3. The bot should respond!

Try:
- `@ServerMate tell me about yourself`
- `!memory` (to see what it remembers)
- Reply to one of its messages
- Send an image with `@ServerMate what's in this image?`

---

## Troubleshooting

### Bot is offline
- Check Railway logs for errors
- Make sure all environment variables are set
- Verify Discord token is correct

### Bot doesn't respond
- Check bot permissions in Discord server
- Make sure "Message Content Intent" is enabled
- Check Railway logs

### Database errors
- Verify DATABASE_URL is correct
- Make sure PostgreSQL service is running in Railway
- Database will auto-initialize on first run

---

## Cost Estimates

- **Railway**: ~$5-10/month (includes database + hosting)
- **Gemini API**: Your $300 will last months or even a year with normal use!
- **Serper**: 2,500 free searches, then $50 per 100k searches
- **Discord**: Free

---

## Next Steps

After the bot is running:

1. Test all features:
   - Mention detection
   - Reply handling
   - Image analysis
   - Web search
   - Memory system

2. Customize:
   - Change bot name in environment variables
   - Adjust personality in bot.py
   - Modify memory depth in memory.py

3. Monitor:
   - Check Railway logs regularly
   - Use `!stats` command to see usage
   - Monitor Gemini API usage in Google Cloud Console

---

**You're all set! Your AI bot with consciousness is ready to evolve! 🧠✨**

