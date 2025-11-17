"""
ServerMate Official Website
Public-facing website with admin dashboard
"""
from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for, send_from_directory
import asyncpg
import os
import asyncio
from datetime import datetime, timedelta, date
import json
import aiohttp
import time
from functools import wraps

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'servermate-official-website-secret-key-change-in-production')

# Admin credentials
ADMIN_USERNAME = 'Hakoke'
ADMIN_PASSWORD = 'Ironman'

# Discord API cache
discord_guild_cache = {}
CACHE_DURATION = 3600  # 1 hour

# Discord invite link
DISCORD_INVITE = 'https://discord.gg/DxMYMY7Z6v'

async def get_discord_guild_info(guild_id: str):
    """Fetch Discord server (guild) information from Discord API"""
    discord_token = os.getenv('DISCORD_TOKEN')
    if not discord_token:
        return None
    
    # Check cache first
    if guild_id in discord_guild_cache:
        cached = discord_guild_cache[guild_id]
        if time.time() - cached['cached_at'] < CACHE_DURATION:
            return cached['data']
    
    try:
        url = f"https://discord.com/api/v10/guilds/{guild_id}?with_counts=true"
        headers = {
            "Authorization": f"Bot {discord_token}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    guild_info = {
                        'name': data.get('name', 'Unknown Server'),
                        'icon': data.get('icon'),
                        'id': str(data.get('id', guild_id)),
                        'member_count': data.get('approximate_member_count'),
                        'description': data.get('description')
                    }
                    
                    # Cache the result
                    discord_guild_cache[guild_id] = {
                        'data': guild_info,
                        'cached_at': time.time()
                    }
                    
                    return guild_info
                elif response.status == 404:
                    print(f"Discord API 404 for guild {guild_id}: Bot may not be in server")
                    return {'name': None, 'id': guild_id, 'not_found': True}
                else:
                    error_text = await response.text()
                    print(f"Discord API error for guild {guild_id}: Status {response.status}, Response: {error_text}")
                    return {'name': None, 'id': guild_id, 'error': f"HTTP {response.status}"}
    except Exception as e:
        import traceback
        error_msg = f"Error fetching Discord guild info for {guild_id}: {e}"
        print(error_msg)
        print(traceback.format_exc())
        # Return a dict with None name so template knows it failed
        return {'name': None, 'id': guild_id, 'error': str(e)}

def get_discord_guild_info_sync(guild_id: str):
    """Synchronous wrapper for get_discord_guild_info"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(get_discord_guild_info(guild_id))
    finally:
        pass

def run_async(coro):
    """Run async function in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(coro)
    finally:
        pass

def admin_required(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.template_filter('tojson')
def tojson_filter(obj):
    """Convert object to JSON string, handling date/datetime objects"""
    def json_serial(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    return json.dumps(obj, default=json_serial)

# Route for profile picture
@app.route('/profile.png')
def profile_picture():
    """Serve the profile picture from images folder"""
    try:
        import os
        from flask import Response
        
        images_dir = 'images'
        # Try different file formats in order of preference
        profile_files = [
            ('generated_1.webp', 'image/webp'),
            ('profile.png', 'image/png'),
            ('profile.jpg', 'image/jpeg'),
            ('profile.jpeg', 'image/jpeg'),
            ('profile.webp', 'image/webp')
        ]
        
        # First try images folder
        for filename, mimetype in profile_files:
            filepath = os.path.join(images_dir, filename)
            if os.path.exists(filepath):
                response = send_from_directory(images_dir, filename)
                # Set correct mimetype for .webp files
                if filename.endswith('.webp'):
                    response.mimetype = mimetype
                return response
        
        # If not found in images folder, try root
        for filename, mimetype in profile_files:
            if os.path.exists(filename):
                response = send_from_directory('.', filename)
                if filename.endswith('.webp'):
                    response.mimetype = mimetype
                return response
        
        return '', 404
    except Exception as e:
        print(f"Error serving profile picture: {e}")
        import traceback
        traceback.print_exc()
        return '', 404

# Public Homepage
PUBLIC_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ServerMate - Complete AI Assistant for Discord</title>
    <link rel="icon" type="image/png" href="/profile.png">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
            --primary: #5865F2;
            --primary-hover: #4752C4;
            --secondary: #57F287;
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-card-hover: #1c2128;
            --border: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --text-accent: #58a6ff;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 0 20px; }
        
        /* Header */
        header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(10px);
        }
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 24px;
            font-weight: bold;
            color: var(--text-primary);
            text-decoration: none;
        }
        .logo img {
            width: 50px;
            height: 50px;
            border-radius: 50%;
        }
        .nav-links {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            transition: color 0.2s;
        }
        .nav-links a:hover {
            color: var(--text-primary);
        }
        .btn {
            padding: 10px 20px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s;
            display: inline-block;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        .btn-primary:hover {
            background: var(--primary-hover);
        }
        .btn-secondary {
            background: var(--border);
            color: var(--text-primary);
        }
        .btn-secondary:hover {
            background: #21262d;
        }
        
        /* Hero Section */
        .hero {
            text-align: center;
            padding: 80px 20px;
            background: linear-gradient(135deg, rgba(88, 101, 242, 0.1) 0%, rgba(87, 242, 135, 0.1) 100%);
        }
        .hero h1 {
            font-size: 48px;
            margin-bottom: 20px;
            background: linear-gradient(135deg, var(--text-accent), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero p {
            font-size: 20px;
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto 30px;
        }
        .hero-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        /* Features Section */
        .features {
            padding: 80px 20px;
        }
        .section-title {
            text-align: center;
            font-size: 36px;
            margin-bottom: 50px;
            color: var(--text-primary);
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 60px;
        }
        .feature-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 30px;
            transition: all 0.3s;
        }
        .feature-card:hover {
            border-color: var(--primary);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(88, 101, 242, 0.2);
        }
        .feature-icon {
            font-size: 48px;
            margin-bottom: 20px;
        }
        .feature-card h3 {
            font-size: 24px;
            margin-bottom: 15px;
            color: var(--text-accent);
        }
        .feature-card p {
            color: var(--text-secondary);
            line-height: 1.8;
        }
        
        /* Servers Section */
        .servers {
            padding: 80px 20px;
            background: var(--bg-card);
        }
        .servers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        .server-card {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s;
        }
        .server-card:hover {
            border-color: var(--primary);
            transform: translateY(-3px);
        }
        .server-icon {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--border);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        .server-name {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text-primary);
        }
        .server-stats {
            display: flex;
            gap: 15px;
            font-size: 14px;
            color: var(--text-secondary);
        }
        .server-stats span {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .stat-number {
            color: var(--text-accent);
            font-weight: 600;
        }
        
        /* Footer */
        footer {
            background: var(--bg-card);
            border-top: 1px solid var(--border);
            padding: 40px 20px;
            text-align: center;
            color: var(--text-secondary);
        }
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .footer-links a {
            color: var(--text-secondary);
            text-decoration: none;
            transition: color 0.2s;
        }
        .footer-links a:hover {
            color: var(--text-primary);
        }
        
        /* Stats Section */
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 60px 20px;
        }
        .stat-card {
            text-align: center;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 30px;
        }
        .stat-number {
            font-size: 48px;
            font-weight: bold;
            color: var(--text-accent);
            margin-bottom: 10px;
        }
        .stat-label {
            color: var(--text-secondary);
            font-size: 16px;
        }
        
        @media (max-width: 768px) {
            .hero h1 { font-size: 36px; }
            .hero p { font-size: 18px; }
            .section-title { font-size: 28px; }
            nav { flex-direction: column; gap: 15px; }
        }
    </style>
</head>
<body>
    <header>
        <nav class="container">
            <a href="/" class="logo">
                <img src="/profile.png" alt="ServerMate" onerror="this.style.display='none'">
                <span>ServerMate</span>
            </a>
            <div class="nav-links">
                <a href="#features">Features</a>
                <a href="#servers">Servers</a>
                <a href="{{ discord_invite }}" target="_blank">Discord</a>
                <a href="/admin/login" class="btn btn-secondary">Admin</a>
            </div>
        </nav>
    </header>
    
    <section class="hero">
        <div class="container">
            <h1>ServerMate</h1>
            <p>Complete AI assistant for Discord. Images, search, documents, memory and more!</p>
            <div class="hero-buttons">
                <a href="{{ discord_invite }}" target="_blank" class="btn btn-primary">Invite to Discord</a>
                <a href="https://top.gg/bot/1438667256866537482" target="_blank" class="btn btn-secondary">View on Top.gg</a>
            </div>
        </div>
    </section>
    
    <section class="stats">
        <div class="stat-card">
            <div class="stat-number">{{ stats.unique_servers or 0 }}</div>
            <div class="stat-label">Servers</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.total_interactions }}</div>
            <div class="stat-label">Total Interactions</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.unique_users }}</div>
            <div class="stat-label">Unique Users</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.memory_records }}</div>
            <div class="stat-label">Memory Records</div>
        </div>
    </section>
    
    <section id="features" class="features">
        <div class="container">
            <h2 class="section-title">What Can ServerMate Do?</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🌐</div>
                    <h3>Real-Time Web Browsing</h3>
                    <p>Visit websites in real time, click buttons, scroll pages, and show you screenshots so you can see exactly what's happening. Fact-check links, images, and rumors.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🖼️</div>
                    <h3>Image Analysis & Generation</h3>
                    <p>Send it a picture and it'll analyze it. Give it an idea and it'll generate images. Powered by advanced AI models for stunning results.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💻</div>
                    <h3>Code Assistant</h3>
                    <p>Ask it to code and it'll write, debug, or explain anything you throw at it. Supports multiple programming languages and frameworks.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📄</div>
                    <h3>Document Generation</h3>
                    <p>Need a PDF or Word file? It can generate those too - poems, notes, reports, anything you need. Full document creation capabilities.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h3>Advanced Search</h3>
                    <p>Searches the internet on its own - not just Google, but Reddit, YouTube, Instagram and more. It decides the best way to get the information you need.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <h3>Memory & Context</h3>
                    <p>Understands context, remembers conversations, and builds a personality over time based on how you interact with it. Truly intelligent.</p>
                </div>
            </div>
            
            <div style="max-width: 800px; margin: 0 auto; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 30px;">
                <h3 style="color: var(--text-accent); margin-bottom: 20px; font-size: 24px;">How to Use ServerMate</h3>
                <p style="color: var(--text-secondary); line-height: 1.8; margin-bottom: 15px;">
                    You can talk to it casually, mention it by name, reply to one of its messages, or just say "hey servermate" (even with typos). It'll know you're talking to it.
                </p>
                <div style="margin-top: 25px;">
                    <h4 style="color: var(--text-primary); margin-bottom: 15px;">Slash Commands:</h4>
                    <ul style="color: var(--text-secondary); line-height: 2; list-style: none; padding: 0;">
                        <li><code style="background: var(--bg-dark); padding: 4px 8px; border-radius: 4px;">/stop</code> - Stops the current AI request (only your request)</li>
                        <li><code style="background: var(--bg-dark); padding: 4px 8px; border-radius: 4px;">/profile</code> - Shows your profile (what the AI saved about you)</li>
                        <li><code style="background: var(--bg-dark); padding: 4px 8px; border-radius: 4px;">/help</code> - Full capability list of the AI and information</li>
                    </ul>
                </div>
            </div>
        </div>
    </section>
    
    <section id="servers" class="servers">
        <div class="container">
            <h2 class="section-title">Servers Using ServerMate</h2>
            {% if servers %}
            <div class="servers-grid">
                {% for server in servers %}
                <div class="server-card">
                    {% if server.guild_info and server.guild_info.icon %}
                    <div class="server-icon" style="background-image: url('https://cdn.discordapp.com/icons/{{ server.guild_id }}/{{ server.guild_info.icon }}.png'); background-size: cover;"></div>
                    {% else %}
                    <div class="server-icon">🖥️</div>
                    {% endif %}
                    <div class="server-name">
                        {% if server.guild_info and server.guild_info.name %}
                            {{ server.guild_info.name }}
                        {% else %}
                            Server {{ loop.index }}
                        {% endif %}
                    </div>
                    <div class="server-stats">
                        <span>👥 <span class="stat-number">{{ server.guild_info.member_count if server.guild_info and server.guild_info.member_count else '?' }}</span></span>
                        <span>💬 <span class="stat-number">{{ server.interaction_count }}</span></span>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="text-align: center; color: var(--text-secondary); padding: 40px;">
                No servers found yet. Be the first to invite ServerMate!
            </p>
            {% endif %}
        </div>
    </section>
    
    <footer>
        <div class="container">
            <div class="footer-links">
                <a href="{{ discord_invite }}" target="_blank">Discord Server</a>
                <a href="https://top.gg/bot/1438667256866537482" target="_blank">Top.gg Page</a>
                <a href="/admin/login">Admin Login</a>
            </div>
            <p>&copy; 2024 ServerMate. Complete AI assistant for Discord.</p>
        </div>
    </footer>
</body>
</html>
"""

# Admin Login Page
ADMIN_LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Login - ServerMate</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .login-container {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
        }
        .login-container h1 {
            color: #58a6ff;
            margin-bottom: 30px;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #8b949e;
            font-size: 14px;
        }
        .form-group input {
            width: 100%;
            padding: 12px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
        }
        .form-group input:focus {
            outline: none;
            border-color: #58a6ff;
        }
        .btn {
            width: 100%;
            padding: 12px;
            background: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn:hover {
            background: #2ea043;
        }
        .error {
            color: #f85149;
            margin-top: 10px;
            font-size: 14px;
        }
        .back-link {
            text-align: center;
            margin-top: 20px;
        }
        .back-link a {
            color: #58a6ff;
            text-decoration: none;
        }
        .back-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>Admin Login</h1>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/admin/login">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required autofocus>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="btn">Login</button>
        </form>
        <div class="back-link">
            <a href="/">← Back to Home</a>
        </div>
    </div>
</body>
</html>
"""

# Admin Dashboard HTML (will be created in the next part)
# Due to length, I'll create a separate function for admin dashboard

def get_admin_dashboard_html():
    """Generate admin dashboard HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - ServerMate</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #30363d;
        }
        header h1 { color: #58a6ff; }
        .header-actions {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .btn {
            padding: 10px 20px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }
        .btn-secondary {
            background: #30363d;
            color: #c9d1d9;
        }
        .btn-secondary:hover { background: #21262d; }
        .btn-danger {
            background: #da3633;
            color: white;
        }
        .btn-danger:hover { background: #f85149; }
        .search-bar {
            width: 100%;
            padding: 12px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .search-bar:focus {
            outline: none;
            border-color: #58a6ff;
        }
        .filters {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .filter-group label {
            font-size: 12px;
            color: #8b949e;
        }
        .filter-group select, .filter-group input {
            padding: 8px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 1px solid #30363d;
        }
        .tab {
            padding: 10px 20px;
            background: none;
            border: none;
            color: #8b949e;
            cursor: pointer;
            font-size: 14px;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab.active {
            color: #58a6ff;
            border-bottom-color: #58a6ff;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            overflow: hidden;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }
        th {
            background: #0d1117;
            color: #58a6ff;
            font-weight: 600;
            font-size: 13px;
        }
        tr:hover { background: #1c2128; }
        .server-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .server-card:hover {
            border-color: #58a6ff;
            background: #1c2128;
        }
        .server-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .server-name {
            font-size: 18px;
            font-weight: 600;
            color: #79c0ff;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
        }
        .stat-number {
            font-size: 32px;
            font-weight: bold;
            color: #58a6ff;
        }
        .stat-label {
            color: #8b949e;
            font-size: 12px;
            margin-top: 5px;
        }
        .message-preview {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 5px;
        }
        .badge-image { background: #238636; color: white; }
        .badge-doc { background: #1f6feb; color: white; }
        .badge-search { background: #8957e5; color: white; }
        .pagination {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }
        .pagination button {
            padding: 8px 15px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            cursor: pointer;
        }
        .pagination button:hover {
            background: #1c2128;
        }
        .pagination button.active {
            background: #238636;
            border-color: #238636;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔒 Admin Dashboard</h1>
            <div class="header-actions">
                <span style="color: #8b949e;">Logged in as: <strong>{{ username }}</strong></span>
                <a href="/admin/logout" class="btn btn-danger">Logout</a>
                <a href="/" class="btn btn-secondary">← Home</a>
            </div>
        </header>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('servers')">Servers</button>
            <button class="tab" onclick="switchTab('users')">Users</button>
            <button class="tab" onclick="switchTab('messages')">Messages</button>
            <button class="tab" onclick="switchTab('memory')">Memory Logs</button>
        </div>
        
        <!-- Servers Tab -->
        <div id="servers-tab" class="tab-content active">
            <input type="text" class="search-bar" id="server-search" placeholder="Search servers..." oninput="filterServers()">
            <div id="servers-list"></div>
        </div>
        
        <!-- Users Tab -->
        <div id="users-tab" class="tab-content">
            <div class="filters">
                <div class="filter-group">
                    <label>Server</label>
                    <select id="user-server-filter" onchange="loadUsers()">
                        <option value="">All Servers</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search</label>
                    <input type="text" id="user-search" placeholder="Search users..." oninput="filterUsers()">
                </div>
            </div>
            <div id="users-list"></div>
        </div>
        
        <!-- Messages Tab -->
        <div id="messages-tab" class="tab-content">
            <div class="filters">
                <div class="filter-group">
                    <label>Server</label>
                    <select id="msg-server-filter" onchange="loadMessages()">
                        <option value="">All Servers</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>User</label>
                    <input type="text" id="msg-user-filter" placeholder="User ID or username..." onchange="loadMessages()">
                </div>
                <div class="filter-group">
                    <label>Search</label>
                    <input type="text" id="msg-search" placeholder="Search messages..." oninput="filterMessages()">
                </div>
                <div class="filter-group">
                    <label>Type</label>
                    <select id="msg-type-filter" onchange="filterMessages()">
                        <option value="">All</option>
                        <option value="image">With Images</option>
                        <option value="doc">With Documents</option>
                        <option value="search">With Search</option>
                    </select>
                </div>
            </div>
            <table id="messages-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Server</th>
                        <th>User</th>
                        <th>User Message</th>
                        <th>Bot Response</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody id="messages-body"></tbody>
            </table>
            <div class="pagination" id="messages-pagination"></div>
        </div>
        
        <!-- Memory Tab -->
        <div id="memory-tab" class="tab-content">
            <div class="filters">
                <div class="filter-group">
                    <label>Server</label>
                    <select id="mem-server-filter" onchange="loadMemory()">
                        <option value="">All Servers</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search</label>
                    <input type="text" id="mem-search" placeholder="Search memory..." oninput="filterMemory()">
                </div>
            </div>
            <div id="memory-list"></div>
        </div>
    </div>
    
    <script>
        let currentTab = 'servers';
        let allServers = [];
        let allUsers = [];
        let allMessages = [];
        let allMemory = [];
        let currentPage = 1;
        const itemsPerPage = 50;
        
        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
            
            if (tab === 'servers') loadServers();
            else if (tab === 'users') loadUsers();
            else if (tab === 'messages') loadMessages();
            else if (tab === 'memory') loadMemory();
        }
        
        async function loadServers() {
            try {
                const response = await fetch('/admin/api/servers');
                allServers = await response.json();
                displayServers(allServers);
                populateServerFilters();
            } catch (error) {
                console.error('Error loading servers:', error);
            }
        }
        
        function displayServers(servers) {
            const container = document.getElementById('servers-list');
            if (servers.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #8b949e; padding: 40px;">No servers found.</p>';
                return;
            }
            
            container.innerHTML = servers.map(server => `
                <div class="server-card" onclick="viewServerDetail('${server.guild_id}')">
                    <div class="server-header">
                        <div class="server-name">${server.guild_info?.name || 'Server ' + server.guild_id}</div>
                        <div>ID: ${server.guild_id}</div>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">${server.interaction_count || 0}</div>
                            <div class="stat-label">Interactions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${server.unique_users || 0}</div>
                            <div class="stat-label">Users</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${server.guild_info?.member_count || '?'}</div>
                            <div class="stat-label">Members</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${server.last_activity ? new Date(server.last_activity).toLocaleDateString() : 'N/A'}</div>
                            <div class="stat-label">Last Active</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        function filterServers() {
            const search = document.getElementById('server-search').value.toLowerCase();
            const filtered = allServers.filter(s => 
                (s.guild_info?.name || '').toLowerCase().includes(search) ||
                s.guild_id.toLowerCase().includes(search)
            );
            displayServers(filtered);
        }
        
        async function loadUsers() {
            try {
                const serverId = document.getElementById('user-server-filter').value;
                const response = await fetch('/admin/api/users' + (serverId ? '?guild_id=' + serverId : ''));
                allUsers = await response.json();
                displayUsers(allUsers);
            } catch (error) {
                console.error('Error loading users:', error);
            }
        }
        
        function displayUsers(users) {
            const container = document.getElementById('users-list');
            if (users.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #8b949e; padding: 40px;">No users found.</p>';
                return;
            }
            
            container.innerHTML = '<table><thead><tr><th>User</th><th>Interactions</th><th>Last Active</th><th>Memory</th></tr></thead><tbody>' +
                users.map(user => `
                    <tr>
                        <td><strong>${user.username}</strong><br><span style="color: #8b949e; font-size: 12px;">${user.user_id}</span></td>
                        <td>${user.interaction_count || 0}</td>
                        <td>${user.last_interaction ? new Date(user.last_interaction).toLocaleString() : 'N/A'}</td>
                        <td>${user.memory?.memory_summary ? '<details><summary>View</summary><pre>' + JSON.stringify(user.memory, null, 2) + '</pre></details>' : 'None'}</td>
                    </tr>
                `).join('') + '</tbody></table>';
        }
        
        function filterUsers() {
            const search = document.getElementById('user-search').value.toLowerCase();
            const filtered = allUsers.filter(u => 
                u.username.toLowerCase().includes(search) ||
                u.user_id.toLowerCase().includes(search)
            );
            displayUsers(filtered);
        }
        
        async function loadMessages(page = 1) {
            try {
                currentPage = page;
                const serverId = document.getElementById('msg-server-filter').value;
                const userId = document.getElementById('msg-user-filter').value;
                let url = '/admin/api/messages?page=' + page + '&limit=' + itemsPerPage;
                if (serverId) url += '&guild_id=' + serverId;
                if (userId) url += '&user_id=' + userId;
                
                const response = await fetch(url);
                const data = await response.json();
                allMessages = data.messages || [];
                displayMessages(allMessages);
                
                // Pagination
                const totalPages = Math.ceil((data.total || 0) / itemsPerPage);
                displayPagination(totalPages, page);
            } catch (error) {
                console.error('Error loading messages:', error);
            }
        }
        
        function displayMessages(messages) {
            const tbody = document.getElementById('messages-body');
            if (messages.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #8b949e; padding: 40px;">No messages found.</td></tr>';
                return;
            }
            
            tbody.innerHTML = messages.map(msg => `
                <tr>
                    <td>${new Date(msg.timestamp).toLocaleString()}</td>
                    <td>${msg.guild_id || 'DM'}</td>
                    <td><strong>${msg.username}</strong></td>
                    <td class="message-preview">${escapeHtml(msg.user_message || '')}</td>
                    <td class="message-preview">${escapeHtml(msg.bot_response || '')}</td>
                    <td>
                        ${msg.has_images ? '<span class="badge badge-image">📸 Image</span>' : ''}
                        ${msg.has_documents ? '<span class="badge badge-doc">📄 Doc</span>' : ''}
                        ${msg.search_query ? '<span class="badge badge-search">🔍 Search</span>' : ''}
                        ${!msg.has_images && !msg.has_documents && !msg.search_query ? '-' : ''}
                    </td>
                </tr>
            `).join('');
        }
        
        function filterMessages() {
            const search = document.getElementById('msg-search').value.toLowerCase();
            const typeFilter = document.getElementById('msg-type-filter').value;
            let filtered = allMessages;
            
            if (search) {
                filtered = filtered.filter(m => 
                    (m.user_message || '').toLowerCase().includes(search) ||
                    (m.bot_response || '').toLowerCase().includes(search)
                );
            }
            
            if (typeFilter === 'image') filtered = filtered.filter(m => m.has_images);
            else if (typeFilter === 'doc') filtered = filtered.filter(m => m.has_documents);
            else if (typeFilter === 'search') filtered = filtered.filter(m => m.search_query);
            
            displayMessages(filtered);
        }
        
        function displayPagination(totalPages, current) {
            const pagination = document.getElementById('messages-pagination');
            if (totalPages <= 1) {
                pagination.innerHTML = '';
                return;
            }
            
            let html = '';
            if (current > 1) html += `<button onclick="loadMessages(${current - 1})">Previous</button>`;
            for (let i = 1; i <= totalPages; i++) {
                if (i === 1 || i === totalPages || (i >= current - 2 && i <= current + 2)) {
                    html += `<button class="${i === current ? 'active' : ''}" onclick="loadMessages(${i})">${i}</button>`;
                } else if (i === current - 3 || i === current + 3) {
                    html += `<button disabled>...</button>`;
                }
            }
            if (current < totalPages) html += `<button onclick="loadMessages(${current + 1})">Next</button>`;
            pagination.innerHTML = html;
        }
        
        async function loadMemory() {
            try {
                const serverId = document.getElementById('mem-server-filter').value;
                const response = await fetch('/admin/api/memory' + (serverId ? '?guild_id=' + serverId : ''));
                allMemory = await response.json();
                displayMemory(allMemory);
            } catch (error) {
                console.error('Error loading memory:', error);
            }
        }
        
        function displayMemory(memory) {
            const container = document.getElementById('memory-list');
            if (memory.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #8b949e; padding: 40px;">No memory records found.</p>';
                return;
            }
            
            container.innerHTML = memory.map(m => `
                <div class="server-card">
                    <div class="server-header">
                        <div class="server-name">${m.username}</div>
                        <div>User ID: ${m.user_id}</div>
                    </div>
                    <div style="margin-top: 15px;">
                        <strong>Memory Summary:</strong>
                        <p style="color: #8b949e; margin-top: 5px;">${m.memory_summary || 'No summary available'}</p>
                        <details style="margin-top: 10px;">
                            <summary style="color: #58a6ff; cursor: pointer;">View Full Memory Profile</summary>
                            <pre style="background: #0d1117; padding: 15px; border-radius: 6px; margin-top: 10px; overflow-x: auto;">${JSON.stringify(m, null, 2)}</pre>
                        </details>
                    </div>
                </div>
            `).join('');
        }
        
        function filterMemory() {
            const search = document.getElementById('mem-search').value.toLowerCase();
            const filtered = allMemory.filter(m => 
                m.username.toLowerCase().includes(search) ||
                m.user_id.toLowerCase().includes(search) ||
                (m.memory_summary || '').toLowerCase().includes(search)
            );
            displayMemory(filtered);
        }
        
        function populateServerFilters() {
            const selects = ['user-server-filter', 'msg-server-filter', 'mem-server-filter'];
            selects.forEach(selectId => {
                const select = document.getElementById(selectId);
                select.innerHTML = '<option value="">All Servers</option>' +
                    allServers.map(s => `<option value="${s.guild_id}">${s.guild_info?.name || s.guild_id}</option>`).join('');
            });
        }
        
        function viewServerDetail(guildId) {
            window.location.href = '/admin/server/' + guildId;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Load initial data
        loadServers();
    </script>
</body>
</html>
    """

# Routes
@app.route('/')
def index():
    """Public homepage"""
    try:
        data = run_async(get_public_data())
        return render_template_string(
            PUBLIC_HTML,
            stats=data['stats'],
            servers=data['servers'],
            discord_invite=DISCORD_INVITE
        )
    except Exception as e:
        import traceback
        return f"<pre style='color: red;'>{str(e)}\n\n{traceback.format_exc()}</pre>", 500

async def get_public_data():
    """Fetch public data for homepage"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("⚠️ DATABASE_URL not set")
        return {'stats': {'unique_servers': 0, 'total_interactions': 0, 'unique_users': 0, 'memory_records': 0}, 'servers': []}
    
    discord_token = os.getenv('DISCORD_TOKEN')
    if not discord_token:
        print("⚠️ DISCORD_TOKEN not set - server names will not be available")
    else:
        print(f"✅ DISCORD_TOKEN is set (length: {len(discord_token)})")
    
    conn = await asyncpg.connect(database_url)
    
    try:
        # Overall stats
        stats = {
            'total_interactions': await conn.fetchval('SELECT COUNT(*) FROM interactions') or 0,
            'unique_users': await conn.fetchval('SELECT COUNT(DISTINCT user_id) FROM interactions') or 0,
            'memory_records': await conn.fetchval('SELECT COUNT(*) FROM user_memory') or 0,
            'unique_servers': await conn.fetchval('SELECT COUNT(DISTINCT guild_id) FROM interactions WHERE guild_id IS NOT NULL') or 0,
        }
        
        # Get servers with interaction counts
        servers = await conn.fetch('''
            SELECT 
                guild_id,
                COUNT(*) as interaction_count,
                COUNT(DISTINCT user_id) as unique_users,
                MAX(timestamp) as last_activity
            FROM interactions
            WHERE guild_id IS NOT NULL
            GROUP BY guild_id
            ORDER BY interaction_count DESC
            LIMIT 50
        ''')
        
        servers_list = []
        for server in servers:
            server_dict = dict(server)
            # Fetch Discord guild info
            try:
                guild_info = await get_discord_guild_info(server_dict['guild_id'])
                server_dict['guild_info'] = guild_info
                if guild_info and guild_info.get('name'):
                    print(f"✅ Fetched guild info for {server_dict['guild_id']}: {guild_info.get('name')}")
                else:
                    print(f"⚠️ No guild name for {server_dict['guild_id']} (guild_info: {guild_info})")
            except Exception as e:
                print(f"❌ Exception fetching guild info for {server_dict['guild_id']}: {e}")
                import traceback
                traceback.print_exc()
                server_dict['guild_info'] = {'name': None, 'id': server_dict['guild_id']}
            servers_list.append(server_dict)
        
        return {
            'stats': stats,
            'servers': servers_list
        }
    finally:
        await conn.close()

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template_string(ADMIN_LOGIN_HTML, error='Invalid username or password')
    
    return render_template_string(ADMIN_LOGIN_HTML)

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    return render_template_string(get_admin_dashboard_html(), username=session.get('admin_username', 'Admin'))

@app.route('/admin/api/servers')
@admin_required
def api_servers():
    """API: Get all servers"""
    try:
        data = run_async(get_admin_servers())
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

async def get_admin_servers():
    """Get all servers with details for admin"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return []
    
    conn = await asyncpg.connect(database_url)
    
    try:
        servers = await conn.fetch('''
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
        
        servers_list = []
        for server in servers:
            server_dict = dict(server)
            guild_info = await get_discord_guild_info(server_dict['guild_id'])
            server_dict['guild_info'] = guild_info
            servers_list.append(server_dict)
        
        return servers_list
    finally:
        await conn.close()

@app.route('/admin/api/users')
@admin_required
def api_users():
    """API: Get all users"""
    try:
        guild_id = request.args.get('guild_id')
        data = run_async(get_admin_users(guild_id))
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

async def get_admin_users(guild_id=None):
    """Get all users with memory for admin"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return []
    
    conn = await asyncpg.connect(database_url)
    
    try:
        if guild_id:
            query = '''
                SELECT DISTINCT
                    i.user_id,
                    i.username,
                    COUNT(*) as interaction_count,
                    MAX(i.timestamp) as last_interaction
                FROM interactions i
                WHERE i.guild_id = $1
                GROUP BY i.user_id, i.username
                ORDER BY interaction_count DESC
            '''
            users = await conn.fetch(query, guild_id)
        else:
            query = '''
                SELECT 
                    user_id,
                    username,
                    COUNT(*) as interaction_count,
                    MAX(timestamp) as last_interaction
                FROM interactions
                GROUP BY user_id, username
                ORDER BY interaction_count DESC
                LIMIT 1000
            '''
            users = await conn.fetch(query)
        
        users_list = []
        for user in users:
            user_dict = dict(user)
            # Get memory
            memory = await conn.fetchrow('SELECT * FROM user_memory WHERE user_id = $1', user_dict['user_id'])
            user_dict['memory'] = dict(memory) if memory else None
            users_list.append(user_dict)
        
        return users_list
    finally:
        await conn.close()

@app.route('/admin/api/messages')
@admin_required
def api_messages():
    """API: Get messages with pagination"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        guild_id = request.args.get('guild_id')
        user_id = request.args.get('user_id')
        
        data = run_async(get_admin_messages(page, limit, guild_id, user_id))
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

async def get_admin_messages(page=1, limit=50, guild_id=None, user_id=None):
    """Get messages with pagination"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return {'messages': [], 'total': 0}
    
    conn = await asyncpg.connect(database_url)
    
    try:
        offset = (page - 1) * limit
        conditions = []
        params = []
        param_count = 1
        
        if guild_id:
            conditions.append(f'guild_id = ${param_count}')
            params.append(guild_id)
            param_count += 1
        
        if user_id:
            conditions.append(f'user_id = ${param_count}')
            params.append(user_id)
            param_count += 1
        
        where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''
        
        # Get total count
        count_query = f'SELECT COUNT(*) FROM interactions {where_clause}'
        total = await conn.fetchval(count_query, *params) or 0
        
        # Get messages
        query = f'''
            SELECT * FROM interactions
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_count} OFFSET ${param_count + 1}
        '''
        params.extend([limit, offset])
        
        messages = await conn.fetch(query, *params)
        
        return {
            'messages': [dict(m) for m in messages],
            'total': total,
            'page': page,
            'limit': limit
        }
    finally:
        await conn.close()

@app.route('/admin/api/memory')
@admin_required
def api_memory():
    """API: Get memory logs"""
    try:
        guild_id = request.args.get('guild_id')
        data = run_async(get_admin_memory(guild_id))
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

async def get_admin_memory(guild_id=None):
    """Get memory logs"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return []
    
    conn = await asyncpg.connect(database_url)
    
    try:
        if guild_id:
            # Get users from this guild first
            user_ids = await conn.fetch('''
                SELECT DISTINCT user_id FROM interactions WHERE guild_id = $1
            ''', guild_id)
            user_id_list = [u['user_id'] for u in user_ids]
            
            if not user_id_list:
                return []
            
            query = '''
                SELECT * FROM user_memory
                WHERE user_id = ANY($1::text[])
                ORDER BY last_interaction DESC
            '''
            memory = await conn.fetch(query, user_id_list)
        else:
            query = 'SELECT * FROM user_memory ORDER BY last_interaction DESC LIMIT 1000'
            memory = await conn.fetch(query)
        
        return [dict(m) for m in memory]
    finally:
        await conn.close()

@app.route('/admin/server/<guild_id>')
@admin_required
def admin_server_detail(guild_id):
    """Admin server detail page"""
    try:
        data = run_async(get_server_detail_data(guild_id))
        guild_info = get_discord_guild_info_sync(guild_id)
        
        # Create a detailed view HTML (similar to dashboard.py server view)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Server Details - {guild_info.get('name', guild_id) if guild_info else guild_id}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0d1117;
                    color: #c9d1d9;
                    padding: 20px;
                }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                header {{
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #30363d;
                }}
                .btn {{
                    padding: 10px 20px;
                    border-radius: 6px;
                    text-decoration: none;
                    display: inline-block;
                    margin-right: 10px;
                    background: #30363d;
                    color: #c9d1d9;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                    padding: 20px;
                }}
                .stat-number {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #58a6ff;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #21262d;
                }}
                th {{
                    background: #0d1117;
                    color: #58a6ff;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <a href="/admin" class="btn">← Back to Dashboard</a>
                    <h1 style="color: #58a6ff; margin-top: 15px;">{guild_info.get('name', 'Server') if guild_info else 'Server'} ({guild_id})</h1>
                </header>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{data.get('server_stats', {}).get('total_interactions', 0)}</div>
                        <div>Total Interactions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{data.get('server_stats', {}).get('unique_users', 0)}</div>
                        <div>Unique Users</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{data.get('server_stats', {}).get('member_count', '?') if guild_info else '?'}</div>
                        <div>Members</div>
                    </div>
                </div>
                <h2 style="color: #79c0ff; margin: 30px 0 15px 0;">Recent Interactions</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>User</th>
                            <th>Message</th>
                            <th>Response</th>
                        </tr>
                    </thead>
                    <tbody>
                        {' '.join([f"<tr><td>{m.get('timestamp')}</td><td>{m.get('username')}</td><td>{m.get('user_message', '')[:100]}</td><td>{m.get('bot_response', '')[:100]}</td></tr>" for m in data.get('recent_interactions', [])[:50]])}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"<pre style='color: red;'>{str(e)}</pre>", 500

async def get_server_detail_data(guild_id):
    """Get detailed server data"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return {}
    
    conn = await asyncpg.connect(database_url)
    
    try:
        server_stats = await conn.fetchrow('''
            SELECT 
                COUNT(*) as total_interactions,
                COUNT(DISTINCT user_id) as unique_users
            FROM interactions
            WHERE guild_id = $1
        ''', guild_id)
        
        recent_interactions = await conn.fetch('''
            SELECT * FROM interactions
            WHERE guild_id = $1
            ORDER BY timestamp DESC
            LIMIT 50
        ''', guild_id)
        
        return {
            'server_stats': dict(server_stats) if server_stats else {},
            'recent_interactions': [dict(r) for r in recent_interactions]
        }
    finally:
        await conn.close()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
