"""
ServerMate Dashboard - Server-focused analytics
Simple, clean dashboard to view bot statistics by server
"""
from flask import Flask, render_template_string, jsonify, request
import asyncpg
import os
import asyncio
from datetime import datetime, timedelta, date
import json
import aiohttp
from functools import lru_cache
import time

app = Flask(__name__)

# Discord API cache (guild_id -> {name, icon, cached_at})
discord_guild_cache = {}
CACHE_DURATION = 3600  # 1 hour cache

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
        url = f"https://discord.com/api/v10/guilds/{guild_id}"
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
                    # Guild not found or bot not in server
                    return {'name': 'Server Not Found', 'id': guild_id}
                else:
                    return None
    except Exception as e:
        print(f"Error fetching Discord guild info: {e}")
        return None

def get_discord_guild_info_sync(guild_id: str):
    """Synchronous wrapper for get_discord_guild_info"""
    # This will be defined later, but we'll call it directly in routes
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

# Add custom Jinja2 filters
@app.template_filter('tojson')
def tojson_filter(obj):
    """Convert object to JSON string, handling date/datetime objects"""
    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    return json.dumps(obj, default=json_serial)

# Main HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ServerMate Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 10px; }
        h2 { color: #79c0ff; margin: 30px 0 15px 0; }
        h3 { color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 8px; }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 15px;
        }
        .refresh-btn {
            background: #238636;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        .refresh-btn:hover { background: #2ea043; }
        .back-btn {
            background: #30363d;
            color: #c9d1d9;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
            display: inline-block;
            transition: background 0.2s;
        }
        .back-btn:hover { background: #21262d; }
        
        /* Stats Grid */
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            transition: border-color 0.2s;
        }
        .stat-card:hover { border-color: #58a6ff; }
        .stat-card .number {
            font-size: 32px;
            font-weight: bold;
            color: #58a6ff;
            margin-top: 5px;
        }
        
        /* Server List */
        .server-list {
            display: grid;
            gap: 15px;
            margin: 20px 0;
        }
        .server-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
            color: inherit;
            display: block;
        }
        .server-card:hover {
            border-color: #58a6ff;
            background: #1c2128;
        }
        .server-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .server-id {
            font-family: 'Courier New', monospace;
            color: #8b949e;
            font-size: 12px;
        }
        .server-stats {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 14px;
        }
        .server-stats span {
            color: #8b949e;
        }
        .server-stats strong {
            color: #58a6ff;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            overflow: hidden;
            margin: 20px 0;
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
        tr:last-child td { border-bottom: none; }
        tr:hover { background: #1c2128; }
        
        /* Memory Blocks */
        .memory-block {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
        }
        .memory-block h4 {
            margin: 0 0 10px 0;
            color: #79c0ff;
        }
        .memory-summary {
            color: #8b949e;
            font-size: 14px;
            margin: 8px 0;
        }
        .memory-details {
            margin-top: 10px;
        }
        details {
            margin-top: 10px;
        }
        summary {
            cursor: pointer;
            color: #58a6ff;
            font-size: 13px;
        }
        summary:hover { text-decoration: underline; }
        pre {
            background: #0d1117;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
            margin-top: 10px;
        }
        
        /* Charts */
        .chart-container {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }
        .chart-wrapper {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        
        /* User Cards */
        .user-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .user-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
        }
        .user-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .user-name {
            font-weight: 600;
            color: #79c0ff;
        }
        .user-stats {
            font-size: 12px;
            color: #8b949e;
            margin-top: 8px;
        }
        
        .timestamp { color: #8b949e; font-size: 12px; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            background: #238636;
            color: white;
            margin-left: 5px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #8b949e;
        }
        
        /* Ban controls */
        .ban-btn {
            background: #da3633;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        .ban-btn:hover { background: #f85149; }
        .unban-btn {
            background: #238636;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        .unban-btn:hover { background: #2ea043; }
        .ban-dropdown {
            position: relative;
            display: inline-block;
        }
        .ban-menu {
            display: none;
            position: absolute;
            right: 0;
            top: 100%;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px;
            margin-top: 5px;
            min-width: 200px;
            z-index: 1000;
        }
        .ban-menu.show { display: block; }
        .ban-option {
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 4px;
            margin: 4px 0;
            transition: background 0.2s;
        }
        .ban-option:hover { background: #21262d; }
        .ban-option.permanent { color: #f85149; }
        .ban-reason-input {
            width: 100%;
            padding: 8px;
            margin-top: 10px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 4px;
            color: #c9d1d9;
            font-size: 13px;
        }
        .ban-status-banner {
            background: #da3633;
            border: 1px solid #f85149;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 0;
            color: white;
        }
        .ban-status-banner.permanent { background: #8b2635; }
        .ban-status-banner.temporary { background: #7c2d12; }
    </style>
</head>
<body>
    <div class="container">
        {% if view == 'home' %}
            <div class="header">
                <h1>🧠 ServerMate Dashboard</h1>
                <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Servers</h3>
                    <div class="number">{{ stats.unique_servers or 0 }}</div>
                </div>
                <div class="stat-card">
                    <h3>Total Interactions</h3>
                    <div class="number">{{ stats.total_interactions }}</div>
                </div>
                <div class="stat-card">
                    <h3>Unique Users</h3>
                    <div class="number">{{ stats.unique_users }}</div>
                </div>
                <div class="stat-card">
                    <h3>Memory Records</h3>
                    <div class="number">{{ stats.memory_records }}</div>
                </div>
            </div>
            
            <h2>📊 Usage Over Time (Last 30 Days)</h2>
            <div class="chart-container">
                <canvas id="usageChart"></canvas>
            </div>
            
            <h2>🏆 Top Users (All Servers)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>User</th>
                        <th>Interactions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for user in top_users %}
                    <tr>
                        <td>#{{ loop.index }}</td>
                        <td><strong>{{ user.username }}</strong></td>
                        <td>{{ user.interaction_count }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <h2>🖥️ Servers</h2>
            {% if servers %}
            <div class="server-list">
                {% for server in servers %}
                <a href="/server/{{ server.guild_id }}" class="server-card">
                    <div class="server-card-header">
                        <div>
                            {% if server.guild_info and server.guild_info.name %}
                                <strong style="color: #79c0ff; font-size: 18px;">{{ server.guild_info.name }}</strong>
                            {% else %}
                                <strong style="color: #79c0ff; font-size: 18px;">Server {{ loop.index }}</strong>
                            {% endif %}
                            <div class="server-id">ID: {{ server.guild_id }}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 24px; color: #58a6ff; font-weight: bold;">{{ server.interaction_count }}</div>
                            <div style="font-size: 12px; color: #8b949e;">interactions</div>
                        </div>
                    </div>
                    <div class="server-stats">
                        <span><strong>{{ server.unique_users }}</strong> users</span>
                        <span>Last active: <strong>{{ server.last_activity.strftime('%Y-%m-%d %H:%M') if server.last_activity else 'N/A' }}</strong></span>
                    </div>
                </a>
                {% endfor %}
            </div>
            {% else %}
            <p style="color: #8b949e; text-align: center; padding: 40px;">No servers found. The bot needs to receive messages in servers first.</p>
            {% endif %}
            
            <script>
                // Usage chart
                const usageData = {{ usage_data | tojson }};
                const ctx = document.getElementById('usageChart');
                if (ctx && usageData && usageData.length > 0) {
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: usageData.map(d => new Date(d.date).toLocaleDateString()),
                            datasets: [{
                                label: 'Interactions',
                                data: usageData.map(d => d.count),
                                borderColor: '#58a6ff',
                                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: '#161b22',
                                titleColor: '#c9d1d9',
                                bodyColor: '#c9d1d9',
                                borderColor: '#30363d',
                                borderWidth: 1
                            }
                        },
                        scales: {
                            x: {
                                ticks: { color: '#8b949e' },
                                grid: { color: '#21262d' }
                            },
                            y: {
                                ticks: { color: '#8b949e' },
                                grid: { color: '#21262d' },
                                beginAtZero: true
                            }
                        }
                    }
                    });
                } else {
                    ctx.getContext('2d').fillText('No data available', 10, 50);
                }
            </script>
            
        {% elif view == 'server' %}
            <div class="header">
                <div>
                    <a href="/" class="back-btn">← Back to All Servers</a>
                    <h1 style="margin-top: 15px;">Server Details</h1>
                    <div class="server-id" style="margin-top: 5px;">{{ server_id }}</div>
                </div>
                <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
            </div>
            
            {% if server_stats %}
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Interactions</h3>
                    <div class="number">{{ server_stats.total_interactions }}</div>
                </div>
                <div class="stat-card">
                    <h3>Unique Users</h3>
                    <div class="number">{{ server_stats.unique_users }}</div>
                </div>
                <div class="stat-card">
                    <h3>Active Days</h3>
                    <div class="number">{{ server_stats.active_days }}</div>
                </div>
                <div class="stat-card">
                    <h3>Image Interactions</h3>
                    <div class="number">{{ server_stats.image_interactions or 0 }}</div>
                </div>
                <div class="stat-card">
                    <h3>Document Interactions</h3>
                    <div class="number">{{ server_stats.document_interactions or 0 }}</div>
                </div>
                <div class="stat-card">
                    <h3>Search Interactions</h3>
                    <div class="number">{{ server_stats.search_interactions or 0 }}</div>
                </div>
            </div>
            
            <h2>📊 Usage Over Time (Last 30 Days)</h2>
            <div class="chart-container">
                <canvas id="serverUsageChart"></canvas>
            </div>
            
            <h2>🏆 Top Users</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>User</th>
                        <th>Interactions</th>
                        <th>Images</th>
                        <th>Documents</th>
                        <th>Last Active</th>
                    </tr>
                </thead>
                <tbody>
                    {% for user in server_users %}
                    <tr>
                        <td>#{{ loop.index }}</td>
                        <td><strong>{{ user.username }}</strong></td>
                        <td>{{ user.interaction_count }}</td>
                        <td>{{ user.image_count or 0 }}</td>
                        <td>{{ user.document_count or 0 }}</td>
                        <td class="timestamp">{{ user.last_interaction.strftime('%Y-%m-%d %H:%M') if user.last_interaction else 'N/A' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <h2>👥 User Memories</h2>
            {% if server_users %}
            <div class="user-grid">
                {% for user in server_users %}
                <div class="user-card">
                    <div class="user-card-header">
                        <div>
                            <div class="user-name">{{ user.username }}</div>
                            <div class="user-stats">
                                {{ user.interaction_count }} interactions
                                {% if user.image_count %}<span class="badge">📸 {{ user.image_count }}</span>{% endif %}
                                {% if user.document_count %}<span class="badge">📄 {{ user.document_count }}</span>{% endif %}
                            </div>
                        </div>
                    </div>
                    {% if user.memory and user.memory.memory_summary %}
                    <div class="memory-block">
                        <div class="memory-summary">{{ user.memory.memory_summary }}</div>
                        <div class="memory-details">
                            <p class="timestamp">
                                First: {{ user.memory.first_interaction.strftime('%Y-%m-%d') if user.memory.first_interaction else 'N/A' }} | 
                                Last: {{ user.memory.last_interaction.strftime('%Y-%m-%d %H:%M') if user.memory.last_interaction else 'N/A' }}
                            </p>
                            {% if user.memory.personality_profile_json %}
                            <details>
                                <summary>Full Memory Profile</summary>
                                <pre>{{ user.memory.personality_profile_json }}</pre>
                            </details>
                            {% endif %}
                        </div>
                    </div>
                    {% else %}
                    <p style="color: #8b949e; font-size: 13px; margin-top: 10px;">No memory data yet</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="color: #8b949e; text-align: center; padding: 40px;">No users found in this server.</p>
            {% endif %}
            
            <h2>📝 Recent Interactions</h2>
            <table>
                <thead>
                    <tr>
                        <th>User</th>
                        <th>Message</th>
                        <th>Response</th>
                        <th>Type</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {% for interaction in recent_interactions %}
                    <tr>
                        <td><strong>{{ interaction.username }}</strong></td>
                        <td>{{ interaction.user_message[:80] }}{% if interaction.user_message|length > 80 %}...{% endif %}</td>
                        <td>{{ interaction.bot_response[:80] }}{% if interaction.bot_response|length > 80 %}...{% endif %}</td>
                        <td>
                            {% if interaction.has_images %}📸{% endif %}
                            {% if interaction.has_documents %}📄{% endif %}
                            {% if interaction.search_query %}🔍{% endif %}
                            {% if not interaction.has_images and not interaction.has_documents and not interaction.search_query %}-{% endif %}
                        </td>
                        <td class="timestamp">{{ interaction.timestamp.strftime('%Y-%m-%d %H:%M') if interaction.timestamp else 'N/A' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <script>
                // Server usage chart
                const serverUsageData = {{ server_usage_data | tojson }};
                const serverCtx = document.getElementById('serverUsageChart');
                if (serverCtx && serverUsageData && serverUsageData.length > 0) {
                    new Chart(serverCtx, {
                        type: 'line',
                        data: {
                            labels: serverUsageData.map(d => new Date(d.date).toLocaleDateString()),
                            datasets: [{
                                label: 'Interactions',
                                data: serverUsageData.map(d => d.count),
                                borderColor: '#58a6ff',
                                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: '#161b22',
                                titleColor: '#c9d1d9',
                                bodyColor: '#c9d1d9',
                                borderColor: '#30363d',
                                borderWidth: 1
                            }
                        },
                        scales: {
                            x: {
                                ticks: { color: '#8b949e' },
                                grid: { color: '#21262d' }
                            },
                            y: {
                                ticks: { color: '#8b949e' },
                                grid: { color: '#21262d' },
                                beginAtZero: true
                            }
                        }
                    }
                    });
                } else {
                    serverCtx.getContext('2d').fillText('No data available', 10, 50);
                }
            </script>
            {% else %}
            <p style="color: #8b949e; text-align: center; padding: 40px;">Server not found or has no data.</p>
            {% endif %}
        {% endif %}
        
        <p class="timestamp" style="text-align: center; margin-top: 40px;">
            Last updated: {{ now.strftime('%Y-%m-%d %H:%M:%S UTC') }}
        </p>
    </div>
    
    <script>
        // Server ban management (only on server detail page)
        {% if view == 'server' %}
        const guildId = '{{ server_id }}';
        
        async function loadBanStatus() {
            try {
                const response = await fetch(`/api/server/${guildId}/ban-status`);
                const data = await response.json();
                
                const banControls = document.getElementById('ban-controls');
                const banStatus = document.getElementById('ban-status');
                
                if (data.banned && data.ban_info) {
                    const ban = data.ban_info;
                    const isPermanent = ban.ban_type === 'permanent';
                    const expiresAt = ban.expires_at ? new Date(ban.expires_at) : null;
                    const isExpired = expiresAt && expiresAt < new Date();
                    
                    if (isExpired) {
                        // Ban expired, reload
                        location.reload();
                        return;
                    }
                    
                    // Show ban status
                    let statusHtml = `<div class="ban-status-banner ${ban.ban_type}">`;
                    statusHtml += `<strong>⚠️ Server is ${isPermanent ? 'PERMANENTLY BANNED' : 'TEMPORARILY BANNED'}</strong><br>`;
                    if (!isPermanent && expiresAt) {
                        statusHtml += `Expires: ${expiresAt.toLocaleString()}<br>`;
                    }
                    if (ban.reason) {
                        statusHtml += `Reason: ${ban.reason}<br>`;
                    }
                    statusHtml += `Banned at: ${new Date(ban.banned_at).toLocaleString()}`;
                    statusHtml += `</div>`;
                    banStatus.innerHTML = statusHtml;
                    
                    // Show unban button
                    banControls.innerHTML = `
                        <button class="unban-btn" onclick="unbanServer()">✅ Unban Server</button>
                    `;
                } else {
                    // Show ban button
                    banControls.innerHTML = `
                        <div class="ban-dropdown">
                            <button class="ban-btn" onclick="toggleBanMenu()">🚫 Remove AI from Server</button>
                            <div id="ban-menu" class="ban-menu">
                                <div class="ban-option" onclick="banServer('temporary', 7)">Temporary (7 days)</div>
                                <div class="ban-option" onclick="banServer('temporary', 30)">Temporary (30 days)</div>
                                <div class="ban-option" onclick="banServerCustom()">Temporary (Custom days)</div>
                                <div class="ban-option permanent" onclick="banServer('permanent')">Permanent (Forever)</div>
                                <input type="text" id="ban-reason" class="ban-reason-input" placeholder="Optional reason (leave empty for no reason)">
                            </div>
                        </div>
                    `;
                    banStatus.innerHTML = '';
                }
            } catch (error) {
                console.error('Error loading ban status:', error);
            }
        }
        
        function toggleBanMenu() {
            const menu = document.getElementById('ban-menu');
            menu.classList.toggle('show');
        }
        
        function banServer(type, days = null) {
            const reason = document.getElementById('ban-reason').value.trim() || null;
            const daysInput = days || parseInt(prompt('Enter number of days:') || '7');
            
            if (isNaN(daysInput) && type === 'temporary') {
                alert('Invalid number of days');
                return;
            }
            
            fetch(`/api/server/${guildId}/ban`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    type: type,
                    days: type === 'temporary' ? daysInput : null,
                    reason: reason
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Server banned successfully');
                    location.reload();
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                alert('Error: ' + error.message);
            });
        }
        
        function banServerCustom() {
            const days = parseInt(prompt('Enter number of days for temporary ban:') || '7');
            if (!isNaN(days) && days > 0) {
                banServer('temporary', days);
            }
        }
        
        function unbanServer() {
            if (!confirm('Are you sure you want to unban this server?')) {
                return;
            }
            
            fetch(`/api/server/${guildId}/unban`, {
                method: 'POST'
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Server unbanned successfully');
                    location.reload();
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                alert('Error: ' + error.message);
            });
        }
        
        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            const menu = document.getElementById('ban-menu');
            if (menu && !menu.contains(e.target) && !e.target.closest('.ban-dropdown')) {
                menu.classList.remove('show');
            }
        });
        
        // Load ban status on page load
        if (guildId) {
            loadBanStatus();
        }
        {% endif %}
    </script>
</body>
</html>
"""

async def get_db_data():
    """Fetch all database data for home page"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    conn = await asyncpg.connect(database_url)
    
    try:
        # Overall stats
        stats = {
            'total_interactions': await conn.fetchval('SELECT COUNT(*) FROM interactions'),
            'unique_users': await conn.fetchval('SELECT COUNT(DISTINCT user_id) FROM interactions'),
            'memory_records': await conn.fetchval('SELECT COUNT(*) FROM user_memory'),
            'unique_servers': await conn.fetchval('SELECT COUNT(DISTINCT guild_id) FROM interactions WHERE guild_id IS NOT NULL'),
        }
        
        # All servers
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
        
        # Usage over time
        usage_data = await conn.fetch('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count
            FROM interactions
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        ''')
        
        # Top users
        top_users = await conn.fetch('''
            SELECT 
                user_id,
                username,
                COUNT(*) as interaction_count
            FROM interactions
            GROUP BY user_id, username
            ORDER BY interaction_count DESC
            LIMIT 10
        ''')
        
        # Convert date objects (not datetime) to strings for JSON serialization
        # Keep datetime objects for template .strftime() calls
        def convert_dates(obj):
            """Recursively convert date objects (not datetime) to ISO format strings"""
            if isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            elif isinstance(obj, date) and not isinstance(obj, datetime):
                # Convert date objects to strings (for chart data)
                return obj.isoformat()
            # Keep datetime objects for template .strftime()
            return obj
        
        servers_list = [dict(s) for s in servers]
        
        # Fetch Discord server info for each server
        for server in servers_list:
            guild_id = server.get('guild_id')
            if guild_id:
                guild_info = await get_discord_guild_info(guild_id)
                server['guild_info'] = guild_info
        
        usage_data_list = [dict(u) for u in usage_data]
        top_users_list = [dict(u) for u in top_users]
        
        return {
            'stats': stats,
            'servers': convert_dates(servers_list),
            'usage_data': convert_dates(usage_data_list),
            'top_users': convert_dates(top_users_list),
        }
    finally:
        await conn.close()

async def get_server_data(guild_id: str):
    """Fetch data for a specific server"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    conn = await asyncpg.connect(database_url)
    
    try:
        # Server stats
        server_stats = await conn.fetchrow('''
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
        
        if not server_stats:
            return None
        
        # Server users with memory
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
                # Format JSON fields - ensure personality_profile is a dict or None
                if user_dict['memory'].get('personality_profile'):
                    if isinstance(user_dict['memory']['personality_profile'], str):
                        try:
                            user_dict['memory']['personality_profile'] = json.loads(user_dict['memory']['personality_profile'])
                        except:
                            user_dict['memory']['personality_profile'] = None
                # Convert to JSON string for template display
                if user_dict['memory'].get('personality_profile'):
                    user_dict['memory']['personality_profile_json'] = json.dumps(user_dict['memory']['personality_profile'], indent=2)
            else:
                user_dict['memory'] = None
            
            users_with_memory.append(user_dict)
        
        # Recent interactions
        recent_interactions = await conn.fetch('''
            SELECT * FROM interactions
            WHERE guild_id = $1
            ORDER BY timestamp DESC
            LIMIT 50
        ''', guild_id)
        
        # Usage over time for this server
        server_usage_data = await conn.fetch('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count
            FROM interactions
            WHERE guild_id = $1
                AND timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        ''', guild_id)
        
        # Convert date objects (not datetime) to strings for JSON serialization
        # Keep datetime objects for template .strftime() calls
        def convert_dates(obj):
            """Recursively convert date objects (not datetime) to ISO format strings"""
            if isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            elif isinstance(obj, date) and not isinstance(obj, datetime):
                # Convert date objects to strings (for chart data)
                return obj.isoformat()
            # Keep datetime objects for template .strftime()
            return obj
        
        server_stats_dict = dict(server_stats)
        recent_interactions_list = [dict(r) for r in recent_interactions]
        server_usage_data_list = [dict(u) for u in server_usage_data]
        
        return {
            'server_stats': convert_dates(server_stats_dict),
            'server_users': convert_dates(users_with_memory),
            'recent_interactions': convert_dates(recent_interactions_list),
            'server_usage_data': convert_dates(server_usage_data_list),
        }
    finally:
        await conn.close()

def run_async(coro):
    """Run async function in sync context - works with gunicorn"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(coro)
    finally:
        # Don't close the loop - gunicorn workers reuse it
        pass

@app.route('/')
def index():
    """Main dashboard page - shows all servers"""
    try:
        data = run_async(get_db_data())
        
        return render_template_string(
            HTML_TEMPLATE,
            view='home',
            stats=data['stats'],
            servers=data['servers'],
            usage_data=data['usage_data'],
            top_users=data['top_users'],
            now=datetime.utcnow()
        )
    except Exception as e:
        import traceback
        error_msg = f"Error loading dashboard: {str(e)}\n\n{traceback.format_exc()}"
        return f"<pre style='color: red; background: #0d1117; padding: 20px;'>{error_msg}</pre>", 500

@app.route('/server/<guild_id>')
def server_detail(guild_id):
    """Server detail page"""
    try:
        data = run_async(get_server_data(guild_id))
        
        # Fetch Discord server name
        guild_info = get_discord_guild_info_sync(guild_id)
        server_name = guild_info.get('name') if guild_info else None
        
        if not data:
            return render_template_string(
                HTML_TEMPLATE,
                view='server',
                server_id=guild_id,
                server_name=server_name,
                server_stats=None,
                server_users=[],
                recent_interactions=[],
                server_usage_data=[],
                now=datetime.utcnow()
            )
        
        return render_template_string(
            HTML_TEMPLATE,
            view='server',
            server_id=guild_id,
            server_name=server_name,
            server_stats=data['server_stats'],
            server_users=data['server_users'],
            recent_interactions=data['recent_interactions'],
            server_usage_data=data['server_usage_data'],
            now=datetime.utcnow()
        )
    except Exception as e:
        import traceback
        error_msg = f"Error loading server data: {str(e)}\n\n{traceback.format_exc()}"
        return f"<pre style='color: red; background: #0d1117; padding: 20px;'>{error_msg}</pre>", 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for stats"""
    try:
        data = run_async(get_db_data())
        return jsonify(data['stats'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/server/<guild_id>/ban', methods=['POST'])
def ban_server(guild_id):
    """Ban a server (temporary or permanent)"""
    try:
        data = request.get_json()
        ban_type = data.get('type', 'temporary')  # 'temporary' or 'permanent'
        days = data.get('days', 7) if ban_type == 'temporary' else None
        reason = data.get('reason', '').strip() or None
        banned_by = data.get('banned_by', 'Dashboard Admin')
        
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return jsonify({'error': 'DATABASE_URL not set'}), 500
        
        async def ban():
            conn = await asyncpg.connect(database_url)
            try:
                from datetime import timedelta
                expires_at = None
                if ban_type == 'temporary' and days:
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
            finally:
                await conn.close()
        
        run_async(ban())
        return jsonify({'success': True, 'message': f'Server banned ({ban_type})'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/server/<guild_id>/unban', methods=['POST'])
def unban_server(guild_id):
    """Unban a server"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return jsonify({'error': 'DATABASE_URL not set'}), 500
        
        async def unban():
            conn = await asyncpg.connect(database_url)
            try:
                await conn.execute('DELETE FROM server_bans WHERE guild_id = $1', guild_id)
            finally:
                await conn.close()
        
        run_async(unban())
        return jsonify({'success': True, 'message': 'Server unbanned'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/server/<guild_id>/ban-status', methods=['GET'])
def get_ban_status(guild_id):
    """Get ban status for a server"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return jsonify({'error': 'DATABASE_URL not set'}), 500
        
        async def get_ban():
            conn = await asyncpg.connect(database_url)
            try:
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
            finally:
                await conn.close()
        
        ban_info = run_async(get_ban())
        if ban_info:
            # Convert datetime to ISO string for JSON
            if ban_info.get('banned_at'):
                ban_info['banned_at'] = ban_info['banned_at'].isoformat()
            if ban_info.get('expires_at'):
                ban_info['expires_at'] = ban_info['expires_at'].isoformat()
        return jsonify({'banned': ban_info is not None, 'ban_info': ban_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
