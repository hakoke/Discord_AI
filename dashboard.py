"""
ServerMate Dashboard - Server-focused analytics
Simple, clean dashboard to view bot statistics by server
"""
from flask import Flask, render_template_string, jsonify, request
import asyncpg
import os
import asyncio
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Add custom Jinja2 filters
@app.template_filter('tojson')
def tojson_filter(obj):
    """Convert object to JSON string"""
    return json.dumps(obj)

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
                            <strong style="color: #79c0ff; font-size: 18px;">Server {{ loop.index }}</strong>
                            <div class="server-id">{{ server.guild_id }}</div>
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
                const ctx = document.getElementById('usageChart').getContext('2d');
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
                const serverCtx = document.getElementById('serverUsageChart').getContext('2d');
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
            </script>
            {% else %}
            <p style="color: #8b949e; text-align: center; padding: 40px;">Server not found or has no data.</p>
            {% endif %}
        {% endif %}
        
        <p class="timestamp" style="text-align: center; margin-top: 40px;">
            Last updated: {{ now.strftime('%Y-%m-%d %H:%M:%S UTC') }}
        </p>
    </div>
</body>
</html>
"""

async def get_db_data():
    """Fetch all database data for home page"""
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    
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
        
        return {
            'stats': stats,
            'servers': [dict(s) for s in servers],
            'usage_data': [dict(u) for u in usage_data],
            'top_users': [dict(u) for u in top_users],
        }
    finally:
        await conn.close()

async def get_server_data(guild_id: str):
    """Fetch data for a specific server"""
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    
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
        
        return {
            'server_stats': dict(server_stats),
            'server_users': users_with_memory,
            'recent_interactions': [dict(r) for r in recent_interactions],
            'server_usage_data': [dict(u) for u in server_usage_data],
        }
    finally:
        await conn.close()

@app.route('/')
def index():
    """Main dashboard page - shows all servers"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(get_db_data())
    
    return render_template_string(
        HTML_TEMPLATE,
        view='home',
        stats=data['stats'],
        servers=data['servers'],
        usage_data=data['usage_data'],
        top_users=data['top_users'],
        now=datetime.utcnow()
    )

@app.route('/server/<guild_id>')
def server_detail(guild_id):
    """Server detail page"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(get_server_data(guild_id))
    
    if not data:
        return render_template_string(
            HTML_TEMPLATE,
            view='server',
            server_id=guild_id,
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
        server_stats=data['server_stats'],
        server_users=data['server_users'],
        recent_interactions=data['recent_interactions'],
        server_usage_data=data['server_usage_data'],
        now=datetime.utcnow()
    )

@app.route('/api/stats')
def api_stats():
    """API endpoint for stats"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(get_db_data())
    return jsonify(data['stats'])

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
