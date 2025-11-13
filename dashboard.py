"""
Simple web dashboard to view bot's database/memory
Deploy separately on Railway with a domain
"""
from flask import Flask, render_template_string, jsonify
import asyncpg
import os
import asyncio
from datetime import datetime
import json

app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ServerMate Database Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #58a6ff; }
        h2 { color: #79c0ff; margin-top: 40px; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
        }
        .stat-card h3 { margin: 0 0 10px 0; color: #8b949e; font-size: 14px; }
        .stat-card .number { font-size: 32px; font-weight: bold; color: #58a6ff; }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border: 1px solid #30363d;
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
        }
        tr:hover { background: #1c2128; }
        .memory-block {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
        }
        .memory-block h4 { margin: 0 0 10px 0; color: #79c0ff; }
        pre {
            background: #0d1117;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
        }
        .timestamp { color: #8b949e; font-size: 12px; }
        .refresh-btn {
            background: #238636;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin: 20px 0;
        }
        .refresh-btn:hover { background: #2ea043; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 ServerMate Database Dashboard</h1>
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
        
        <div class="stats">
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
            <div class="stat-card">
                <h3>Consciousness Entries</h3>
                <div class="number">{{ stats.consciousness_entries }}</div>
            </div>
        </div>

        <h2>📝 Recent Interactions</h2>
        <table>
            <thead>
                <tr>
                    <th>User</th>
                    <th>Message</th>
                    <th>Response</th>
                    <th>Images</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {% for interaction in interactions %}
                <tr>
                    <td><strong>{{ interaction.username }}</strong></td>
                    <td>{{ interaction.user_message[:100] }}{% if interaction.user_message|length > 100 %}...{% endif %}</td>
                    <td>{{ interaction.bot_response[:100] }}{% if interaction.bot_response|length > 100 %}...{% endif %}</td>
                    <td>{% if interaction.has_images %}📸{% else %}-{% endif %}</td>
                    <td class="timestamp">{{ interaction.timestamp.strftime('%Y-%m-%d %H:%M') }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2>👥 User Memories</h2>
        {% for user in user_memories %}
        <div class="memory-block">
            <h4>{{ user.username }} ({{ user.interaction_count }} interactions)</h4>
            <p><strong>First seen:</strong> <span class="timestamp">{{ user.first_interaction.strftime('%Y-%m-%d') }}</span></p>
            <p><strong>Last seen:</strong> <span class="timestamp">{{ user.last_interaction.strftime('%Y-%m-%d %H:%M') }}</span></p>
            {% if user.memory_summary %}
            <p><strong>Summary:</strong> {{ user.memory_summary }}</p>
            {% endif %}
            {% if user.personality_profile %}
            <details>
                <summary><strong>Full Memory Profile (Click to expand)</strong></summary>
                <pre>{{ user.personality_profile }}</pre>
            </details>
            {% endif %}
        </div>
        {% endfor %}

        <h2>🧠 Consciousness Stream (Recent Thoughts)</h2>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Thought</th>
                    <th>Emotional State</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {% for thought in consciousness %}
                <tr>
                    <td>{{ thought.thought_type }}</td>
                    <td>{{ thought.content[:150] }}{% if thought.content|length > 150 %}...{% endif %}</td>
                    <td>{{ thought.emotional_state or '-' }}</td>
                    <td class="timestamp">{{ thought.timestamp.strftime('%H:%M:%S') }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2>📚 Learned Behaviors</h2>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Confidence</th>
                    <th>Learned From</th>
                </tr>
            </thead>
            <tbody>
                {% for behavior in behaviors %}
                <tr>
                    <td>{{ behavior.behavior_type }}</td>
                    <td>{{ behavior.description }}</td>
                    <td>{{ (behavior.confidence_score * 100)|round }}%</td>
                    <td class="timestamp">{{ behavior.timestamp.strftime('%Y-%m-%d') }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <p class="timestamp" style="text-align: center; margin-top: 40px;">
            Last updated: {{ now.strftime('%Y-%m-%d %H:%M:%S UTC') }}
        </p>
    </div>
</body>
</html>
"""

async def get_db_data():
    """Fetch all database data"""
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    
    # Stats
    stats = {
        'total_interactions': await conn.fetchval('SELECT COUNT(*) FROM interactions'),
        'unique_users': await conn.fetchval('SELECT COUNT(DISTINCT user_id) FROM interactions'),
        'memory_records': await conn.fetchval('SELECT COUNT(*) FROM user_memory'),
        'consciousness_entries': await conn.fetchval('SELECT COUNT(*) FROM consciousness_stream'),
    }
    
    # Recent interactions
    interactions = await conn.fetch('''
        SELECT * FROM interactions 
        ORDER BY timestamp DESC 
        LIMIT 20
    ''')
    
    # User memories
    user_memories = await conn.fetch('''
        SELECT * FROM user_memory 
        ORDER BY last_interaction DESC
    ''')
    
    # Process memories to format JSON
    formatted_memories = []
    for mem in user_memories:
        mem_dict = dict(mem)
        if mem_dict['personality_profile']:
            # Pretty print JSON
            if isinstance(mem_dict['personality_profile'], str):
                mem_dict['personality_profile'] = mem_dict['personality_profile']
            else:
                mem_dict['personality_profile'] = json.dumps(mem_dict['personality_profile'], indent=2)
        formatted_memories.append(mem_dict)
    
    # Consciousness stream
    consciousness = await conn.fetch('''
        SELECT * FROM consciousness_stream 
        ORDER BY timestamp DESC 
        LIMIT 30
    ''')
    
    # Learned behaviors
    behaviors = await conn.fetch('''
        SELECT * FROM learned_behaviors 
        ORDER BY timestamp DESC
    ''')
    
    await conn.close()
    
    return {
        'stats': stats,
        'interactions': [dict(r) for r in interactions],
        'user_memories': formatted_memories,
        'consciousness': [dict(r) for r in consciousness],
        'behaviors': [dict(r) for r in behaviors],
    }

@app.route('/')
def index():
    """Main dashboard page"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(get_db_data())
    
    return render_template_string(
        HTML_TEMPLATE,
        stats=data['stats'],
        interactions=data['interactions'],
        user_memories=data['user_memories'],
        consciousness=data['consciousness'],
        behaviors=data['behaviors'],
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

