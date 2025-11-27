# ServerMate Official Website

The official public-facing website for ServerMate with an admin dashboard.

## Features

### Public Website
- Beautiful homepage showcasing ServerMate capabilities
- Server list showing all servers using the bot with member counts
- Stats display (total servers, interactions, users, memory records)
- Discord invite link
- Responsive design

### Admin Dashboard
- **Features**:
  - View all servers with detailed stats
  - View all users and their interaction counts
  - View all messages with search and filters
  - View memory logs with full memory profiles
  - Server detail pages with full interaction history
  - Search and filter functionality across all sections
  - Pagination for messages

## Setup

1. **Profile Picture**: 
   - Place `profile.png` in the workspace root directory
   - This is the ServerMate bot profile picture
   - If not present, the logo will not display (gracefully handled)

2. **Environment Variables**:
   - `DATABASE_URL`: PostgreSQL connection string (required)
   - `DISCORD_TOKEN`: Discord bot token (required)
   - `FLASK_SECRET_KEY`: Secret key for sessions (required)
   - `ADMIN_USERNAME`: Admin dashboard username (required)
   - `ADMIN_PASSWORD`: Admin dashboard password (required)

3. **Run the Website**:
   ```bash
   python website.py
   ```
   
   Or with gunicorn:
   ```bash
   gunicorn website:app -b 0.0.0.0:5000
   ```

## Routes

- `/` - Public homepage
- `/admin/login` - Admin login page
- `/admin` - Admin dashboard (requires login)
- `/admin/server/<guild_id>` - Detailed server view
- `/admin/api/*` - API endpoints for admin dashboard data

## Admin Credentials

Admin credentials must be set via environment variables:
- `ADMIN_USERNAME`: Set your admin username
- `ADMIN_PASSWORD`: Set your admin password

⚠️ **Important**: Never commit credentials to the repository. Always use environment variables.

## Integration with Existing Dashboard

This website runs separately from `dashboard.py`. You can:
- Run both on different ports
- Or replace `dashboard.py` with this website
- Or run one or the other as needed

The website uses the same database and Discord API access as the bot.
