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
- **Login**: Username: `Hakoke`, Password: `Ironman`
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
   - `DATABASE_URL`: PostgreSQL connection string (already set)
   - `DISCORD_TOKEN`: Discord bot token (already set)
   - `FLASK_SECRET_KEY`: Secret key for sessions (optional, defaults to a development key)

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

- **Username**: Hakoke
- **Password**: Ironman

⚠️ **Important**: Change the admin credentials in production by modifying `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `website.py`

## Integration with Existing Dashboard

This website runs separately from `dashboard.py`. You can:
- Run both on different ports
- Or replace `dashboard.py` with this website
- Or run one or the other as needed

The website uses the same database and Discord API access as the bot.
