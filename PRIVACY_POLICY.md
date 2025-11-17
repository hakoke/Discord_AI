## ServerMate Discord Bot — Privacy Policy

_Last updated: November 17, 2025_

This Privacy Policy explains how the ServerMate Discord bot (the “Bot”) processes information when operating in your Discord server.

### 1. Data Controller
For self-hosted deployments, the server owner or organization running the Bot acts as the data controller. If you deploy ServerMate, you are responsible for complying with applicable privacy laws and for honoring user rights requests.

### 2. Information We Collect
When the Bot is active, it may collect and store:

- **Discord Metadata**: Server ID, channel ID, message ID, user ID, usernames, timestamps, attachments, and reactions received through Discord’s APIs.
- **Message Content**: Text, prompts, and commands you or your members send to the Bot (mentions, replies, direct messages, slash commands, custom commands, etc.).
- **AI-Generated Content**: Bot responses, image generation prompts, analysis summaries, and internal “consciousness stream” entries.
- **Memory Records**: Personality notes, preferences, relationships, and behavior patterns stored in PostgreSQL as described in `README.md`.
- **System Logs**: Errors, performance metrics, and request metadata necessary to operate the Bot (e.g., API latency, model choices).
- **Optional Web Search Data**: If Serper search is enabled, the Bot sends user prompts to Serper’s API.

### 3. How We Use Information
Data is processed to:
- Generate conversational responses and execute commands.
- Choose appropriate AI models (Gemini 2.0 Flash, Gemini 2.5 Pro, etc.).
- Provide image analysis, generation, and editing via Vertex AI / Imagen.
- Maintain user-specific memory that makes the Bot more contextual over time.
- Monitor performance, troubleshoot issues, and prevent abuse.
- Satisfy legal obligations and enforce the Terms of Service.

### 4. Legal Bases
Depending on your jurisdiction, processing may rely on:
- Consent (e.g., server rules notifying members about the Bot).
- Legitimate interest in operating and moderating the server.
- Contractual necessity where the Bot is provided as part of a service agreement.
Server owners should assess which basis applies and document it.

### 5. Data Retention
- Conversation logs, memory entries, and AI outputs remain in the PostgreSQL database until removed.
- Administrators can use commands (e.g., `!forget`), database maintenance scripts, or server removal to delete stored data.
- Backups (if configured) may persist for up to 30 days before automatic rotation.

### 6. Data Sharing and Transfers
- **Discord**: All interactions happen within Discord’s infrastructure and remain subject to Discord’s Privacy Policy.
- **Google Cloud**: Message content and images may be sent to Google Gemini and Vertex AI for processing.
- **Serper** (optional): Search queries are transmitted to Serper’s API.
- **Hosting Providers**: Railway or any infrastructure you use will process data as part of hosting the Bot and database.
- We do not sell user data. Access is limited to those administering the Bot or maintaining the infrastructure.

### 7. Security
- Sensitive credentials (Discord tokens, API keys, service account JSON) should be managed using environment variables and are excluded via `.gitignore`.
- PostgreSQL access should be restricted using strong passwords, TLS, and network rules as supported by your host.
- Regularly rotate tokens, API keys, and database passwords, and monitor logs for abuse.

### 8. User Choices and Rights
- Users may request deletion of their personal data via the `!forget` command or by contacting the server administrators.
- Administrators can remove or reset the database tables (`init_db.py`) or delete the Bot from a server to stop further collection.
- Depending on local law, users may have rights to access, correct, delete, or port their data. Server owners must respond to such requests directly.

### 9. Children
The Bot is intended for Discord communities that comply with Discord’s age restrictions (13+ or higher where required). Do not deploy ServerMate in servers primarily targeted at children.

### 10. International Data Transfers
If you or your infrastructure providers are located outside the user’s jurisdiction, data may be transferred internationally. Use reputable hosting providers and, where required, implement appropriate safeguards (e.g., SCCs, DPAs).

### 11. Changes to This Policy
We may update this Privacy Policy as the Bot evolves. Updates will be committed to the repository with a new “Last updated” date. Continued use after changes means you accept the revised Policy.

### 12. Contact
For privacy inquiries or data requests, contact the server owner or open an issue in the repository. Include sufficient information to verify your identity and the relevant Discord server.

