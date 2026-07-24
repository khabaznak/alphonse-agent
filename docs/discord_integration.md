# Discord Integration

Discord is an optional first-class v2 integration. It receives direct messages
and guild messages from known Alphonse users, queues them through the same
durable pipeline as Telegram, and delivers replies, attachments, typing, and
reactions through the Discord bot.

## Discord setup

1. Create an application and bot in the Discord Developer Portal.
2. Enable the **Message Content Intent** under the bot's privileged Gateway
   intents. Alphonse needs this to read message text.
3. Install the bot in the desired server with permission to view channels, send
   messages, attach files, add reactions, and send messages in threads.
4. In Alphonse's TUI, open `/integrations`, select Discord, and enter the bot
   token, your Discord user ID, then enable the integration.

Guild messages are observed in every channel that the bot can read, subject to
the optional guild/channel allow-lists. Alphonse only replies when mentioned;
after an invocation in a Discord thread it continues that thread conversation
without additional mentions. Direct messages from unknown users create an
administrator approval request; unknown guild members are not processed.
