# Notion Setup

The team keeps shared notes, server info, and credentials on the **Knighthacks** Notion page:

https://app.notion.com/p/Knighthacks-3f596ef5fec5807ebaa2dacc55626600

Credentials live in Notion only. Never commit them to this repo.

## 1. Get access to the page

1. Make sure you have a Notion account.
2. Ask the page owner (Dom) to share the Knighthacks page with your Notion email (Share → invite → "Can edit").
3. Open the link above and confirm you can see the page.

## 2. Connect Notion to Claude Code (optional)

This lets Claude read and update the page for you.

### Option A: claude.ai connector (if you log into Claude Code with a claude.ai account)

1. Go to https://claude.ai → Settings → Connectors → add **Notion** and authorize the workspace that contains the Knighthacks page.
2. In Claude Code, run `/mcp` and confirm **claude.ai Notion** shows as connected. If it does not, select it and reconnect.

### Option B: add the Notion MCP server directly

```bash
claude mcp add --transport http notion https://mcp.notion.com/mcp
```

Then run `/mcp` in Claude Code, select `notion`, and finish the OAuth login in the browser.

### Check it works

Ask Claude:

> fetch the Notion page https://app.notion.com/p/Knighthacks-3f596ef5fec5807ebaa2dacc55626600

## 3. Allow Claude to edit the page (optional)

In auto mode, Claude Code may block writes to Notion (especially anything containing credentials). To allow edits, run `/permissions` → **Allow** → **Add a new rule**, and enter the tool name for your connection:

- Option A (claude.ai connector): `mcp__claude_ai_Notion__notion-update-page`
- Option B (direct server): `mcp__notion__notion-update-page`

Or add it to `.claude/settings.local.json` in this repo (git-ignored):

```json
{
  "permissions": {
    "allow": [
      "mcp__claude_ai_Notion__notion-update-page"
    ]
  }
}
```

This allows **all** Notion page edits without a prompt, not just this page. Skip it if you would rather approve each edit.

## What is on the page

- **AMD Servers**: login URL and usernames for the AMD GPU servers (`dub.aupcloud.io`). Passwords are on the Notion page only.
