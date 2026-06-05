import process from "node:process";

function writeError(code, message) {
  process.stderr.write(JSON.stringify({ error: { code, message } }) + "\n");
}

function extractText(response) {
  if (!response) return "";
  if (typeof response === "string") return response;
  if (typeof response.content === "string") return response.content;
  if (typeof response.text === "string") return response.text;
  if (typeof response.outputText === "string") return response.outputText;
  if (Array.isArray(response.choices) && response.choices.length > 0) {
    const first = response.choices[0];
    if (first?.message?.content) return String(first.message.content);
    if (first?.text) return String(first.text);
  }
  return "";
}

function renderToolPrompt(payload) {
  return [
    "You are selecting exactly one tool call for Alphonse.",
    "Return one JSON object and no prose.",
    'Schema: {"content":"optional text","planner_intent":"short reason","tool_call":{"kind":"call_tool","tool_name":"name","args":{}}}.',
    "",
    JSON.stringify(
      {
        tool_choice: payload.toolChoice || "auto",
        tools: Array.isArray(payload.tools) ? payload.tools : [],
        messages: Array.isArray(payload.messages) ? payload.messages : [],
      },
      null,
      2,
    ),
  ].join("\n");
}

try {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const payload = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  if (!payload.githubToken) {
    writeError("github_copilot_token_missing", "COPILOT_GITHUB_TOKEN is required.");
    process.exit(1);
  }

  let sdk;
  try {
    sdk = await import("@github/copilot-sdk");
  } catch (error) {
    writeError("github_copilot_sdk_missing", "Install @github/copilot-sdk for github_copilot provider.");
    process.exit(1);
  }

  const CopilotClient = sdk.CopilotClient || sdk.default?.CopilotClient || sdk.default;
  if (typeof CopilotClient !== "function") {
    writeError("github_copilot_sdk_invalid", "CopilotClient export was not found.");
    process.exit(1);
  }

  const client = new CopilotClient({
    githubToken: payload.githubToken,
    clientId: payload.clientId || undefined,
    useLoggedInUser: false,
  });
  const session = await client.createSession({ model: payload.model || undefined });
  const prompt =
    payload.mode === "complete_with_tools"
      ? renderToolPrompt(payload)
      : `${payload.systemPrompt || ""}\n\n${payload.userPrompt || ""}`.trim();
  const response = await session.sendAndWait({ prompt });
  const content = extractText(response).trim();
  if (payload.mode === "complete_with_tools") {
    process.stdout.write(content);
  } else {
    process.stdout.write(JSON.stringify({ content }));
  }
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  const lowered = message.toLowerCase();
  const code =
    lowered.includes("unauthorized") || lowered.includes("forbidden") || lowered.includes("auth")
      ? "github_copilot_auth_failed"
      : "github_copilot_bridge_failed";
  writeError(code, message.slice(0, 300));
  process.exit(1);
}
