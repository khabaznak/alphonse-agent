export const DESKTOP_COMMANDS = [
  "/project",
  "/project-context",
  "/integrations",
  "/model-provider",
  "/model",
  "/agent-config",
  "/stop",
  "/exit",
  "/quit",
] as const;

export function matchingCommands(value: string): string[] {
  return value.startsWith("/") ? DESKTOP_COMMANDS.filter((command) => command.startsWith(value)) : [];
}
