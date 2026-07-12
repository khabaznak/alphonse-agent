export type ActivityEvent = {
  sequence: number;
  phase: string;
  label: string;
  message: string;
  speaker: string;
  task_id: string;
  message_id: string;
  user: string;
  integration_id: string;
  channel_target: string;
};

export type Delivery = {
  outbox_message_id: string;
  message: string;
  kind: string;
  metadata: Record<string, unknown>;
};

export type Question = {
  question_id: string;
  message: string;
  kind: "open_text" | "yes_no" | "single_choice";
  choices: Array<{ id: string; label: string }>;
};

export type Project = {
  project_id: string;
  name: string;
  description: string;
  root_path: string;
  visibility: "private" | "shared";
  owner_user_id: string;
  status: "active" | "archived";
  archived_at: string | null;
  created_at: string;
  updated_at: string;
};

export type AgentDocument = { file_name: string; display_name: string; content?: string };
export type InferenceSettings = { provider_key: string; model_id: string; validation_error?: string };
export type WebToolsSettings = { enabled: boolean; searxng_base_url: string; search_timeout_seconds: number; fetch_timeout_seconds: number; fetch_max_chars: number; configured: boolean; available: boolean };

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at?: string;
};
