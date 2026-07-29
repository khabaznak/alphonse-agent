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
  project_id: string;
  created_at?: string;
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
export type MemorySettings = { max_ledger_bytes: number; compaction_summary_max_words: number };
export type VerificationState = { ready: boolean; verified_at: string; error: string; preview: string };
export type MediaToolsSettings = {
  platform: string; say_available: boolean;
  tts: { enabled: boolean; model_id: string; device_map: string; dtype: string; language: string; speaker: string; instruct: string; attn_implementation: string; local_files_only: boolean; available: boolean; verification: VerificationState };
  stt: { enabled: boolean; executable_path: string; model: string; default_language: string; available: boolean; verification: VerificationState };
  ocr: { enabled: boolean; ollama_base_url: string; model_id: string; timeout_seconds: number; available: boolean; verification: VerificationState };
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at?: string;
  source?: string;
  project_id?: string;
  sequence?: number;
};
