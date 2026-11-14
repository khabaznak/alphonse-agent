import type { OwnerRequest } from "./ownerConversation";

/** The daemon returns only confirmed questions visible to the current owner. */
export type OwnerQuestion = { message_id: string; question: string };

export async function listOwnerQuestions(request: OwnerRequest): Promise<OwnerQuestion[]> {
  const result = await request<{ questions: OwnerQuestion[] }>("experimental_owner_questions", {});
  if (!result || !Array.isArray(result.questions) || result.questions.some((item) =>
    !item || !/^[0-9a-f]{32}$/.test(item.message_id) || typeof item.question !== "string" || !item.question.trim())) {
    throw new Error("Your confirmed questions could not be checked.");
  }
  return result.questions;
}
