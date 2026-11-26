export function quotedRequestText(original: string): string {
  return original.split(/\r?\n/).map((line) => `> ${line}`).join("\n");
}

/** Prepare an editable question; this does not submit or retry the request. */
export function draftQuestionAboutRequest(existing: string, original: string): string {
  const quote = quotedRequestText(original);
  const question = `I have a question about this earlier request text:\n\n${quote}`;
  if (existing.includes(question)) return existing;
  return existing.length ? `${existing}\n\n${question}` : question;
}
