import type { OwnerQuestion } from "./ownerQuestions";

type Props = {
  questions: OwnerQuestion[];
  selectedId: string;
  disabled: boolean;
  onSelect: (messageId: string) => void;
};

/** Selection is always explicit; a normal message never inherits a target. */
export function OwnerQuestionSelector({ questions, selectedId, disabled, onSelect }: Props) {
  if (!questions.length && !selectedId) return null;
  const selected = questions.find((question) => question.message_id === selectedId);
  return <section className="composer-hint" aria-label="Reply to a question">
    {selectedId && <p role="status">Replying to: {selected?.question || "Selected question"} <button type="button" disabled={disabled} onClick={() => onSelect("")}>Cancel reply</button></p>}
    <details><summary>Reply to a previous question</summary>
      <p>Choose a question explicitly. Your reply starts a new request with that question as context.</p>
      {questions.map((question) => <p key={question.message_id}>
        <button type="button" disabled={disabled} aria-pressed={selectedId === question.message_id}
          onClick={() => onSelect(question.message_id)}>Reply to: {question.question}</button>
      </p>)}
    </details>
  </section>;
}
