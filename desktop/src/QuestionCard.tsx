import { useRef, useState } from "react";
import { daemonRequest } from "./api";
import type { Question } from "./types";

type AnswerResult = { handled?: boolean; invalid?: boolean; ambiguous?: boolean; question_id?: string; task_id?: string; message?: string; answer?: Record<string, unknown> };

export async function submitQuestionAnswer(question: Question, user: string, payload: Record<string, unknown>): Promise<void> {
  const result = await daemonRequest<AnswerResult>("answer_question", { user, question_id: question.question_id, payload });
  if (!result.handled || result.invalid || result.ambiguous || result.question_id !== question.question_id || result.task_id !== question.task_id || !Object.keys(result.answer || {}).length) {
    throw new Error(result.message || "This question is no longer available. Refresh to check its status.");
  }
}

export function QuestionCard({ question, user, timezone, onDone }: { question: Question; user: string; timezone: string; onDone: () => Promise<void> }) {
  const [value, setValue] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const busy = useRef(false);
  const expired = Boolean(question.expires_at && Date.parse(question.expires_at) <= Date.now());
  const disabled = Boolean(status) || expired;
  const act = async (payload?: Record<string, unknown>) => {
    if (busy.current || disabled) return;
    busy.current = true;
    setStatus("Sending…"); setError("");
    try {
      if (payload) await submitQuestionAnswer(question, user, payload);
      else {
        const result = await daemonRequest<{ cancelled: boolean }>("cancel_question", { user, question_id: question.question_id });
        if (!result.cancelled) throw new Error("This question is no longer available. Refresh to check its status.");
      }
      setStatus(payload ? "Answer received. Resuming work…" : "Question cancelled.");
      await onDone();
    } catch (cause) {
      setStatus(""); setError(cause instanceof Error ? cause.message : "Could not send your answer. Please retry.");
    } finally { busy.current = false; }
  };
  const submit = () => {
    if (question.kind === "multi_choice") return act({ choice_ids: selected });
    if (question.kind === "datetime") return act({ datetime: value });
    return act({ text: value.trim() });
  };
  return <section className="question-card" data-question-id={question.question_id} aria-label="Question from Alphonse" tabIndex={-1}>
    <strong>Alphonse needs your input</strong><p>{question.message}</p>
    {question.context?.text && <p className="question-context">{question.context.text}</p>}
    {question.context?.artifacts?.map((artifact, index) => <article className="question-resource" key={artifact.artifact_id || index}>
      {artifact.name ? <><strong>{artifact.name}</strong><p>{artifact.description}</p><small>{artifact.status === "ready" ? "Related skill resource · read-only context" : "Resource currently unavailable"}</small></> : <p>Related resource unavailable or no longer authorized.</p>}
    </article>)}
    <form onSubmit={(event) => { event.preventDefault(); void submit(); }}>
      <fieldset disabled={disabled} className="question-fields">
        <legend className="question-answer-legend">Your answer</legend>
        {question.kind === "yes_no" && <div className="question-actions"><button type="button" onClick={() => void act({ answer: true })}>Yes</button><button type="button" onClick={() => void act({ answer: false })}>No</button></div>}
        {question.kind === "single_choice" && <div className="question-actions">{question.choices.map((choice) => <button type="button" key={choice.id} onClick={() => void act({ choice_id: choice.id })}>{choice.label}</button>)}</div>}
        {question.kind === "multi_choice" && <fieldset className="a2ui-choice-picker"><legend>Choose one or more options</legend>{question.choices.map((choice) => <label key={choice.id}><input type="checkbox" checked={selected.includes(choice.id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, choice.id] : current.filter((id) => id !== choice.id))} />{choice.label}</label>)}</fieldset>}
        {question.kind === "datetime" && <label>Date and time ({timezone})<input required type="datetime-local" value={value} onChange={(event) => setValue(event.target.value)} /></label>}
        {question.kind === "open_text" && <label>Your answer<textarea required value={value} onChange={(event) => setValue(event.target.value)} rows={2} /></label>}
        <div className="question-actions">
          {!["yes_no", "single_choice"].includes(question.kind) && <button type="submit" disabled={question.kind === "multi_choice" ? !selected.length : !value.trim()}>Answer</button>}
          <button type="button" className="question-cancel" onClick={() => void act()}>Cancel question</button>
        </div>
      </fieldset>
    </form>
    {error && <p className="error" role="alert">{error}</p>}
    {(status || expired) && <p role="status">{expired ? "This question has expired." : status}</p>}
    {(error || expired) && <button type="button" onClick={() => void onDone()}>Refresh questions</button>}
  </section>;
}
