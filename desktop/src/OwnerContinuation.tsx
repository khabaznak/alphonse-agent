import { useCallback, useEffect, useRef, useState } from "react";
import type { OwnerRequest } from "./ownerConversation";

export type ContinuationQuestion = { message_id: string; checkpoint_version: number; question: string };

export async function listContinuations(request: OwnerRequest): Promise<ContinuationQuestion[]> {
  const result = await request<{ items: ContinuationQuestion[] }>("experimental_owner_continuation_list", {});
  if (!result || !Array.isArray(result.items) || result.items.some(item =>
    !item || !/^[0-9a-f]{32}$/.test(item.message_id) || !Number.isSafeInteger(item.checkpoint_version)
      || item.checkpoint_version < 1 || typeof item.question !== "string" || !item.question.trim())) {
    throw new Error("Continuation requests could not be checked.");
  }
  return result.items;
}

export async function chooseContinuation(request: OwnerRequest, question: ContinuationQuestion, approve: boolean): Promise<void> {
  if (!/^[0-9a-f]{32}$/.test(question.message_id) || !Number.isSafeInteger(question.checkpoint_version)
      || question.checkpoint_version < 1 || typeof approve !== "boolean") throw new Error("This choice could not be recorded.");
  const result = await request<{ status: string }>("experimental_owner_continuation_choose", {
    message_id: question.message_id, checkpoint_version: question.checkpoint_version, approve,
  });
  if (result?.status !== (approve ? "accepted" : "declined")) throw new Error("This choice could not be recorded.");
}

export function ContinuationChoices({ items, busy, onChoose }: {
  items: ContinuationQuestion[]; busy: string;
  onChoose: (question: ContinuationQuestion, approve: boolean) => void;
}) {
  if (!items.length) return null;
  return <section aria-label="Work waiting for your decision" className="owner-continuations">
    {items.map(item => <article key={`${item.message_id}:${item.checkpoint_version}`}>
      <p>{item.question}</p>
      <button type="button" disabled={Boolean(busy)} onClick={() => onChoose(item, true)}>Continue</button>{" "}
      <button type="button" disabled={Boolean(busy)} onClick={() => onChoose(item, false)}>Don’t continue</button>
    </article>)}
  </section>;
}

/** Only advertised servers expose these controls. The server owns all task and grant state. */
export function OwnerContinuation({ request, active }: { request: OwnerRequest; active: boolean }) {
  const [items, setItems] = useState<ContinuationQuestion[]>([]);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const generation = useRef(0);
  const refresh = useCallback(async () => {
    const current = ++generation.current;
    try { const found = await listContinuations(request); if (current === generation.current) { setItems(found); setNotice(""); } }
    catch { if (current === generation.current) setNotice("Continuation requests could not be checked."); }
  }, [request]);
  useEffect(() => {
    if (!active) { ++generation.current; setItems([]); setNotice(""); return; }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => { ++generation.current; window.clearInterval(timer); };
  }, [active, refresh]);
  const choose = async (item: ContinuationQuestion, approve: boolean) => {
    if (busy) return;
    setBusy(item.message_id);
    ++generation.current;
    try { await chooseContinuation(request, item, approve); setItems(items => items.filter(i => i.message_id !== item.message_id)); await refresh(); }
    catch { setNotice("This choice could not be recorded. Check the request and try again."); }
    finally { setBusy(""); }
  };
  return <><ContinuationChoices items={items} busy={busy} onChoose={(item, approve) => void choose(item, approve)} />
    {notice && <p role="status">{notice}</p>}</>;
}
