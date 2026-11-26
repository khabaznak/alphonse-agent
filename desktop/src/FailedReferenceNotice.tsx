import type { RefObject } from "react";
import { formatMessageTime } from "./messageTime";

export function FailedReferenceNotice({ id, createdAt, timezone, unavailable, removeRef, onRemove }: {
  id: string; createdAt?: string; timezone: string; unavailable: boolean;
  removeRef: RefObject<HTMLButtonElement | null>; onRemove: () => void;
}) {
  const time = createdAt ? formatMessageTime(createdAt, timezone) : null;
  return <div className="composer-hint" role="status">Referencing an earlier request that did not complete{time && <> · <time dateTime={createdAt} title={time.tooltip}>{time.visible}</time></>}.
    {unavailable && <p role="alert">This reference is no longer available. Your draft was not sent. Remove the reference or select an earlier request again.</p>}
    <button ref={removeRef} type="button" onClick={onRemove}>Remove reference</button>
    <details><summary>Reference details</summary><span>Request ID: {id}</span></details>
  </div>;
}
