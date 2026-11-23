/** Opt-in owner-conversation transport; the daemon remains the authority. */
import type { CommittedEffect } from "./types";
export type OwnerRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>;

type Pending = { messageId: string; pickupStarted: boolean; paused?: boolean; acknowledgementToken?: string; acknowledgementInFlight?: boolean; acknowledgementRetryAttempted?: boolean };
type Status = "pending" | "continuation_paused" | "continuation_accepted" | "continuation_declined" | "ready" | "started_unconfirmed" | "client_acknowledged" | "failed" | "unavailable";
export type OwnerCapability = { owner_only?: boolean; processing_ready?: boolean; continuation_available?: boolean; blocking_reason?: string | null; paused_legacy_requests?: number };
type StatusReply = OwnerCapability & { status?: Status; failure_code?: string };
type Pickup = { message_id: string; message: string; acknowledgement_token: string; delivery_status: string; kind?: "answer" | "question"; committed_effect?: CommittedEffect };
type OwnerIngressReply = { message_id: string; accepted_at?: string };

/** Prefer the daemon's durable ingress instant; older daemons may omit it. */
export function ownerAcceptedAt(value: unknown, fallback: string): string {
  if (typeof value === "string" && /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) && Number.isFinite(Date.parse(value))) return value;
  return fallback;
}

/** Old notifications remain untouched until their intended recipient and delivery are reconciled. */
export function legacyNotificationReviewMessage(value: unknown): string {
  if (value === null) return "Earlier notifications could not be checked for review.";
  if (!value || typeof value !== "object") return "";
  const summary = value as Record<string, unknown>;
  const counts = [summary.owner_scoped, summary.ambiguous, summary.unassigned];
  if (!counts.every((n) => typeof n === "number" && Number.isSafeInteger(n) && n >= 0)) return "";
  const total = (counts as number[]).reduce((a, b) => a + b, 0);
  if (!total) return "";
  return `${total} earlier ${total === 1 ? "notification needs" : "notifications need"} review. They have not been resent automatically.`;
}

/** The exclusive owner worker cannot process the legacy queue. */
export function ownerUnavailableMessage(reason?: string | null): string {
  if (reason === "legacy_claim_conflict") return "Earlier queued work or its saved state needs review before the simpler loop can start; your message was not sent.";
  if (reason === "queue_unavailable") return "The conversation queue is unavailable. Your message was not sent.";
  return "Owner conversation is not ready; your message was not sent.";
}

/** Fixed, non-sensitive categories supplied by the daemon; never render raw errors. */
export function ownerFailureMessage(code?: string): string {
  const messages: Record<string, string> = {
    cancel_requested: "Cancellation was requested. Check unfinished requests before trying again.",
    time_limit: "This request exceeded its time limit. Check unfinished requests before trying again.",
    model_unavailable: "The model could not complete this request. Check unfinished requests before trying again.",
    usage_unconfirmed: "Model usage could not be verified. Check unfinished requests before trying again.",
    usage_limit_exceeded: "The model exceeded the request's usage limit. Check unfinished requests before trying again.",
    source_unavailable: "A needed source was unavailable or access was denied. Check access and unfinished requests before trying again.",
    source_changed: "A source changed or access was revoked during the request. Check unfinished requests before trying again.",
    source_incomplete: "A source could not be verified as complete. Check unfinished requests before trying again.",
    unsupported_operation: "This request needs an operation that is not available in owner conversation mode.",
    action_limit: "I reached the action limit before finishing. Check unfinished requests before trying again.",
    answer_unverified: "I could not verify enough source information to answer. Check unfinished requests before trying again.",
    clarification_unverified: "The follow-up question could not be checked, so it was not sent. Check unfinished requests before trying again.",
    clarification_review_unavailable: "The clarification check is unavailable, so the follow-up question was not sent. Check unfinished requests before trying again.",
  };
  return (code && Object.prototype.hasOwnProperty.call(messages, code))
    ? messages[code] : "This conversation needs attention; its result may be unavailable or unconfirmed.";
}

/**
 * In-memory tracking is intentional: a lost pickup or submission response is
 * uncertain and must not be repeated automatically.
 */
/** Ignore a pending snapshot from a previous member's conversation instance. */
export function ownerBusyFor(snapshot: { conversation: OwnerConversation | null; busy: boolean }, current: OwnerConversation | null): boolean {
  return current !== null && snapshot.conversation === current && snapshot.busy;
}

export class OwnerConversation {
  private pending = new Map<string, Pending>();
  private submitting = false;
  private submissionUncertain = false;
  constructor(private readonly request: OwnerRequest) {}

  hasPending(): boolean { return this.submitting || this.submissionUncertain || this.pending.size > 0; }
  /** Only claim active work while submission or an uncollected request is pending. */
  isWorking(): boolean { return this.submitting || [...this.pending.values()].some((item) => !item.pickupStarted && !item.paused); }
  hasUncertainSubmission(): boolean { return this.submissionUncertain; }
  activeMessageId(): string | undefined {
    if (this.blockingState() !== "working" || this.submitting) return undefined;
    return [...this.pending.values()].find((item) => !item.pickupStarted)?.messageId;
  }

  async queueFollowup(prompt: string, activeMessageId: string, requestId: string): Promise<"queued" | "late"> {
    const result = await this.request<{ status?: string }>("experimental_owner_followup_add", {
      active_message_id: activeMessageId, prompt, request_id: requestId,
    });
    if (result?.status !== "queued" && result?.status !== "late") throw new Error("Follow-up submission could not be confirmed. Review the follow-up list before sending again.");
    return result.status;
  }

  /** Adopt only a server-confirmed ready released item; never repick an uncertain delivery. */
  async adoptReleased(messageId: string): Promise<boolean> {
    if (this.hasPending()) return false;
    const status = await this.request<StatusReply>("experimental_owner_status", { message_id: messageId });
    if (status.status !== "ready" || this.hasPending()) return false;
    this.adoptReady(messageId);
    return true;
  }


  /** Distinguish active work from a one-shot delivery requiring recovery. */
  blockingState(): "idle" | "working" | "acknowledgement" | "uncertain" {
    if (this.submissionUncertain) return "uncertain";
    if (this.submitting || [...this.pending.values()].some((item) => !item.pickupStarted)) return "working";
    if ([...this.pending.values()].some((item) => !item.acknowledgementToken)) return "uncertain";
    return this.pending.size ? "acknowledgement" : "idle";
  }

  /** Remove a locally tracked request only after the server confirms a stop. */
  forgetAbandoned(messageId: string): void { this.pending.delete(messageId); }

  /** Explicitly resume a listed ready answer after the Desktop app reloads. */
  adoptReady(messageId: string): void {
    if (!/^[0-9a-f]{32}$/.test(messageId) || this.submitting
        || (this.pending.size > 0 && !this.pending.has(messageId))) {
      throw new Error("Resolve the current request before opening another answer.");
    }
    const current = this.pending.get(messageId);
    if (current?.pickupStarted) throw new Error("This answer is already unconfirmed and cannot be fetched again automatically.");
    if (!current) this.pending.set(messageId, { messageId, pickupStarted: false });
  }

  async send(prompt: string, replyToMessageId?: string): Promise<OwnerIngressReply> {
    const blocked = this.blockingState();
    if (blocked === "working") throw new Error("Wait for the current conversation to finish.");
    if (blocked === "acknowledgement") throw new Error("The answer is shown, but saving it to conversation history is not confirmed. Check earlier work and recovery before sending another message.");
    if (blocked === "uncertain") throw new Error("The previous message or answer has an uncertain result. Check earlier work and recovery before sending another message.");
    if (replyToMessageId !== undefined && !/^[0-9a-f]{32}$/.test(replyToMessageId)) {
      throw new Error("This question could not be selected for a reply.");
    }
    this.submitting = true;
    let queued: OwnerIngressReply;
    try {
      const capability = await this.request<StatusReply>("experimental_owner_status", {});
      if (capability.processing_ready !== true) throw new Error(ownerUnavailableMessage(capability.blocking_reason));
      try {
        queued = await this.request("experimental_owner_ingress", {
          prompt, ...(replyToMessageId === undefined ? {} : { reply_to_message_id: replyToMessageId }),
        });
      } catch {
        this.submissionUncertain = true;
        throw new Error("Message submission could not be confirmed. Check the conversation before sending it again.");
      }
      if (!queued || typeof queued.message_id !== "string" || !/^[0-9a-f]{32}$/.test(queued.message_id)) {
        this.submissionUncertain = true;
        throw new Error("Message submission could not be confirmed. Check the conversation before sending it again.");
      }
      this.pending.set(queued.message_id, { messageId: queued.message_id, pickupStarted: false });
      return queued;
    } finally {
      this.submitting = false;
    }
  }

  async poll(onAnswer: (messageId: string, answer: string, kind: "answer" | "question", effect?: CommittedEffect) => void): Promise<boolean> {
    let acknowledgementRecovered = false;
    for (const pending of this.pending.values()) {
      if (pending.pickupStarted) {
        // Only the exact-token acknowledgement is retryable. Never ask for
        // the draft again, even after an unknown socket outcome.
        if (!pending.acknowledgementToken || pending.acknowledgementInFlight) continue;
        const state = await this.request<StatusReply>("experimental_owner_status", { message_id: pending.messageId });
        if (!this.pending.has(pending.messageId) || pending.acknowledgementInFlight) continue;
        if (state.status === "client_acknowledged") {
          this.pending.delete(pending.messageId);
          acknowledgementRecovered = true;
          continue;
        }
        // A daemon restart loses its in-memory status cache even when the
        // original acknowledgment committed. The server's exact-token
        // acknowledgment endpoint verifies the durable owner, checkpoint,
        // history and token before returning success, so the same bounded
        // retry is safe when status is unavailable too. This never repeats
        // the one-shot pickup or proves that a person read the answer.
        if (state.status !== "started_unconfirmed" && state.status !== "unavailable") {
          throw new Error("The answer is shown, but saving it to conversation history could not be confirmed.");
        }
        if (pending.acknowledgementRetryAttempted) {
          throw new Error("The answer is shown, but saving it to conversation history could not be confirmed.");
        }
        // A bounded same-token retry cannot start another pickup. Repeated
        // failures need explicit recovery rather than an endless retry loop.
        pending.acknowledgementRetryAttempted = true;
        await this.acknowledge(pending);
        acknowledgementRecovered = true;
        continue;
      }
      const state = await this.request<StatusReply>("experimental_owner_status", { message_id: pending.messageId });
      // Poll cycles can overlap; another cycle may have started the pickup
      // while this status call was in flight.
      if (!this.pending.has(pending.messageId) || pending.pickupStarted) continue;
      if (state.status === "pending" || state.status === "continuation_accepted" || state.status === "continuation_paused") {
        pending.paused = state.status === "continuation_paused";
        continue;
      }
      if (state.status === "continuation_declined") { this.pending.delete(pending.messageId); continue; }
      if (state.status !== "ready") {
        if (state.status === "failed") this.pending.delete(pending.messageId);
        pending.pickupStarted = true;
        throw new Error(state.status === "failed" ? ownerFailureMessage(state.failure_code) : state.status === "started_unconfirmed"
          ? "The answer could not be confirmed. It will not be fetched again automatically."
          : "This conversation needs attention; its result may be unavailable or unconfirmed.");
      }
      // Mark locally before making the one-shot call, including if the network
      // fails after the server has durably started the response.
      pending.pickupStarted = true;
      let pickup: Pickup;
      try {
        pickup = await this.request("experimental_owner_take_draft", { message_id: pending.messageId });
      } catch {
        throw new Error("The answer could not be confirmed. It will not be fetched again automatically.");
      }
      if (!pickup || pickup.message_id !== pending.messageId || typeof pickup.message !== "string"
          || typeof pickup.acknowledgement_token !== "string" || pickup.delivery_status !== "started_unconfirmed"
          || (pickup.kind !== undefined && pickup.kind !== "answer" && pickup.kind !== "question")
          || (pickup.committed_effect !== undefined && (!pickup.committed_effect
            || pickup.committed_effect.kind !== "scheduled_task_cancelled"
            || typeof pickup.committed_effect.name !== "string"
            || typeof pickup.committed_effect.schedule_id !== "string"
            || typeof pickup.committed_effect.execution_may_have_started !== "boolean"))) {
        throw new Error("The answer could not be confirmed. It will not be fetched again automatically.");
      }
      if (pickup.committed_effect) onAnswer(pending.messageId, pickup.message, pickup.kind ?? "answer", pickup.committed_effect);
      else onAnswer(pending.messageId, pickup.message, pickup.kind ?? "answer");
      pending.acknowledgementToken = pickup.acknowledgement_token;
      await this.acknowledge(pending);
    }
    return acknowledgementRecovered;
  }

  /** Retry only a previously received exact token; never repeat pickup or execution. */
  async retryAcknowledgement(): Promise<boolean> {
    const pending = [...this.pending.values()].find((item) => item.pickupStarted && item.acknowledgementToken);
    if (!pending || pending.acknowledgementInFlight) return false;
    await this.acknowledge(pending);
    return !this.pending.has(pending.messageId);
  }

  private async acknowledge(pending: Pending): Promise<void> {
    if (!pending.acknowledgementToken || pending.acknowledgementInFlight) return;
    pending.acknowledgementInFlight = true;
    try {
      const result = await this.request<{ status?: string }>("experimental_owner_ack_draft", {
        message_id: pending.messageId, acknowledgement_token: pending.acknowledgementToken,
      });
      if (result?.status !== "client_acknowledged") throw new Error("acknowledgment unavailable");
      this.pending.delete(pending.messageId);
    } catch {
      throw new Error("The answer is shown, but saving it to conversation history could not be confirmed.");
    } finally {
      pending.acknowledgementInFlight = false;
    }
  }
}
