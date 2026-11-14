/** Opt-in owner-conversation transport; the daemon remains the authority. */
export type OwnerRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>;

type Pending = { messageId: string; pickupStarted: boolean; acknowledgementToken?: string; acknowledgementInFlight?: boolean; acknowledgementRetryAttempted?: boolean };
type Status = "pending" | "ready" | "started_unconfirmed" | "client_acknowledged" | "failed" | "unavailable";
type StatusReply = { processing_ready?: boolean; status?: Status };
type Pickup = { message_id: string; message: string; acknowledgement_token: string; delivery_status: string };

/** The exclusive owner worker cannot process the legacy queue. */
export function assertLegacyConversationAvailable(ownerWorkerReady: boolean): void {
  if (ownerWorkerReady) throw new Error("Turn on the simpler conversation loop to send a message while this owner-only service is running.");
}

/**
 * In-memory tracking is intentional: a lost pickup or submission response is
 * uncertain and must not be repeated automatically. The default UI never
 * constructs this transport unless the explicit local opt-in is set.
 */
export class OwnerConversation {
  private pending = new Map<string, Pending>();
  private submitting = false;
  private submissionUncertain = false;
  constructor(private readonly request: OwnerRequest) {}

  hasPending(): boolean { return this.submitting || this.submissionUncertain || this.pending.size > 0; }
  hasUncertainSubmission(): boolean { return this.submissionUncertain; }

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

  async send(prompt: string): Promise<{ message_id: string }> {
    if (this.hasPending()) throw new Error("Wait for the current conversation to finish or resolve its uncertain result.");
    this.submitting = true;
    let queued: { message_id: string };
    try {
      const capability = await this.request<StatusReply>("experimental_owner_status", {});
      if (capability.processing_ready !== true) throw new Error("Owner conversation is not ready; your message was not sent.");
      try {
        queued = await this.request("experimental_owner_ingress", { prompt });
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

  async poll(onAnswer: (messageId: string, answer: string) => void): Promise<boolean> {
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
        if (state.status !== "started_unconfirmed") {
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
      if (state.status === "pending") continue;
      if (state.status !== "ready") {
        if (state.status === "failed") this.pending.delete(pending.messageId);
        pending.pickupStarted = true;
        throw new Error(state.status === "started_unconfirmed"
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
          || typeof pickup.acknowledgement_token !== "string" || pickup.delivery_status !== "started_unconfirmed") {
        throw new Error("The answer could not be confirmed. It will not be fetched again automatically.");
      }
      onAnswer(pending.messageId, pickup.message);
      pending.acknowledgementToken = pickup.acknowledgement_token;
      await this.acknowledge(pending);
    }
    return acknowledgementRecovered;
  }

  private async acknowledge(pending: Pending): Promise<void> {
    if (!pending.acknowledgementToken || pending.acknowledgementInFlight) return;
    pending.acknowledgementInFlight = true;
    try {
      await this.request("experimental_owner_ack_draft", {
        message_id: pending.messageId, acknowledgement_token: pending.acknowledgementToken,
      });
      this.pending.delete(pending.messageId);
    } catch {
      throw new Error("The answer is shown, but saving it to conversation history could not be confirmed.");
    } finally {
      pending.acknowledgementInFlight = false;
    }
  }
}
