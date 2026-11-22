import { describe, expect, it, vi } from "vitest";
import { OwnerConversation, ownerAcceptedAt, ownerBusyFor, ownerFailureMessage, ownerUnavailableMessage, type OwnerRequest } from "./ownerConversation";
import { avatarState } from "./layoutState";

const id = "a".repeat(32);
const token = "b".repeat(32);

it("distinguishes known usage overrun from unconfirmed usage", () => {
  expect(ownerFailureMessage("usage_limit_exceeded")).toContain("exceeded");
  expect(ownerFailureMessage("usage_unconfirmed")).toContain("could not be verified");
  expect(ownerFailureMessage("action_limit")).toContain("action limit");
});

it("uses the accepted server instant even when the response arrives after midnight", () => {
  const beforeMidnight = "2026-11-16T05:59:59+00:00";
  const responseTime = "2026-11-16T06:00:01Z";
  expect(ownerAcceptedAt(beforeMidnight, responseTime)).toBe(beforeMidnight);
  expect(ownerAcceptedAt(undefined, responseTime)).toBe(responseTime);
  expect(ownerAcceptedAt("2026-11-16T05:59:59", responseTime)).toBe(responseTime);
  expect(ownerAcceptedAt("invalidZ", responseTime)).toBe(responseTime);
});

describe("explicit owner conversation transport", () => {
  it("keeps a request pending through polling, scopes it to the conversation, and clears it on acknowledgement", async () => {
    let status = "pending";
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const otherMember = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await client.poll(() => {});
    const snapshot = { conversation: client, busy: client.hasPending() };
    expect(avatarState(true, "", "idle", ownerBusyFor(snapshot, client))).toBe("pending");
    expect(avatarState(false, "", "idle", ownerBusyFor(snapshot, client))).toBe("disconnected");
    expect(ownerBusyFor(snapshot, otherMember)).toBe(false);
    status = "ready";
    await client.poll(() => {});
    expect(avatarState(true, "", "idle", client.hasPending())).toBe("idle");
  });
  it("binds a reply only when a question was explicitly selected", async () => {
    const request = vi.fn(async (method: string) => method === "experimental_owner_status"
      ? { processing_ready: true } : { message_id: id });
    const client = new OwnerConversation(request as OwnerRequest);
    const questionId = "c".repeat(32);
    await client.send("The blue one", questionId);
    expect(request).toHaveBeenCalledWith("experimental_owner_ingress", {
      prompt: "The blue one", reply_to_message_id: questionId,
    });
  });

  it("rejects malformed question selection before making a request", async () => {
    const request = vi.fn();
    const client = new OwnerConversation(request as OwnerRequest);
    await expect(client.send("reply", "invalid")).rejects.toThrow("question could not be selected");
    expect(request).not.toHaveBeenCalled();
  });

  it("sends a new message without binding any previous question", async () => {
    const request = vi.fn(async (method: string) => method === "experimental_owner_status"
      ? { processing_ready: true } : { message_id: id });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("A different topic");
    expect(request).toHaveBeenCalledWith("experimental_owner_ingress", { prompt: "A different topic" });
  });
  it("does not submit unless the daemon confirms a connected owner worker", async () => {
    const request = vi.fn(async () => ({ processing_ready: false }));
    const client = new OwnerConversation(request as OwnerRequest);
    await expect(client.send("hello")).rejects.toThrow("not ready");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("explains a legacy-claim conflict before submission", async () => {
    const request = vi.fn(async () => ({ owner_only: true, processing_ready: false, blocking_reason: "legacy_claim_conflict" }));
    const client = new OwnerConversation(request as OwnerRequest);
    await expect(client.send("hello")).rejects.toThrow("queued work or its saved state");
    expect(request).toHaveBeenCalledTimes(1);
    expect(ownerUnavailableMessage("queue_unavailable")).toContain("queue is unavailable");
    expect(ownerUnavailableMessage("future_unknown_reason")).toContain("not ready");
  });

  it("does not submit a second turn while the previous turn is unresolved", async () => {
    const request = vi.fn(async (method: string) => method === "experimental_owner_status" ? { processing_ready: true } : { message_id: id });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("first");
    await expect(client.send("second")).rejects.toThrow("current conversation");
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_ingress")).toHaveLength(1);
  });

  it("does not automatically resubmit after an uncertain ingress response", async () => {
    const request = vi.fn(async (method: string) => {
      if (method === "experimental_owner_status") return { processing_ready: true };
      throw new Error("socket dropped");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await expect(client.send("hello")).rejects.toThrow("could not be confirmed");
    expect(client.blockingState()).toBe("uncertain");
    await expect(client.send("hello")).rejects.toThrow("uncertain result");
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_ingress")).toHaveLength(1);
  });

  it("submits and accepts one answer before acknowledging its exact token", async () => {
    const calls: string[] = [];
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push(method);
      if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn(() => calls.push("display"));
    await client.send("hello");
    await client.poll(seen);
    await client.poll(seen);
    expect(seen).toHaveBeenCalledExactlyOnceWith(id, "answer", "answer");
    expect(calls).toEqual(["experimental_owner_status", "experimental_owner_ingress", "experimental_owner_status", "experimental_owner_take_draft", "display", "experimental_owner_ack_draft"]);
    expect(request).toHaveBeenLastCalledWith("experimental_owner_ack_draft", { message_id: id, acknowledgement_token: token });
  });

  it("exposes a delivered clarification as a question", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "Which one?", kind: "question", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("hello");
    await client.poll(seen);
    expect(seen).toHaveBeenCalledExactlyOnceWith(id, "Which one?", "question");
    expect(client.hasPending()).toBe(false);
  });

  it("passes verified cancellation uncertainty to the conversation", async () => {
    const effect = { kind: "scheduled_task_cancelled", schedule_id: "fixture", name: "Pantry", execution_may_have_started: true };
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "Listo.", committed_effect: effect, acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("Cancel the check");
    await client.poll(seen);
    expect(seen).toHaveBeenCalledExactlyOnceWith(id, "Listo.", "answer", effect);
  });

  it("does not display or acknowledge an unknown pickup kind", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", kind: "other", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("hello");
    await expect(client.poll(seen)).rejects.toThrow("could not be confirmed");
    expect(seen).not.toHaveBeenCalled();
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_ack_draft")).toHaveLength(0);
  });

  it("never repeats a pickup whose response was lost", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      throw new Error("socket dropped");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("not be fetched again automatically");
    await client.poll(() => {});
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(1);
  });

  it("does not claim an unavailable result was never delivered", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: "unavailable" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      throw new Error("unexpected request");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("unavailable or unconfirmed");
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(0);
  });

  it("explains a known failure without retrying the request or fetching a draft", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id
        ? { status: "failed", failure_code: "source_incomplete" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      throw new Error("unexpected request");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("source could not be verified as complete");
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(0);
    expect(client.hasPending()).toBe(false);
  });

  it("explains when a follow-up question cannot be checked", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id
        ? { status: "failed", failure_code: "clarification_unverified" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      throw new Error("unexpected request");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("follow-up question could not be checked");
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(0);
  });

  it("distinguishes an unavailable clarification check", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id
        ? { status: "failed", failure_code: "clarification_review_unavailable" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      throw new Error("unexpected request");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("clarification check is unavailable");
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(0);
  });

  it("never displays an unknown backend failure string", async () => {
    const secret = "raw private source or exception";
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id
        ? { status: "failed", failure_code: secret } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      throw new Error("unexpected request");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("result may be unavailable or unconfirmed");
  });

  it("does not start the same pickup twice when poll cycles overlap", async () => {
    let finishStatus!: (value: { status: string }) => void;
    const waiting = new Promise<{ status: string }>((resolve) => { finishStatus = resolve; });
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? waiting : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("hello");
    const first = client.poll(seen);
    const second = client.poll(seen);
    finishStatus({ status: "ready" });
    await Promise.all([first, second]);
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(1);
    expect(seen).toHaveBeenCalledTimes(1);
  });

  it("does not pick up an answer after the owner stopped that request", async () => {
    let finishStatus!: (value: { status: string }) => void;
    const waiting = new Promise<{ status: string }>((resolve) => { finishStatus = resolve; });
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? waiting : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      throw new Error("unexpected pickup");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    const poll = client.poll(() => {});
    client.forgetAbandoned(id);
    finishStatus({ status: "ready" });
    await poll;
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(0);
  });

  it("opens a listed ready answer only after an explicit adoption following reload", async () => {
    const request = vi.fn(async (method: string) => {
      if (method === "experimental_owner_status") return { status: "ready" };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.poll(seen);
    expect(request).not.toHaveBeenCalled();
    client.adoptReady(id);
    await client.poll(seen);
    expect(seen).toHaveBeenCalledExactlyOnceWith(id, "answer", "answer");
    expect(client.hasPending()).toBe(false);
  });

  it("reconciles a lost ack response from durable status without repeating pickup", async () => {
    let ackAttempted = false;
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: ackAttempted ? "client_acknowledged" : "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      ackAttempted = true;
      throw new Error("socket dropped");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("hello");
    await expect(client.poll(seen)).rejects.toThrow("saving it to conversation history could not be confirmed");
    expect(await client.poll(seen)).toBe(true);
    expect(seen).toHaveBeenCalledTimes(1);
    expect(client.hasPending()).toBe(false);
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_ack_draft")).toHaveLength(1);
  });

  it("retries only the same ack token after a daemon restart loses status", async () => {
    let ackAttempts = 0;
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: ackAttempts ? "unavailable" : "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      if (++ackAttempts === 1) throw new Error("ack response lost");
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("hello");
    await expect(client.poll(seen)).rejects.toThrow("could not be confirmed");
    expect(client.hasPending()).toBe(true);
    await expect(client.poll(seen)).resolves.toBe(true);
    expect(client.hasPending()).toBe(false);
    expect(seen).toHaveBeenCalledTimes(1);
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(1);
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_ack_draft")).toEqual([
      ["experimental_owner_ack_draft", { message_id: id, acknowledgement_token: token }],
      ["experimental_owner_ack_draft", { message_id: id, acknowledgement_token: token }],
    ]);
  });

  it("keeps an unconfirmed answer when the restarted daemon rejects its token", async () => {
    let ackAttempts = 0;
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: ackAttempts ? "unavailable" : "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      ackAttempts++;
      throw new Error("acknowledgment unavailable");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("hello");
    await expect(client.poll(seen)).rejects.toThrow("could not be confirmed");
    await expect(client.poll(seen)).rejects.toThrow("could not be confirmed");
    await expect(client.poll(seen)).rejects.toThrow("could not be confirmed");
    expect(client.hasPending()).toBe(true);
    expect(ackAttempts).toBe(2);
    expect(seen).toHaveBeenCalledTimes(1);
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(1);
  });

  it("does not treat a malformed acknowledgment response as success", async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      return { status: "unexpected" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("could not be confirmed");
    expect(client.hasPending()).toBe(true);
  });

  it("retries only the same exact token if an ack was not committed", async () => {
    let ackAttempts = 0;
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: ackAttempts ? "started_unconfirmed" : "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      if (++ackAttempts === 1) throw new Error("socket dropped");
      return { status: "client_acknowledged" };
    });
    const client = new OwnerConversation(request as OwnerRequest);
    const seen = vi.fn();
    await client.send("hello");
    await expect(client.poll(seen)).rejects.toThrow("could not be confirmed");
    expect(await client.poll(seen)).toBe(true);
    const acks = request.mock.calls.filter(([method]) => method === "experimental_owner_ack_draft");
    expect(acks).toHaveLength(2);
    expect(acks[0][1]).toEqual(acks[1][1]);
    expect(seen).toHaveBeenCalledTimes(1);
    expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(1);
  });

  it("does not endlessly retry an acknowledgment that keeps failing", async () => {
    let ackAttempts = 0;
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === "experimental_owner_status") return params?.message_id ? { status: ackAttempts ? "started_unconfirmed" : "ready" } : { processing_ready: true };
      if (method === "experimental_owner_ingress") return { message_id: id };
      if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
      ++ackAttempts;
      throw new Error("socket dropped");
    });
    const client = new OwnerConversation(request as OwnerRequest);
    await client.send("hello");
    await expect(client.poll(() => {})).rejects.toThrow("could not be confirmed");
    await expect(client.poll(() => {})).rejects.toThrow("could not be confirmed");
    await expect(client.poll(() => {})).rejects.toThrow("could not be confirmed");
    expect(ackAttempts).toBe(2);
  });
});

it("distinguishes active work from a visible answer whose acknowledgement failed", async () => {
  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
    if (method === "experimental_owner_ingress") return { message_id: id };
    if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
    throw new Error("ack response lost");
  });
  const client = new OwnerConversation(request as OwnerRequest);
  await client.send("first");
  expect(client.blockingState()).toBe("working");
  const seen = vi.fn();
  await expect(client.poll(seen)).rejects.toThrow("saving it to conversation history");
  expect(seen).toHaveBeenCalledOnce();
  expect(client.blockingState()).toBe("acknowledgement");
  await expect(client.send("second")).rejects.toThrow("answer is shown");
  expect(request.mock.calls.filter(([method]) => method === "experimental_owner_ingress")).toHaveLength(1);
});

it("retries only the retained acknowledgement token on explicit recovery", async () => {
  let failAck = true;
  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
    if (method === "experimental_owner_ingress") return { message_id: id };
    if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
    if (failAck) throw new Error("lost acknowledgement");
    return { status: "client_acknowledged" };
  });
  const client = new OwnerConversation(request as OwnerRequest);
  await client.send("first");
  await expect(client.poll(() => {})).rejects.toThrow("saving it to conversation history");
  failAck = false;
  expect(await client.retryAcknowledgement()).toBe(true);
  expect(client.blockingState()).toBe("idle");
  expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(1);
  expect(request.mock.calls.filter(([method]) => method === "experimental_owner_ack_draft")).toEqual([
    ["experimental_owner_ack_draft", { message_id: id, acknowledgement_token: token }],
    ["experimental_owner_ack_draft", { message_id: id, acknowledgement_token: token }],
  ]);
});

it("does not claim recovery when acknowledgement is in flight or pickup lacks a token", async () => {
  let finish!: (value: {status: string}) => void;
  const waiting = new Promise<{status: string}>((resolve) => { finish = resolve; });
  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === "experimental_owner_status") return params?.message_id ? { status: "ready" } : { processing_ready: true };
    if (method === "experimental_owner_ingress") return { message_id: id };
    if (method === "experimental_owner_take_draft") return { message_id: id, message: "answer", acknowledgement_token: token, delivery_status: "started_unconfirmed" };
    return waiting;
  });
  const client = new OwnerConversation(request as OwnerRequest);
  await client.send("first");
  expect(await client.retryAcknowledgement()).toBe(false);
  const polling = client.poll(() => {});
  await vi.waitFor(() => expect(request.mock.calls.some(([method]) => method === "experimental_owner_ack_draft")).toBe(true));
  expect(await client.retryAcknowledgement()).toBe(false);
  finish({status: "client_acknowledged"});
  await polling;
  expect(client.hasPending()).toBe(false);
});

it("adopts only a released answer that is ready and never adopts uncertain delivery", async () => {
  let state = "started_unconfirmed";
  const request = vi.fn(async (method: string) => {
    if (method === "experimental_owner_status") return {status: state};
    if (method === "experimental_owner_take_draft") return {message_id: id, message: "follow-up answer", acknowledgement_token: token, delivery_status: "started_unconfirmed"};
    return {status: "client_acknowledged"};
  });
  const client = new OwnerConversation(request as OwnerRequest);
  expect(await client.adoptReleased(id)).toBe(false);
  expect(client.hasPending()).toBe(false);
  state = "ready";
  expect(await client.adoptReleased(id)).toBe(true);
  const seen = vi.fn();
  await client.poll(seen);
  expect(seen).toHaveBeenCalledExactlyOnceWith(id, "follow-up answer", "answer");
  expect(client.hasPending()).toBe(false);
  expect(request.mock.calls.filter(([method]) => method === "experimental_owner_take_draft")).toHaveLength(1);
});

it("uses the supplied stable id for follow-up submissions and reports a late race", async () => {
  const request = vi.fn(async () => ({ status: "late" }));
  const client = new OwnerConversation(request as OwnerRequest);
  expect(await client.queueFollowup("Add apples", id, token)).toBe("late");
  expect(request).toHaveBeenCalledExactlyOnceWith("experimental_owner_followup_add", {
    active_message_id: id, prompt: "Add apples", request_id: token,
  });
});
