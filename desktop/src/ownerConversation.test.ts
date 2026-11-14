import { describe, expect, it, vi } from "vitest";
import { assertLegacyConversationAvailable, OwnerConversation, type OwnerRequest } from "./ownerConversation";

const id = "a".repeat(32);
const token = "b".repeat(32);

describe("explicit owner conversation transport", () => {
  it("blocks the legacy send route while the exclusive owner worker is running", () => {
    expect(() => assertLegacyConversationAvailable(true)).toThrow("Turn on the simpler conversation loop");
    expect(() => assertLegacyConversationAvailable(false)).not.toThrow();
  });
  it("does not submit unless the daemon confirms a connected owner worker", async () => {
    const request = vi.fn(async () => ({ processing_ready: false }));
    const client = new OwnerConversation(request as OwnerRequest);
    await expect(client.send("hello")).rejects.toThrow("not ready");
    expect(request).toHaveBeenCalledTimes(1);
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
    expect(seen).toHaveBeenCalledExactlyOnceWith(id, "answer");
    expect(calls).toEqual(["experimental_owner_status", "experimental_owner_ingress", "experimental_owner_status", "experimental_owner_take_draft", "display", "experimental_owner_ack_draft"]);
    expect(request).toHaveBeenLastCalledWith("experimental_owner_ack_draft", { message_id: id, acknowledgement_token: token });
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
    expect(seen).toHaveBeenCalledExactlyOnceWith(id, "answer");
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
