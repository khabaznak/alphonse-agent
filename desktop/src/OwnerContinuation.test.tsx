import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ContinuationChoices, chooseContinuation, listContinuations } from "./OwnerContinuation";
import type { OwnerRequest } from "./ownerConversation";

const item = { message_id: "a".repeat(32), checkpoint_version: 4, question: "Continue the investigation?" };

describe("owner continuation choices", () => {
  it("shows the server question with explicit choices", () => {
    const html = renderToStaticMarkup(<ContinuationChoices items={[item]} busy="" onChoose={() => {}} />);
    expect(html).toContain("Continue the investigation?");
    expect(html).toContain(">Continue</button>");
    expect(html).toContain(">Don’t continue</button>");
  });
  it("binds a choice to the exact server checkpoint, without sending an actor or question", async () => {
    const request = vi.fn(async () => ({ status: "accepted" }));
    await chooseContinuation(request as OwnerRequest, item, true);
    expect(request).toHaveBeenCalledExactlyOnceWith("experimental_owner_continuation_choose", {
      message_id: item.message_id, checkpoint_version: 4, approve: true,
    });
  });
  it("rejects malformed or unavailable server lists", async () => {
    const request = vi.fn(async () => ({ items: [{ ...item, checkpoint_version: "4" }] }));
    await expect(listContinuations(request as OwnerRequest)).rejects.toThrow();
  });
});
