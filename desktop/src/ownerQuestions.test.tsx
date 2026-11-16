import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { OwnerQuestionSelector } from "./OwnerQuestionSelector";
import { listOwnerQuestions } from "./ownerQuestions";
import type { OwnerRequest } from "./ownerConversation";

const id = "c".repeat(32);

describe("explicit owner question selection", () => {
  it("lists only the server's confirmed questions without choosing one", async () => {
    const questions = [{ message_id: id, question: "Which one?" }];
    const request = vi.fn(async () => ({ questions }));
    expect(await listOwnerQuestions(request as OwnerRequest)).toEqual(questions);
    expect(request).toHaveBeenCalledExactlyOnceWith("experimental_owner_questions", {});
    const html = renderToStaticMarkup(<OwnerQuestionSelector questions={questions} selectedId="" disabled={false} onSelect={() => {}} />);
    expect(html).toContain("Reply to: Which one?");
    expect(html).toContain('aria-pressed="false"');
    expect(html).not.toContain("Replying to:");
  });

  it("shows the selected target and a way to cancel it", () => {
    const html = renderToStaticMarkup(<OwnerQuestionSelector questions={[{ message_id: id, question: "Which one?" }]} selectedId={id} disabled={false} onSelect={() => {}} />);
    expect(html).toContain("Replying to: Which one?");
    expect(html).toContain("Cancel reply");
    expect(html).toContain('aria-pressed="true"');
  });

  it("rejects malformed question results", async () => {
    const request = vi.fn(async () => ({ questions: [{ message_id: "forged", question: "Which?" }] }));
    await expect(listOwnerQuestions(request as OwnerRequest)).rejects.toThrow("could not be checked");
  });
});
