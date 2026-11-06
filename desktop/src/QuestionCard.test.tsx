import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QuestionCard, submitQuestionAnswer } from "./QuestionCard";
import { daemonRequest } from "./api";
import type { Question } from "./types";
vi.mock("./api", () => ({ daemonRequest: vi.fn() }));
const question: Question = { question_id: "q", task_id: "t", project_id: "", kind: "open_text", message: "What next?", choices: [] };
beforeEach(() => vi.clearAllMocks());
describe("question submission", () => {
  it("binds input to the question and actor and accepts only a confirmed matching answer", async () => {
    vi.mocked(daemonRequest).mockResolvedValue({ handled: true, question_id: "q", task_id: "t", answer: { text: "Next" } });
    await submitQuestionAnswer(question, "respondent", { text: "Next" });
    expect(daemonRequest).toHaveBeenCalledWith("answer_question", { user: "respondent", question_id: "q", payload: { text: "Next" } });
  });
  it.each([{handled:false}, {handled:true, invalid:true, message:"Invalid input"}, {handled:true, question_id:"other", task_id:"t", answer:{text:"Wrong"}}, {handled:true, question_id:"q", task_id:"other", answer:{text:"Wrong"}}])("rejects unavailable, invalid and misrouted answers", async (result) => {
    vi.mocked(daemonRequest).mockResolvedValue(result);
    await expect(submitQuestionAnswer(question,"respondent",{text:"Next"})).rejects.toThrow();
  });
  it("propagates transport failure for a visible retry", async () => {
    vi.mocked(daemonRequest).mockRejectedValue(new Error("Disconnected"));
    await expect(submitQuestionAnswer(question,"respondent",{text:"Next"})).rejects.toThrow("Disconnected");
  });
});
describe("recognizable schema-driven questions", () => {
  const render = (overrides: Partial<Question>) => renderToStaticMarkup(<QuestionCard question={{...question,...overrides}} user="respondent" timezone="America/Mexico_City" onDone={async()=>{}} />);
  it("labels text input and shows only the authorized context projection", () => {
    const html=render({context:{text:"Review fixture",artifacts:[{name:"Fixture",description:"Read-only context",status:"ready"},{status:"unavailable"}]}});
    expect(html).toContain("Alphonse needs your input");
    expect(html).toContain("Your answer");
    expect(html).toContain("Review fixture");
    expect(html).toContain("Related resource unavailable or no longer authorized.");
    expect(html).toContain("disabled");
  });
  it("renders deliberate yes/no actions and native multiple-choice controls", () => {
    expect(render({kind:"yes_no"})).toContain(">No</button>");
    expect(render({kind:"yes_no"})).toContain(">Yes</button>");
    const choices=[{id:"a",label:"Alpha"},{id:"b",label:"Beta"}];
    expect(render({kind:"single_choice",choices})).toContain(">Alpha</button>");
    expect(render({kind:"multi_choice",choices})).toContain('type="checkbox"');
    expect(render({kind:"datetime"})).toContain("America/Mexico_City");
  });
  it("disables expired questions and offers refresh", () => {
    const html=render({expires_at:"2000-01-01T00:00:00Z"});
    expect(html).toContain("This question has expired.");
    expect(html).toContain("Refresh questions");
    expect(html).toContain('<fieldset disabled=""');
  });
});
