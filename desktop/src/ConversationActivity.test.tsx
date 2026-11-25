import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { ConversationActivity } from "./ConversationActivity";

it("shows an accessible in-conversation working indicator without repeated activity indicators", () => {
  const html = renderToStaticMarkup(<ConversationActivity />);
  expect(html).toContain('role="status"');
  expect(html).toContain("Alphonse is working");
  expect(html).not.toContain("conversation-activity-grid");
});
