import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { ConversationActivity } from "./ConversationActivity";

it("shows an accessible in-conversation working indicator with nine dots", () => {
  const html = renderToStaticMarkup(<ConversationActivity />);
  expect(html).toContain('role="status"');
  expect(html).toContain("Alphonse is working");
  expect(html.match(/<i><\/i>/g)).toHaveLength(9);
});
