// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { ConversationMessage } from "./ConversationMessage";
import { decodePrivateProviderSurface, type A2uiComponent } from "./a2ui";

const components: A2uiComponent[] = [
  { id: "root", component: "Card", children: ["title", "findings", "citation", "sources"] },
  { id: "title", component: "Text", variant: "heading", text: "Deadlines & next steps" },
  { id: "findings", component: "Markdown", text:
    "## What changed\n\n- **New date** [8](#source-8)\n- [External](https://example.invalid) [9](#source-9)\n\n![Image](https://example.invalid/pixel)\n<script>no</script>" },
  { id: "citation", component: "EmailCitation", sourceNumber: 8 },
  { id: "sources", component: "Disclosure", label: "Cited emails", children: ["source_list"] },
  { id: "source_list", component: "List", children: ["source_8"] },
  { id: "source_8", component: "Column", children: ["source_8_title", "source_8_excerpt"] },
  { id: "source_8_title", component: "Text", variant: "strong", text: "Packing [updated] <latest>" },
  { id: "source_8_excerpt", component: "Text", variant: "quote", text: "**literal** <img src=https://example.invalid>" },
];

function stream(items: A2uiComponent[], messageId = "mail") {
  const surfaceId = `private-gmail:${messageId}`;
  return [
    { version: "v0.9.1", createSurface: { surfaceId, catalogId: "alphonse.desktop.catalog.v2", sendDataModel: false } },
    { version: "v0.9.1", updateComponents: { surfaceId, components: items } },
  ];
}

describe("private Action digest", () => {
  it("does not admit a citation to another source even if the source number is syntactically valid", () => {
    const invalid = components.map((item) => item.id === "citation" ? { ...item, sourceNumber: 9 } : item);
    expect(decodePrivateProviderSurface(stream(invalid), "mail", "gmail")).toBeUndefined();
  });

  it("formats synthesis and opens and focuses the exact literal excerpt from either citation form", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    const host = document.body.appendChild(document.createElement("div"));
    const react = createRoot(host);
    try {
      const surface = decodePrivateProviderSurface(stream(components), "mail", "gmail");
      expect(surface).toBeDefined();
      await act(async () => react.render(<ConversationMessage timezone="America/Mexico_City" message={{
        id: "mail", role: "assistant", content: "fallback", private_reply_owner: "owner", private_reply_surface: surface,
      }} />));
      expect(host.querySelector("h4")?.textContent).toBe("What changed");
      expect(host.querySelector(".provider-markdown ul li strong")?.textContent).toBe("New date");
      expect(host.querySelectorAll("a, img, script")).toHaveLength(0);
      const citations = [...host.querySelectorAll<HTMLButtonElement>('button[aria-label="Read email source 8"]')];
      expect(citations).toHaveLength(2);
      const disclosure = host.querySelector("details")!;
      expect(disclosure.open).toBe(false);
      for (const button of citations) {
        disclosure.open = false;
        await act(async () => button.click());
        const source = host.querySelector('[data-email-source="8"]')!;
        expect(disclosure.open).toBe(true);
        expect(document.activeElement).toBe(source);
        expect(source.querySelector("blockquote")?.textContent).toBe("**literal** <img src=https://example.invalid>");
        expect(source.querySelector("blockquote strong")).toBeNull();
      }
    } finally {
      await act(async () => react.unmount());
      host.remove();
      vi.unstubAllGlobals();
    }
  });
});
