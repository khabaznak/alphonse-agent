import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AtHandDrawer } from "./AtHandDrawer";
import { defaultDrawer } from "./atHandState";
const props = { state: { ...defaultDrawer(), open: true }, act: () => {}, member: "alice", connected: true, questionCount: 7, questionsMember: "alice", onAttention: () => {}, onAgenda: () => {}, toggleRef: { current: null }, saveFailed: false };
describe("At hand rendering boundaries", () => {
  it("never displays previous-member question counts or disconnected data", () => {
    expect(renderToStaticMarkup(<AtHandDrawer {...props} />)).toContain("7 waiting for your input");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} questionsMember="bob" />)).not.toContain("7 waiting");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} connected={false} />)).not.toContain("7 waiting");
    expect(renderToStaticMarkup(<AtHandDrawer {...props} member="" />)).toContain("Loading member");
  });
  it("labels keyboard controls, renders expiration, and explains storage failure", () => {
    const html = renderToStaticMarkup(<AtHandDrawer {...props} state={{ ...props.state, widgets: [{ id: "trip", kind: "agenda", mode: "contextual", expiresAt: 1 }] }} saveFailed />);
    expect(html).toContain("Context expired");
    expect(html).toContain('aria-label="Pin Household agenda"');
    expect(html).toContain('aria-label="Remove Household agenda"');
    expect(html).toContain("Layout could not be saved");
  });
});
