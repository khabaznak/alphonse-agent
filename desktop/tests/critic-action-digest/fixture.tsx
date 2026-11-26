import { useState } from "react";
import { createRoot } from "react-dom/client";
import { decodePrivateProviderSurface } from "../../src/a2ui";
import { ConversationMessage } from "../../src/ConversationMessage";
import fixtures from "./data.json";
import "../../src/styles.css";

function Fixture() {
  const [compact, setCompact] = useState(false);
  const [empty, setEmpty] = useState(false);
  return <main className="fixture">
    <h1>Evidence review Critic fixture</h1>
    <p>Synthetic Gmail and Google Calendar results</p>
    <div className="fixture-controls">
      <button type="button" aria-pressed={compact} onClick={() => setCompact(!compact)}>Compact conversation</button>
      <button type="button" aria-pressed={empty} onClick={() => setEmpty(!empty)}>Empty results</button>
    </div>
    <div className={`fixture-thread ${compact ? "compact" : ""}`}>
      {(["gmail", "calendar"] as const).map((provider) => {
        const id = `fixture-${provider}${empty ? "-empty" : ""}`;
        const surface = decodePrivateProviderSurface(fixtures[empty ? "empty" : "review"][provider], id, provider);
        return <ConversationMessage key={id} timezone="America/Mexico_City" message={{
          id, content: "The A2UI fixture did not decode.", role: "assistant",
          private_reply_owner: "synthetic-owner", private_reply_surface: surface,
        }} />;
      })}
    </div>
  </main>;
}

createRoot(document.getElementById("root")!).render(<Fixture />);
