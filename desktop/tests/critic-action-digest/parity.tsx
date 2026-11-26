import { CalendarDays, Mail } from "lucide-react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { decodePrivateProviderSurface } from "../../src/a2ui";
import { ConversationMessage } from "../../src/ConversationMessage";
import reference from "./reference.html?raw";
import fixtures from "./data.json";
import "../../src/styles.css";

const query = new URLSearchParams(location.search);
const providers = ["gmail", "calendar"] as const;
const width = query.get("width") === "350" ? 350 : 670;
const scheme = query.get("theme") === "dark" ? "dark" : "light";
const documentRoot = document.documentElement;
documentRoot.dataset.alphonseTheme = scheme;
documentRoot.style.background = scheme === "dark" ? "#151a18" : "#f2f3eb";
const root = createRoot(document.getElementById("root")!);

if (query.get("view") === "reference") {
  const stylesheet = reference.match(/<style>[\s\S]*?<\/style>/)?.[0];
  const interaction = reference.match(/<script>[\s\S]*?<\/script>/)?.[0] ?? "";
  const articles = reference.split('data-variant="Evidence review"')[1]?.match(/<article[\s\S]*?<\/article>/g);
  if (!stylesheet || articles?.length !== 2) throw Error("Approved Evidence review not found");
  const icons = [<Mail size={16} aria-hidden />, <CalendarDays size={16} aria-hidden />];
  const cards = articles.map((article, index) => article.replace(/<i data-lucide="[^"]+" aria-hidden="true"><\/i>/, renderToStaticMarkup(icons[index])));
  const srcDoc = `<!doctype html><html lang="en" style="color-scheme:${scheme}"><head>
    <meta name="viewport" content="width=device-width, initial-scale=1">${stylesheet}
    <style>* { box-sizing: border-box } body { margin: 0; background: ${documentRoot.style.background} }</style>
    </head><body><div id="alphonse-card-mocks"><div id="alphonse-evidence" class="prototype" style="width:calc(${width}px + 2 * clamp(10px, 2vw, 18px));max-width:none;border:0;margin:0;border-radius:0"><div class="content" data-parity-panel>${cards.join("")}</div></div></div>${interaction}</body></html>`;
  // Replace the disposable test document so the reference also runs in native WebKit with frame-src 'none'.
  document.open();
  document.write(srcDoc);
  document.close();
} else {
  root.render(<div className="parity-host provider-review-panel" data-parity-panel style={{ width }}>
    {providers.map((provider) => {
      const id = `fixture-${provider}`;
      const surface = decodePrivateProviderSurface(fixtures.parity[provider], id, provider);
      return <ConversationMessage key={id} message={{ id, role: "assistant", content: "Private surface could not be decoded",
        private_reply_owner: "synthetic-owner", private_reply_surface: surface }} timezone="America/Mexico_City" />;
    })}
  </div>);
}
