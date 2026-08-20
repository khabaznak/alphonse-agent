import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { getDesktopDiagnosticMode, getDesktopDiagnosticProjectId } from "./api";
import { parseDesktopDiagnosticMode } from "./diagnosticMode";
import "./styles.css";

async function start() {
  const [rawMode, diagnosticProjectId] = await Promise.all([
    getDesktopDiagnosticMode().catch(() => "normal"),
    getDesktopDiagnosticProjectId().catch(() => ""),
  ]);
  const diagnosticMode = parseDesktopDiagnosticMode(rawMode);
  document.documentElement.dataset.alphonseDiagnosticMode = diagnosticMode;
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App diagnosticMode={diagnosticMode} diagnosticProjectId={diagnosticProjectId.trim()} />
    </StrictMode>,
  );
}

void start();
