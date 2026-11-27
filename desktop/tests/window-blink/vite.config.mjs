import { fileURLToPath } from "node:url";
export default {
  root: fileURLToPath(new URL("../../..", import.meta.url)),
  publicDir: fileURLToPath(new URL("../../../desktop/public", import.meta.url)),
  cacheDir: "/tmp/alphonse-window-blink-vite-cache",
  esbuild: { jsx: "automatic" },
  resolve: { alias: {
    react: fileURLToPath(new URL("../../../desktop/node_modules/react", import.meta.url)),
    "react-dom": fileURLToPath(new URL("../../../desktop/node_modules/react-dom", import.meta.url)),
  } },
  server: { strictPort: true, port: 4180, host: "127.0.0.1" },
};
