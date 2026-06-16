import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    // The ':' in this repo's path breaks Vite's fs allow-list prefix match,
    // wrongly flagging files as "outside" the project. Relax it for dev only
    // (server options don't affect `vite build` / production).
    fs: { strict: false },
    proxy: { "/api": "http://localhost:8080", "/healthz": "http://localhost:8080" },
  },
});
