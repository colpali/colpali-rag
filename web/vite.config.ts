import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the studio API to the FastAPI backend (`colpali-rag studio`).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
