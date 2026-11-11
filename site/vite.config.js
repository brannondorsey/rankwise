import { defineConfig } from "vite";

// base "./" so the built site works from any path (GitHub Pages project sites included)
export default defineConfig({
  base: "./",
  build: { target: "es2020", assetsInlineLimit: 0, chunkSizeWarningLimit: 900 },
  server: { host: true },
});
