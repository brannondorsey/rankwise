import { defineConfig } from "vite";

// Preview of the redesigned site, published at <site>/v2/. It shares site/public (the demo
// data) with the current site instead of keeping a second copy.
export default defineConfig({
  base: "./",
  publicDir: "../site/public",
  build: { target: "es2020", assetsInlineLimit: 0, chunkSizeWarningLimit: 900 },
  server: { host: true },
});
