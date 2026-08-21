import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

export default defineConfig({
  // GitHub Pages віддає сайт із підтеки `/<репо>/`, тож шляхи до ассетів мусять бути відносні
  // саме до неї. Локально й на власному домені база лишається кореневою.
  base: process.env.PLOSHCHA_BASE ?? "/",
  resolve: {
    alias: {
      "@ploshcha/contract-ts": resolve(repoRoot, "packages/contract-ts/src/index.ts"),
      "@fixtures": resolve(repoRoot, "packages/fixtures"),
    },
  },
  server: {
    fs: { allow: [repoRoot] },
  },
  build: {
    target: "es2022", // код і так на ES2022 (tsconfig) → без зайвого down-level
    rollupOptions: {
      // стабільний pixi-чанк окремо від коду застосунку → не інвалідується при кожній зміні
      output: { manualChunks: { pixi: ["pixi.js"] } },
    },
  },
});
