import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

export default defineConfig({
  resolve: {
    alias: {
      "@ploshcha/contract-ts": resolve(repoRoot, "packages/contract-ts/src/index.ts"),
      "@fixtures": resolve(repoRoot, "packages/fixtures"),
    },
  },
  server: {
    fs: { allow: [repoRoot] },
  },
});
