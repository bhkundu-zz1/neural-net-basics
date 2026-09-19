import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  timeout: 30_000,
  use: {
    baseURL: process.env.FRONTEND_URL || "http://localhost:5173",
    headless: true,
  },
});
