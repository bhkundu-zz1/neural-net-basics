import path from "path";
import { fileURLToPath } from "url";
import { test, expect } from "@playwright/test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SAMPLE_PORTFOLIO_CSV = path.join(__dirname, "fixtures", "sample-portfolio.csv");

test.describe("get a signal on an uploaded portfolio", () => {
  test("upload a portfolio CSV and see a per-position signal table", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Paper Trading Simulator" })).toBeVisible();

    await page.getByRole("button", { name: "Get Signal on Portfolio" }).click();
    await expect(page.getByRole("heading", { name: "Get signal on portfolio" })).toBeVisible();

    await page.getByLabel("Portfolio CSV").setInputFiles(SAMPLE_PORTFOLIO_CSV);
    await page.getByRole("button", { name: "Get Signal on Portfolio" }).click();

    // One live pipeline run per distinct ticker — allow generous time.
    await expect(page.locator(".portfolio-result-table")).toBeVisible({ timeout: 30_000 });
    await expect(page.locator(".portfolio-result-table")).toContainText("NVDA");
    await expect(page.locator(".portfolio-result-table")).toContainText("AAPL");

    await expect(page.locator(".portfolio-summary")).toContainText("Total portfolio value");
  });
});
