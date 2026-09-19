import { test, expect } from "@playwright/test";

test.describe("place and track a paper trade", () => {
  test("get a signal, view trade history, and resolve a seeded backdated trade", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Paper Trading Simulator" })).toBeVisible();

    // --- Get a signal ---
    await page.getByLabel("Ticker").fill("NVDA");
    await page.getByRole("main").getByRole("button", { name: "Get Signal" }).click();

    await expect(page.locator(".signal-result h3")).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(".signal-result")).toContainText("NVDA");

    // Place Paper Trade is only enabled when should_trade is true — this is
    // data-dependent (per docs/pipeline_guide.md, only ~7.5% of signals clear
    // the 0.75 threshold), so this scenario does not assert the button is
    // enabled/clickable here. The seeded-trade path below is what actually
    // exercises "a trade exists and gets resolved."

    // --- Trade history shows the seeded backdated trade ---
    await page.getByRole("button", { name: "Trade History" }).click();
    await expect(page.getByText("AAPL")).toBeVisible({ timeout: 10_000 });
    await expect(page.locator(".status-badge").first()).toHaveText("open");

    // --- Resolve outstanding trades ---
    await page.getByRole("button", { name: "Resolve outstanding trades" }).click();
    await expect(page.getByText(/Resolved \d+, still pending \d+/)).toBeVisible({ timeout: 15_000 });

    // The seeded AAPL trade (entry 2026-08-01, 5 trading-day horizon) should
    // now be resolved to won or lost, not still "open".
    const aaplRow = page.locator("tr", { hasText: "AAPL" });
    await expect(aaplRow.locator(".status-badge")).not.toHaveText("open");
  });
});
