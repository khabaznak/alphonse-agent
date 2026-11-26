async (page) => {
  const url = "http://localhost:1420/tests/critic-action-digest/parity.html";
  const directory = `output/playwright/evidence-review-parity/${Date.now()}`;

  for (const theme of ["light", "dark"]) {
    await page.emulateMedia({ colorScheme: theme, reducedMotion: "reduce" });
    for (const width of [350, 670]) {
      await page.setViewportSize({ width: width === 350 ? 380 : 800, height: 1300 });
      for (const view of ["reference", "production"]) {
        const query = new URLSearchParams({ theme, width: String(width), view });
        await page.goto(`${url}?${query}`);
        const panel = page.locator("[data-parity-panel]");
        const cards = panel.locator(view === "reference" ? ".result" : ".provider-result-card");
        await cards.nth(1).waitFor({ state: "visible" });
        await page.evaluate(() => document.fonts.ready.then(() => {}));
        for (const [state, target] of [["panel", panel], ["gmail", cards.first()], ["calendar", cards.last()]]) {
          await target.screenshot({ path: `${directory}/${theme}-${width}-${state}-${view}.png`, animations: "disabled", scale: "css" });
        }
        await cards.first().getByRole("button", { name: /Read (email )?source 8/ }).click();
        await page.mouse.move(1, 1);
        await cards.first().screenshot({ path: `${directory}/${theme}-${width}-gmail-selected-${view}.png`, animations: "disabled", scale: "css" });
      }
    }
  }
  return directory;
}
