async (page) => {
  const url = "http://127.0.0.1:4180/desktop/tests/window-blink/fixture.html";
  async function start(tab, method) {
    await tab.waitForFunction(() => window.reloadProbe, null, { polling: 50, timeout: 8000 });
    await tab.evaluate((name) => window.reloadProbe.start(name), method);
    await tab.waitForFunction(() => window.reloadProbe.result || window.reloadProbe.error, null, { polling: 50, timeout: 18000 });
    const view = await tab.evaluate(() => ({ result: window.reloadProbe.result, error: window.reloadProbe.error }));
    if (view.error) throw new Error(view.error);
    return view.result;
  }
  const results = [];
  for (const mode of ["control", "reload", "restored"]) {
    const tab = page;
    await tab.evaluate(() => sessionStorage.removeItem("alphonse.synthetic-window-blink.v1"));
    await tab.goto(url + "?case=" + mode + (mode === "restored" ? "&history=restore" : ""));
    const before = await start(tab, "prepare");
    if (before.boot !== 1 || !before.working || before.cards.join() !== "1,2") throw new Error("Fixture not prepared");
    if (mode === "control") {
      results.push({ mode, before, after: await start(tab, "finishNormally") });
    } else {
      await tab.reload();
      const after = await start(tab, "settleAfterReload");
      results.push({ mode, before, scrolling: false, ...after });
      await tab.screenshot({ path: "output/playwright/desktop-window-blink/chromium-" + mode + ".png" });
    }
  }
  return results;
}
