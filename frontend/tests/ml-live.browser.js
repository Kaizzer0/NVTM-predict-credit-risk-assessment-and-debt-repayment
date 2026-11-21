// Playwright runCode function. Use with the running local test stack and demo accounts.
async (page) => {
  const base = "http://localhost:5173";
  const api = "http://localhost:8000";
  const checks = [];
  const failedRequests = [];
  const pageErrors = [];
  const collectResponse = (response) => {
    if (response.url().startsWith(api) && response.status() >= 400) {
      failedRequests.push(`${response.status()} ${response.url()}`);
    }
  };
  const collectError = (error) => pageErrors.push(error.message);
  page.on("response", collectResponse);
  page.on("pageerror", collectError);
  const check = (condition, name) => {
    if (!condition) throw new Error(name);
    checks.push(name);
  };

  try {
    await page.goto(base + "/login");
    await page.evaluate(() => localStorage.removeItem("access_token"));
    await page.goto(base + "/login");
    await page.getByRole("button", { name: /Admin:\s+admin@/ }).click();
    await page.getByRole("button", { name: "Đăng nhập", exact: true }).click();
    await page.waitForURL(base + "/");
    check(true, "Đăng nhập admin");

    const modelResponse = page.waitForResponse((r) => r.url() === api + "/api/models/current");
    await page.goto(base + "/models");
    const active = await (await modelResponse).json();
    const table = page.getByRole("table").filter({ hasText: "Mean AP" });
    const names = { DecisionTree: "Decision Tree", LinearSVM: "Linear SVM", ANN: "Artificial Neural Network (ANN)" };
    for (const [algorithm, name] of Object.entries(names)) {
      const row = table.getByRole("row").filter({ hasText: name });
      await row.waitFor();
      const cells = await row.getByRole("cell").allTextContents();
      const result = active.cv_results.models[algorithm];
      check(
        result.folds.every((fold, index) => cells[index] === fold.ap.toFixed(6))
          && cells[5] === result.summary.ap_mean.toFixed(6)
          && cells[6] === result.summary.ap_std.toFixed(6),
        `${name} hiển thị AP của 5 fold, mean và std từ API`,
      );
    }
    await page.getByText(names[active.algorithm], { exact: true }).first().waitFor();
    await page.getByText(active.test_metrics.roc_auc.toFixed(6), { exact: true }).waitFor();
    await page.getByText("Ma trận nhầm lẫn", { exact: true }).waitFor();
    await page.locator("path.recharts-line-curve").nth(1).waitFor();
    await page.getByRole("button", { name: "Bắt đầu huấn luyện" }).waitFor();
    check(true, "Mô hình đang phục vụ, kết quả Test, PR/ROC và nút huấn luyện");

    await page.goto(base + "/customers/new");
    const sample = await page.evaluate(async (url) => {
      const res = await fetch(url, { headers: { Authorization: `Bearer ${localStorage.getItem("access_token")}` } });
      if (!res.ok) throw new Error(`Không tải được mẫu Training: ${res.status}`);
      return res.json();
    }, api + "/api/home-credit/sample");
    const name = `Kiểm thử Home Credit ${Date.now()}`;
    await page.locator("#case-name").fill(name);
    for (const [field, value] of Object.entries(sample)) {
      if (field === "amount_unit") continue;
      const input = page.locator("#" + field);
      if (value === null) {
        await input.locator("..").getByRole("checkbox").check();
      } else if (await input.evaluate((element) => element.tagName === "SELECT")) {
        await input.selectOption(String(value));
      } else {
        await input.fill(String(value));
      }
    }
    await page.locator("#amount-confirmation").check();
    const scoreResponse = page.waitForResponse((r) => /\/api\/credit-cases\/\d+\/score$/.test(r.url()));
    await page.getByRole("button", { name: "Dự đoán và lưu kết quả" }).click();
    const scored = await scoreResponse;
    check(scored.status() === 201, "Dự đoán và lưu hồ sơ");
    const prediction = await scored.json();
    await page.getByRole("heading", { name: `Kết quả đã lưu #${prediction.id}` }).waitFor();
    check(prediction.model_version === active.version && prediction.response_json.probability_calibrated, "Kết quả dùng đúng mô hình và xác suất đã hiệu chỉnh");
    await page.goto(base + "/predictions");
    const saved = page.getByRole("row").filter({ hasText: name });
    await saved.waitFor();
    check((await saved.textContent()).includes(active.version), "Lịch sử lưu đúng phiên bản mô hình");
    check(failedRequests.length === 0 && pageErrors.length === 0, "Không có API lỗi hoặc lỗi JavaScript trong luồng");
    return { passed: checks, caseId: prediction.case_id, predictionId: prediction.id };
  } finally {
    page.off("response", collectResponse);
    page.off("pageerror", collectError);
  }
};
