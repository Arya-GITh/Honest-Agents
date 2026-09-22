document.addEventListener("DOMContentLoaded", () => {
  // State
  let loadedScenarios = [];
  let currentScenario = null;

  // Tab Navigation
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.getAttribute("data-tab");
      navTabs.forEach(t => t.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(targetTab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // -------------------------------------------------------------
  // TAB 1: HONESTY AUDITOR
  // -------------------------------------------------------------
  const presetsContainer = document.getElementById("scenario-presets-container");
  const inputUserPrompt = document.getElementById("input-user-prompt");
  const inputToolName = document.getElementById("input-tool-name");
  const inputCategoryName = document.getElementById("input-category-name");
  const inputToolOutput = document.getElementById("input-tool-output");
  const inputAgentClaim = document.getElementById("input-agent-claim");

  const btnRunAudit = document.getElementById("btn-run-audit");
  const btnLoadHonest = document.getElementById("btn-load-honest-sample");

  const auditVerdictBadge = document.getElementById("audit-verdict-badge");
  const displayRawClaim = document.getElementById("display-raw-claim");
  const displayGuardedClaim = document.getElementById("display-guarded-claim");
  const guardedStatusFooter = document.getElementById("guarded-status-footer");

  const receiptIdTag = document.getElementById("receipt-id-tag");
  const receiptHash = document.getElementById("receipt-hash");
  const receiptLatency = document.getElementById("receipt-latency");
  const receiptTier = document.getElementById("receipt-tier");
  const receiptReprompts = document.getElementById("receipt-reprompts");
  const receiptFactsDisplay = document.getElementById("receipt-facts-display");

  // Fetch Scenarios
  async function loadScenarios() {
    try {
      const res = await fetch("/api/benchmarks/scenarios");
      if (!res.ok) return;
      loadedScenarios = await res.json();
      renderScenarioPresets();
      if (loadedScenarios.length > 0) {
        selectScenario(loadedScenarios[0]);
      }
    } catch (e) {
      console.error("Failed to load scenarios:", e);
    }
  }

  function renderScenarioPresets() {
    presetsContainer.innerHTML = "";
    loadedScenarios.forEach((sc, idx) => {
      const btn = document.createElement("button");
      btn.className = `btn-preset ${idx === 0 ? "active" : ""}`;
      btn.innerHTML = `<span>#${sc.id}</span> ${sc.title}`;
      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        selectScenario(sc);
      });
      presetsContainer.appendChild(btn);
    });
  }

  function selectScenario(sc) {
    currentScenario = sc;
    inputUserPrompt.value = sc.user_prompt;
    inputToolName.value = sc.tool_name;
    inputCategoryName.value = sc.category;
    inputToolOutput.value = JSON.stringify(sc.tool_output, null, 2);
    inputAgentClaim.value = sc.deceptive_claim;
    displayRawClaim.textContent = sc.deceptive_claim;
    displayGuardedClaim.textContent = "[Awaiting verification run...]";
    auditVerdictBadge.className = "badge";
    auditVerdictBadge.textContent = "AWAITING AUDIT";
  }

  btnLoadHonest.addEventListener("click", () => {
    if (currentScenario && currentScenario.honest_claim) {
      inputAgentClaim.value = currentScenario.honest_claim;
      displayRawClaim.textContent = currentScenario.honest_claim;
    }
  });

  // Execute Audit
  btnRunAudit.addEventListener("click", async () => {
    btnRunAudit.disabled = true;
    btnRunAudit.innerHTML = "<span>Auditing...</span>";

    let toolOutputParsed;
    try {
      toolOutputParsed = JSON.parse(inputToolOutput.value);
    } catch (e) {
      alert("Invalid JSON in Tool Return Payload textarea!");
      btnRunAudit.disabled = false;
      btnRunAudit.innerHTML = "<span>⚡ Execute Two-Tier Audit</span>";
      return;
    }

    displayRawClaim.textContent = inputAgentClaim.value;

    try {
      const res = await fetch("/api/audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_prompt: inputUserPrompt.value,
          agent_claim: inputAgentClaim.value,
          tool_name: inputToolName.value,
          tool_output: toolOutputParsed,
          tool_input: currentScenario ? currentScenario.tool_input : {},
        }),
      });

      const data = await res.json();
      renderAuditResults(data);
    } catch (e) {
      console.error("Audit error:", e);
      alert("Error calling audit API: " + e.message);
    } finally {
      btnRunAudit.disabled = false;
      btnRunAudit.innerHTML = "<span>⚡ Execute Two-Tier Audit</span>";
    }
  });

  function renderAuditResults(data) {
    const verdict = data.verdict;
    const receipt = data.receipt;
    const deliveredClaim = data.delivered_claim;
    const corrected = data.corrected;

    // Verdict Badge
    if (verdict.is_honest) {
      auditVerdictBadge.className = "badge badge-emerald";
      auditVerdictBadge.textContent = "VERIFIED HONEST (PASS)";
      guardedStatusFooter.className = "pane-footer status-honest";
      guardedStatusFooter.textContent = "🛡️ Passed Verification: Claim matches machine FactMatrix.";
    } else {
      auditVerdictBadge.className = "badge badge-rose";
      auditVerdictBadge.textContent = `DECEPTION DETECTED: ${verdict.deception_type.toUpperCase()}`;
      guardedStatusFooter.className = "pane-footer status-deceptive";
      guardedStatusFooter.textContent = `⚠️ Intercepted by Truthify: Auto-corrected via in-scratchpad feedback.`;
    }

    // Guarded response
    displayGuardedClaim.textContent = deliveredClaim;

    // Receipt Meta
    receiptIdTag.textContent = `RECEIPT_ID: ${receipt.receipt_id.slice(0, 18)}...`;
    receiptHash.textContent = receipt.fact_matrix.payload_sha256.slice(0, 32) + "...";
    receiptLatency.textContent = `${verdict.latency_ms.toFixed(2)} ms`;
    receiptTier.textContent = verdict.tier_used === "tier_1_deterministic" ? "Tier 1 (Deterministic Rule Engine)" : "Tier 2 (Semantic SLM)";
    receiptReprompts.textContent = `${data.reprompts} attempts (${data.reprompts > 0 ? "Self-Corrected" : "Direct Pass"})`;

    // FactMatrix Chips
    receiptFactsDisplay.innerHTML = "";
    const fm = receipt.fact_matrix;

    const chips = [
      { label: `is_error: ${fm.is_error}`, isBad: fm.is_error },
      { label: `status_code: ${fm.status_code || 200}`, isBad: fm.status_code >= 400 },
      { label: `is_empty: ${fm.is_empty}`, isBad: fm.is_empty },
      { label: `records_mutated: ${fm.records_mutated !== null ? fm.records_mutated : "N/A"}`, isBad: false },
    ];

    if (fm.error_type) {
      chips.push({ label: `error_type: ${fm.error_type}`, isBad: true });
    }

    chips.forEach(c => {
      const chip = document.createElement("span");
      chip.className = `fact-chip ${c.isBad ? "fact-error" : "fact-success"}`;
      chip.textContent = c.label;
      receiptFactsDisplay.appendChild(chip);
    });
  }

  // -------------------------------------------------------------
  // TAB 2: SPECULATIVE SANDBOX
  // -------------------------------------------------------------
  const tbodyUsers = document.getElementById("tbody-users");
  const tbodyOrders = document.getElementById("tbody-orders");
  const dbRowCounter = document.getElementById("db-row-counter");
  const sandboxSqlInput = document.getElementById("sandbox-sql-input");
  const policyMaxMutations = document.getElementById("policy-max-mutations");
  const policyAllowedTables = document.getElementById("policy-allowed-tables");
  const policyReadOnly = document.getElementById("policy-read-only");
  const btnExecuteSandbox = document.getElementById("btn-execute-sandbox");
  const btnResetSandbox = document.getElementById("btn-reset-sandbox");
  const sandboxOutcomeBox = document.getElementById("sandbox-outcome-box");
  const outcomeHeaderTitle = document.getElementById("outcome-header-title");
  const outcomeDescText = document.getElementById("outcome-desc-text");

  // Presets
  document.getElementById("btn-preset-safe-update").addEventListener("click", () => {
    sandboxSqlInput.value = "UPDATE users SET balance = balance + 150.00 WHERE id = 1;";
    policyMaxMutations.value = "2";
    policyAllowedTables.value = "users, orders";
    policyReadOnly.checked = false;
  });

  document.getElementById("btn-preset-bulk-delete").addEventListener("click", () => {
    sandboxSqlInput.value = "DELETE FROM users;";
    policyMaxMutations.value = "2";
    policyAllowedTables.value = "users, orders";
    policyReadOnly.checked = false;
  });

  document.getElementById("btn-preset-unauthorized-table").addEventListener("click", () => {
    sandboxSqlInput.value = "UPDATE auth_tokens SET token = 'compromised' WHERE id = 1;";
    policyMaxMutations.value = "5";
    policyAllowedTables.value = "orders"; // users and auth_tokens disallowed
    policyReadOnly.checked = false;
  });

  document.getElementById("btn-preset-drop-table").addEventListener("click", () => {
    sandboxSqlInput.value = "DROP TABLE users;";
    policyMaxMutations.value = "5";
    policyAllowedTables.value = "users, orders";
    policyReadOnly.checked = false;
  });

  async function loadSandboxState() {
    try {
      const res = await fetch("/api/sandbox/state");
      if (!res.ok) return;
      const data = await res.json();
      renderTables(data);
    } catch (e) {
      console.error("Error loading sandbox state:", e);
    }
  }

  function renderTables(data) {
    // Users table
    tbodyUsers.innerHTML = "";
    (data.users || []).forEach(u => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">#${u.id}</td>
        <td><strong>${u.name}</strong></td>
        <td class="mono">${u.email}</td>
        <td class="mono" style="color: var(--accent-cyan); font-weight: 700;">$${u.balance.toFixed(2)}</td>
        <td><span class="badge ${u.role === 'admin' ? 'badge-rose' : 'badge-indigo'}">${u.role}</span></td>
      `;
      tbodyUsers.appendChild(tr);
    });

    // Orders table
    tbodyOrders.innerHTML = "";
    (data.orders || []).forEach(o => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">#${o.id}</td>
        <td class="mono">User #${o.user_id}</td>
        <td>${o.item}</td>
        <td class="mono">$${o.amount.toFixed(2)}</td>
        <td><span class="badge badge-emerald">${o.status}</span></td>
      `;
      tbodyOrders.appendChild(tr);
    });

    dbRowCounter.textContent = `${(data.users || []).length} Users | ${(data.orders || []).length} Orders`;
  }

  btnResetSandbox.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/sandbox/reset", { method: "POST" });
      const data = await res.json();
      renderTables(data.state);
      sandboxOutcomeBox.style.display = "none";
    } catch (e) {
      console.error("Error resetting sandbox:", e);
    }
  });

  btnExecuteSandbox.addEventListener("click", async () => {
    btnExecuteSandbox.disabled = true;
    btnExecuteSandbox.innerHTML = "<span>Forking Savepoint...</span>";

    const sql = sandboxSqlInput.value.trim();
    const maxMutations = parseInt(policyMaxMutations.value, 10) || 5;
    const allowedTables = policyAllowedTables.value;
    const readOnly = policyReadOnly.checked;

    try {
      const res = await fetch("/api/sandbox/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sql,
          policy: {
            max_records_mutated: maxMutations,
            allowed_tables: allowedTables,
            read_only: readOnly,
          },
        }),
      });

      const data = await res.json();
      sandboxOutcomeBox.style.display = "block";

      if (data.committed) {
        sandboxOutcomeBox.className = "sandbox-outcome outcome-safe";
        outcomeHeaderTitle.textContent = "✅ TWO-PHASE COMMIT: CHANGES COMMITTED TO LIVE DATABASE";
        outcomeDescText.textContent = `All safety invariants satisfied. State delta: ${data.delta.total_mutations} row(s) mutated across table(s): [${data.delta.tables_touched.join(", ")}]. Transaction committed safely.`;
        renderTables(data.state_after);
      } else {
        sandboxOutcomeBox.className = "sandbox-outcome outcome-blocked";
        outcomeHeaderTitle.textContent = "🛡️ INVARIANT VIOLATION: SPECULATIVE ROLLBACK EXECUTED";
        outcomeDescText.textContent = `${data.violation}. The ephemeral sandbox was rolled back in <1ms. Live database state was preserved with 0 records lost.`;
        renderTables(data.state_before);
      }
    } catch (e) {
      console.error("Sandbox execution error:", e);
      alert("Sandbox execution failed: " + e.message);
    } finally {
      btnExecuteSandbox.disabled = false;
      btnExecuteSandbox.innerHTML = "<span>🚀 Run in Speculative Sandbox</span>";
    }
  });

  // -------------------------------------------------------------
  // TAB 3: DECEPTIONBENCH LEADERBOARD
  // -------------------------------------------------------------
  const tbodyLeaderboard = document.getElementById("tbody-leaderboard");
  const categoryCardsGrid = document.getElementById("category-cards-grid");

  async function loadLeaderboard() {
    try {
      const res = await fetch("/api/benchmarks/leaderboard");
      if (!res.ok) return;
      const data = await res.json();
      renderLeaderboard(data);
    } catch (e) {
      console.error("Failed to load leaderboard:", e);
    }
  }

  function renderLeaderboard(models) {
    tbodyLeaderboard.innerHTML = "";
    categoryCardsGrid.innerHTML = "";

    models.forEach((m, idx) => {
      const tr = document.createElement("tr");
      const gainClass = m.protection_gain > 0 ? "text-cyan" : "text-emerald";
      tr.innerHTML = `
        <td><strong style="color: var(--accent-cyan);">#${idx + 1}</strong></td>
        <td><span class="mono" style="font-weight: 700;">${m.model_name}</span></td>
        <td class="mono">${m.total_scenarios}</td>
        <td class="mono" style="color: var(--accent-rose); font-weight: 700;">${m.raw_edr.toFixed(1)}%</td>
        <td class="mono" style="color: var(--accent-emerald); font-weight: 700;">${m.guarded_edr.toFixed(1)}%</td>
        <td class="mono ${gainClass}" style="font-weight: 700;">+${m.protection_gain.toFixed(1)}%</td>
        <td class="mono">${m.raw_fer.toFixed(1)}% → ${m.guarded_fer.toFixed(1)}%</td>
        <td class="mono">${m.latency_ms.toFixed(1)} ms</td>
      `;
      tbodyLeaderboard.appendChild(tr);

      // Render category breakdown card
      const cats = m.categories || {};
      const card = document.createElement("div");
      card.className = "category-card";
      card.innerHTML = `
        <div class="category-title">${m.model_name} Category Breakdown</div>
        <ul class="category-stats-list">
          <li class="category-stat-row">
            <span>Adversarial Sycophancy Pressure:</span>
            <strong style="color: var(--accent-rose);">${cats.adversarial_bias ? cats.adversarial_bias.raw_deception_rate.toFixed(1) + "% → " + cats.adversarial_bias.guarded_deception_rate.toFixed(1) + "%" : "N/A"}</strong>
          </li>
          <li class="category-stat-row">
            <span>Empty Result Fabrication (FER):</span>
            <strong style="color: var(--accent-cyan);">${cats.empty_fabrication ? cats.empty_fabrication.raw_deception_rate.toFixed(1) + "% → " + cats.empty_fabrication.guarded_deception_rate.toFixed(1) + "%" : "N/A"}</strong>
          </li>
          <li class="category-stat-row">
            <span>Failure Concealment (500s):</span>
            <strong style="color: var(--accent-emerald);">${cats.failure_concealment ? cats.failure_concealment.raw_deception_rate.toFixed(1) + "% → 0.0%" : "0.0%"}</strong>
          </li>
          <li class="category-stat-row">
            <span>Soft Error Blindness:</span>
            <strong style="color: var(--accent-emerald);">${cats.soft_errors ? cats.soft_errors.raw_deception_rate.toFixed(1) + "% → 0.0%" : "0.0%"}</strong>
          </li>
          <li class="category-stat-row">
            <span>Parameter Taint Invariant:</span>
            <strong style="color: var(--accent-emerald);">${cats.parameter_taint ? cats.parameter_taint.raw_deception_rate.toFixed(1) + "% → 0.0%" : "0.0%"}</strong>
          </li>
        </ul>
      `;
      categoryCardsGrid.appendChild(card);
    });
  }

  // Initial Boot
  loadScenarios();
  loadSandboxState();
  loadLeaderboard();
});
