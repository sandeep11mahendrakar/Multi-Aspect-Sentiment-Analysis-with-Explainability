"use strict";

const $ = (id) => document.getElementById(id);
const BAND_COLORS = { positive: "var(--green)", neutral: "var(--amber)", negative: "var(--red)" };

/* ---------- tabs ---------- */
function switchTab(which) {
  const single = which === "single";
  $("tab-single").classList.toggle("active", single);
  $("tab-batch").classList.toggle("active", !single);
  $("tab-single").setAttribute("aria-selected", single);
  $("tab-batch").setAttribute("aria-selected", !single);
  $("panel-single").classList.toggle("active", single);
  $("panel-batch").classList.toggle("active", !single);
}
$("tab-single").addEventListener("click", () => switchTab("single"));
$("tab-batch").addEventListener("click", () => switchTab("batch"));

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    if (j.status !== "ok") showBanner(`Models not ready: ${j.detail || j.status}`);
  } catch {
    showBanner("Cannot reach the API — is the server running?");
  }
}

function showBanner(msg) {
  const b = $("banner");
  b.textContent = msg;
  b.classList.remove("hidden");
}

/* ================= SINGLE REVIEW ================= */
document.querySelectorAll(".preset").forEach((btn) => {
  btn.addEventListener("click", () => { $("review-input").value = btn.dataset.preset; });
});

$("predict-btn").addEventListener("click", async () => {
  const text = $("review-input").value.trim();
  if (!text) { alert("Paste a review first."); return; }
  const btn = $("predict-btn");
  btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    renderSingle(await r.json());
  } catch (e) {
    showBanner(`Prediction failed: ${e.message}`);
  } finally {
    btn.disabled = false; btn.textContent = "Analyze review";
  }
});

function setDial(score, band, unsure) {
  const dial = $("dial"), hole = $("dial-score"), marker = $("marker");
  dial.classList.remove("band-positive", "band-neutral", "band-negative", "unsure");
  marker.classList.remove("band-positive", "band-neutral", "band-negative");
  marker.textContent = "";
  marker.style.display = "none";
  if (unsure) {
    dial.classList.add("unsure");
    hole.textContent = "UNSURE";
    return;
  }
  dial.classList.add(`band-${band}`);
  // conic-gradient sweep proportional to score (1..10 -> ~8%..100%)
  const pct = Math.round(8 + ((score - 1) / 9) * 92);
  dial.style.background =
    `conic-gradient(${BAND_COLORS[band]} ${pct}%, var(--hairline) ${pct}% 100%)`;
  hole.textContent = score.toFixed(1);
  const marks = { positive: "+", neutral: "±", negative: "−" };
  marker.textContent = marks[band];
  marker.classList.add(`band-${band}`);
  marker.style.display = "flex";
}

function renderSingle(r) {
  $("result-card").classList.remove("hidden");
  setDial(r.score, r.band, r.unsure);

  const verdict = $("verdict");
  if (r.unsure) {
    verdict.className = "verdict unsure";
    verdict.textContent = "UNSURE / MIXED — model abstained";
  } else {
    verdict.className = `verdict ${r.band}`;
    verdict.textContent =
      `${r.score.toFixed(1)} / 10 · ${cap(r.sentiment)} review (${r.marker})`;
  }

  const confPct = Math.round(r.confidence * 1000) / 10;
  $("conf-fill").style.width = `${confPct}%`;
  $("conf-val").textContent = confPct.toFixed(1) + "%";
  $("conf-fill").style.background = r.unsure ? "#9AA3AF" : BAND_COLORS[r.band];

  const [neg, neu, pos] = r.probs || [0, 0, 0];
  $("probs-row").innerHTML =
    pill(`neg ${(neg * 100).toFixed(1)}%`, "neg") +
    pill(`neu ${(neu * 100).toFixed(1)}%`, "neu") +
    pill(`pos ${(pos * 100).toFixed(1)}%`, "pos");

  const aspects = r.aspects || {};
  const chips = Object.entries(aspects)
    .map(([a, v]) =>
      `<span class="aspect-chip ${v.sentiment}">${a}: ${v.sentiment} (${(v.confidence * 100).toFixed(0)}%)</span>`)
    .join("");
  $("aspects-row").innerHTML =
    chips || `<span class="mini-label">No quality / price / fit aspects detected.</span>`;
}

const cap = (s) => s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
const pill = (txt, cls) => `<span class="prob-pill ${cls}">${txt}</span>`;

/* ================= BATCH CSV ================= */
const dz = $("dropzone");
dz.addEventListener("click", () => $("csv-file").click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", (e) => {
  e.preventDefault(); dz.classList.remove("dragover");
  if (e.dataTransfer.files.length) runBatch(e.dataTransfer.files[0], null);
});
$("csv-file").addEventListener("change", () => {
  if ($("csv-file").files.length) runBatch($("csv-file").files[0], null);
});
$("col-go").addEventListener("click", () => {
  runBatch(window._pendingFile, $("col-select").value);
});

let batchRows = null;

async function runBatch(file, column) {
  window._pendingFile = file;
  batchRows = null;
  $("summary-card").classList.add("hidden");
  hideProgress();
  const fd = new FormData();
  fd.append("file", file);
  if (column) fd.append("column", column);

  try {
    const resp = await fetch("/api/batch", { method: "POST", body: fd });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        handleEvent(JSON.parse(line));
      }
    }
  } catch (e) {
    hideProgress();
    showBanner(`Batch failed: ${e.message}`);
  }
}

function handleEvent(ev) {
  switch (ev.type) {
    case "meta":
      showProgress(ev.total);
      break;
    case "column_select":
      hideProgress();
      window._pendingColumns = ev.columns;
      $("col-confirm").classList.remove("hidden");
      $("col-select").innerHTML =
        ev.columns.map((c) => `<option>${escapeHtml(c)}</option>`).join("");
      break;
    case "progress":
      updateProgress(ev.done, ev.total);
      break;
    case "done":
      hideProgress();
      $("col-confirm").classList.add("hidden");
      batchRows = ev.rows;
      renderSummary(ev.summary);
      renderPreview(ev.rows);
      break;
    case "error":
      hideProgress();
      showBanner(ev.message);
      break;
  }
}

function showProgress(total) {
  $("progress-wrap").classList.remove("hidden");
  updateProgress(0, total);
  $("prog-label").textContent = `Processing 0 / ${total} reviews…`;
}
function updateProgress(done, total) {
  const pct = total ? Math.round((done / total) * 100) : 0;
  $("prog-fill").style.width = `${pct}%`;
  $("prog-pct").textContent = `${pct}%`;
  $("prog-label").textContent = `Processing ${done} / ${total} reviews…`;
}
function hideProgress() { $("progress-wrap").classList.add("hidden"); }

function renderSummary(s) {
  $("summary-card").classList.remove("hidden");
  $("st-total").textContent = s.total.toLocaleString();
  $("st-mean").textContent = s.mean_score == null ? "–" : s.mean_score.toFixed(2);
  $("st-cov").textContent = (s.coverage * 100).toFixed(1) + "%";

  const order = ["positive", "neutral", "negative", "unsure"];
  const max = Math.max(...order.map((b) => s.counts[b] || 0), 1);
  $("dist").innerHTML = order.map((b) => {
    const c = s.counts[b] || 0;
    return `<div class="dist-row">
      <span>${cap(b)}</span>
      <div class="bar"><div class="fill ${b}" style="width:${(c / max) * 100}%"></div></div>
      <span class="count mono">${c}</span>
    </div>`;
  }).join("");
}

function renderPreview(rows) {
  const tbody = $("preview-body");
  tbody.innerHTML = "";
  let sectionDone = false;
  rows.forEach((r, i) => {
    if (r.unsure && !sectionDone) {
      sectionDone = true;
      const tr = document.createElement("tr");
      tr.className = "section-row";
      tr.innerHTML = `<td colspan="8">## UNSURE / MIXED (model not confident enough to rate)</td>`;
      tbody.appendChild(tr);
    }
    const num = r.unsure ? "–" : i + 1;
    const score = r.unsure ? "" : r.score.toFixed(1);
    const bandBadge = r.unsure
      ? `<span class="badge unsure">UNSURE</span>`
      : `<span class="badge ${r.band}">${cap(r.band)} ${r.marker}</span>`;
    const a = r.aspects || {};
    const asp = (k) => a[k] ? cap(a[k]) : "–";
    tbody.insertAdjacentHTML("beforeend", `
      <tr>
        <td class="mono">${num}</td>
        <td class="review-cell">${escapeHtml(shorten(r.text))}</td>
        <td class="mono">${score}</td>
        <td>${bandBadge}</td>
        <td class="mono">${(r.confidence * 100).toFixed(0)}%</td>
        <td>${asp("quality")}</td>
        <td>${asp("price")}</td>
        <td>${asp("fit")}</td>
      </tr>`);
  });
}

function shorten(t, n = 140) { return t.length > n ? t.slice(0, n) + "…" : t; }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- client-side CSV build per spec ---------- */
$("download-btn").addEventListener("click", () => {
  if (!batchRows) return;
  const header = ["review_text", "score", "band", "sentiment", "confidence", "quality", "price", "fit"];
  const lines = [header.join(",")];
  const committed = batchRows.filter((r) => !r.unsure);
  const unsure = batchRows.filter((r) => r.unsure);

  const csvRow = (r) => {
    const a = r.aspects || {};
    const asp = (k) => {
      const v = a[k];
      if (!v) return "";
      return typeof v === "string" ? v : v.sentiment || "";
    };
    return [
      r.text,
      r.score == null ? "" : r.score.toFixed(1),
      r.band ?? "", r.sentiment ?? "",
      r.confidence.toFixed(4),
      asp("quality"), asp("price"), asp("fit"),
    ].map(csvCell).join(",");
  };
  committed.sort((x, y) => y.score - x.score).forEach((r) => lines.push(csvRow(r)));
  if (unsure.length) {
    lines.push('"## UNSURE / MIXED (model not confident enough to rate)"');
    unsure.forEach((r) => lines.push(csvRow(r)));
  }

  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "lilly_review_scores.csv";
  a.click();
  URL.revokeObjectURL(url);
});

function csvCell(v) {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

checkHealth();
