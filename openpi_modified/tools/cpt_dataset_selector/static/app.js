/* global fetch */

const state = {
  taxonomy: null,
  lastRepos: [],
  selected: new Set(),
};

async function loadTaxonomy() {
  const r = await fetch("/api/taxonomy");
  if (!r.ok) throw new Error("taxonomy failed");
  state.taxonomy = await r.json();
}

function getDatasetFilterCheckboxes() {
  return document.querySelectorAll('#dataset-checks input[name="datasets"]');
}

function selectAllDatasetCheckboxes() {
  getDatasetFilterCheckboxes().forEach((cb) => {
    cb.checked = true;
  });
  updateDatasetPickerInput();
}

function collectDatasetsForQuery() {
  const boxes = Array.from(getDatasetFilterCheckboxes());
  const total = boxes.length;
  const checked = boxes.filter((cb) => cb.checked);
  if (total === 0 || checked.length === 0 || checked.length === total) {
    return [];
  }
  return checked.map((cb) => cb.value);
}

function updateDatasetPickerInput() {
  const input = document.getElementById("dataset-picker-input");
  if (!input) return;
  const boxes = Array.from(getDatasetFilterCheckboxes());
  const total = boxes.length;
  const checked = boxes.filter((cb) => cb.checked);
  input.placeholder = "All datasets";
  if (total === 0 || checked.length === 0 || checked.length === total) {
    input.value = "";
    input.removeAttribute("title");
    return;
  }
  if (checked.length === 1) {
    input.value = checked[0].value;
    input.title = "";
    return;
  }
  const first = checked[0].value;
  input.value = `${first} (+${checked.length - 1})`;
  input.title = checked.map((c) => c.value).join(", ");
}

function setDatasetPickerOpen(open) {
  const panel = document.getElementById("dataset-picker-dropdown");
  const input = document.getElementById("dataset-picker-input");
  const field = document.querySelector(".dataset-picker-field");
  if (!panel || !input) return;
  panel.hidden = !open;
  input.setAttribute("aria-expanded", open ? "true" : "false");
  if (field) {
    field.classList.toggle("is-open", open);
  }
}

function wireDatasetPicker() {
  const root = document.getElementById("dataset-picker");
  const input = document.getElementById("dataset-picker-input");
  const panel = document.getElementById("dataset-picker-dropdown");
  const clearBtn = document.getElementById("dataset-clear-sel");
  const checks = document.getElementById("dataset-checks");
  if (!root || !input || !panel || !checks) return;

  setDatasetPickerOpen(false);

  root.addEventListener("click", (e) => {
    e.stopPropagation();
  });
  const flipOpen = () => {
    setDatasetPickerOpen(panel.hidden);
  };
  input.addEventListener("click", flipOpen);
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      selectAllDatasetCheckboxes();
    });
  }
  checks.addEventListener("change", () => {
    updateDatasetPickerInput();
  });
  document.addEventListener("click", () => {
    setDatasetPickerOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setDatasetPickerOpen(false);
    }
  });
}

async function loadDatasets() {
  const r = await fetch("/api/datasets");
  if (!r.ok) return;
  const data = await r.json();
  const holder = document.getElementById("dataset-checks");
  if (!holder) return;
  holder.textContent = "";
  for (const name of data.datasets || []) {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.name = "datasets";
    cb.value = name;
    cb.checked = true;
    cb.autocomplete = "off";
    const span = document.createElement("span");
    span.textContent = name;
    lab.appendChild(cb);
    lab.appendChild(span);
    holder.appendChild(lab);
  }
  updateDatasetPickerInput();
}

/** Reset toolbar controls so browser bfcache / form restore cannot carry over a previous search. */
function resetToolbarDefaults() {
  selectAllDatasetCheckboxes();
  setDatasetPickerOpen(false);
  const mr = document.getElementById("min-match-ratio");
  if (mr) mr.value = "";
}

function renderCheckboxes(containerId, key, items) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  for (const it of items) {
    const id = `${key}-${it.id}`;
    const lab = document.createElement("label");
    lab.innerHTML = `<input type="checkbox" name="${key}" value="${it.id}" id="${id}" /> <span>${escapeHtml(it.label)} <small class="muted">(${it.id})</small></span>`;
    el.appendChild(lab);
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function collectChecked(name) {
  return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map((x) => x.value);
}

function parseMinMatchRatio() {
  const raw = document.getElementById("min-match-ratio").value.trim();
  if (raw === "") return null;
  const n = parseFloat(raw);
  if (Number.isNaN(n)) return null;
  return Math.max(0, Math.min(1, n));
}

async function runQuery() {
  const body = {
    atomic_actions: collectChecked("atomic_actions"),
    object_categories: collectChecked("object_categories"),
    scenes: collectChecked("scenes"),
    include_incomplete_meta: false,
    datasets: collectDatasetsForQuery(),
    min_match_ratio: parseMinMatchRatio(),
  };
  const r = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const t = await r.text();
    alert(`Query failed: ${t}`);
    return;
  }
  const data = await r.json();
  state.lastRepos = data.repos || [];
  state.selected.clear();
  document.getElementById("stat-filtered").innerHTML = `Filtered: <strong>${data.count}</strong> repos`;
  const ep = data.total_episodes;
  const hrs = data.total_duration_hours;
  document.getElementById("stat-episodes").innerHTML =
    `Episodes: <strong>${ep != null ? ep : "—"}</strong>`;
  document.getElementById("stat-hours").innerHTML =
    `Duration: <strong>${hrs != null ? Number(hrs).toFixed(2) : "—"}</strong> h`;
  renderTable();
}

function formatMatchPct(row) {
  const tc = row.task_count;
  const m = row.matched_tasks;
  if (tc == null || tc === 0) return "";
  if (typeof row.match_ratio === "number") {
    return `${Math.round(row.match_ratio * 100)}%`;
  }
  return `${Math.round((100 * m) / tc)}%`;
}

function formatRepoDurationHours(row) {
  const h = row.repo_duration ?? row.duration_hours;
  if (h == null || Number.isNaN(Number(h))) return "";
  return Number(h).toFixed(2);
}

function renderTable() {
  const tbody = document.getElementById("repo-body");
  tbody.innerHTML = "";
  for (const row of state.lastRepos) {
    const tr = document.createElement("tr");
    const rid = row.repo_id;
    const checked = state.selected.has(rid) ? "checked" : "";
    tr.innerHTML = `
      <td><input type="checkbox" class="row-chk" data-repo="${escapeHtml(rid)}" ${checked} /></td>
      <td>${escapeHtml(rid)}</td>
      <td>${escapeHtml(String(row.dataset ?? ""))}</td>
      <td>${escapeHtml(String(row.robot_type ?? ""))}</td>
      <td>${escapeHtml(formatRepoDurationHours(row))}</td>
      <td>${row.task_count ?? ""}</td>
      <td>${row.matched_tasks ?? ""}</td>
      <td>${formatMatchPct(row)}</td>
    `;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll(".row-chk").forEach((cb) => {
    cb.addEventListener("change", () => {
      const id = cb.getAttribute("data-repo");
      if (cb.checked) state.selected.add(id);
      else state.selected.delete(id);
      updateSelectedCount();
    });
  });
  document.getElementById("select-all").checked = false;
  updateSelectedCount();
}

function updateSelectedCount() {
  document.getElementById("stat-selected").innerHTML = `Selected: <strong>${state.selected.size}</strong> repos`;
}

function downloadTextFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType || "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function exportFmt(format) {
  const ids = state.selected.size ? Array.from(state.selected) : state.lastRepos.map((r) => r.repo_id);
  const r = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_ids: ids, format }),
  });
  if (!r.ok) {
    const t = await r.text();
    alert(`Export failed: ${t}`);
    return;
  }
  const data = await r.json();
  const content = data.content ?? "";
  document.getElementById("export-preview").textContent = content;
  const isJson = format === "json";
  downloadTextFile(
    content,
    isJson ? "cpt_repo_ids.json" : "cpt_repo_ids.py",
    isJson ? "application/json;charset=utf-8" : "text/plain;charset=utf-8",
  );
}

function resetAggregateStats() {
  document.getElementById("stat-filtered").innerHTML = `Filtered: <strong>0</strong> repos`;
  document.getElementById("stat-episodes").innerHTML = `Episodes: <strong>—</strong>`;
  document.getElementById("stat-hours").innerHTML = `Duration: <strong>—</strong> h`;
}

function clearResults() {
  state.lastRepos = [];
  state.selected.clear();
  resetAggregateStats();
  const prev = document.getElementById("export-preview");
  if (prev) prev.textContent = "";
  renderTable();
  updateSelectedCount();
}

function clearFilters() {
  document.querySelectorAll('.filters input[type="checkbox"]').forEach((x) => {
    x.checked = false;
  });
  selectAllDatasetCheckboxes();
  setDatasetPickerOpen(false);
  const mr = document.getElementById("min-match-ratio");
  if (mr) mr.value = "";
  clearResults();
}

async function init() {
  resetToolbarDefaults();
  await loadTaxonomy();
  await loadDatasets();
  wireDatasetPicker();
  const t = state.taxonomy;
  renderCheckboxes("atomic-actions", "atomic_actions", t.atomic_actions);
  renderCheckboxes("object-categories", "object_categories", t.object_categories);
  renderCheckboxes("scenes", "scenes", t.scenes);

  document.getElementById("btn-query").addEventListener("click", () => runQuery());
  document.getElementById("btn-clear").addEventListener("click", () => clearFilters());
  document.getElementById("btn-export-json").addEventListener("click", () => exportFmt("json"));
  document.getElementById("btn-export-py").addEventListener("click", () => exportFmt("python_list"));
  document.getElementById("select-all").addEventListener("change", (e) => {
    const on = e.target.checked;
    document.querySelectorAll(".row-chk").forEach((cb) => {
      cb.checked = on;
      const id = cb.getAttribute("data-repo");
      if (on) state.selected.add(id);
      else state.selected.delete(id);
    });
    updateSelectedCount();
  });

  renderTable();
  resetAggregateStats();
  updateSelectedCount();
}

window.addEventListener("pageshow", (ev) => {
  if (!ev.persisted) return;
  resetToolbarDefaults();
  document.querySelectorAll('.filters input[type="checkbox"]').forEach((x) => {
    x.checked = false;
  });
  selectAllDatasetCheckboxes();
  setDatasetPickerOpen(false);
  state.lastRepos = [];
  state.selected.clear();
  const prev = document.getElementById("export-preview");
  if (prev) prev.textContent = "";
  resetAggregateStats();
  renderTable();
  updateSelectedCount();
});

init().catch((e) => {
  console.error(e);
  alert(`Init failed: ${e}`);
});
