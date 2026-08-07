(() => {
  "use strict";

  const POLL_INTERVAL_MS = 1000;
  const DATA_REFRESH_INTERVAL_MS = 3500;

  const stageLabels = {
    idle: "等待启动",
    starting: "正在准备实验",
    startup: "正在准备实验",
    run: "端到端实验",
    cli: "实验入口收尾",
    input_registration: "输入登记与 SHA-256 去重",
    input: "输入登记与 SHA-256 去重",
    sam3: "SAM3 自动点网格分割",
    dinov3: "DINOv3 候选指纹",
    clustering: "DINOv3 跨图聚类",
    cluster_review: "Qwen3-VL 聚类语义审查",
    memory: "写入 SQLite 对象记忆",
    report: "生成结果摘要",
    completed: "实验已完成",
    failed: "实验已失败",
    interrupted: "实验已中断",
  };

  const statusLabels = {
    idle: "空闲",
    starting: "启动中",
    running: "运行中",
    completed: "已完成",
    completed_with_errors: "完成但有错误",
    passed: "结构通过",
    failed: "失败",
    interrupted: "已中断",
    duplicate: "重复跳过",
    filtered: "已过滤",
    decided: "已判断",
    pending: "待确认",
    new: "新对象",
    existing: "已有对象",
    match: "视觉匹配",
    no_match: "未达视觉阈值",
    ambiguous: "视觉歧义",
    ignored: "已忽略",
    uncertain: "不确定",
  };

  const stagePurposes = {
    input:
      "作用：登记输入文件并按 SHA-256 识别内容副本。页面区分输入文件、本次在所选记忆库中新登记的源图，以及因内容相同而跳过的文件。",
    sam3:
      "作用：SAM3 对每张新图铺设自动点网格，脚本按分数、面积、重叠和包含关系清理候选。这里的候选还不是对象。",
    clustering:
      "作用：DINOv3 为过滤后的候选生成视觉指纹，并把不同图片中的相似候选聚为对象假设。聚类只减少审查单位，不删除任何视角证据。",
    cluster_review:
      "作用：Qwen3-VL 按聚类判断完整物体、背景、零件或不确定项，再结合DINOv3历史匹配形成最终对象归属。",
  };

  const state = {
    inputs: [],
    inputSummary: { total: 0, unique: 0, duplicates: 0 },
    run: { status: "idle" },
    events: [],
    latestSequence: 0,
    report: null,
    serverSummary: null,
    memory: null,
    memoryLibraries: [],
    selectedMemoryId: "default",
    serverCatalogLocked: false,
    catalogMutationInFlight: false,
    catalogRequestSerial: 0,
    runRequestSerial: 0,
    resultRequestSerial: 0,
    memoryRequestSerial: 0,
    viewEpoch: 0,
    selectedStage: "input",
    selectedMemoryView: "objects",
    candidateAssetKinds: new Map(),
    lastIntermediateRenderKey: "",
    pendingIntermediateRender: false,
    stagePointerActive: false,
    lastDataRefresh: 0,
    pollInFlight: false,
    pollTimer: null,
    clockTimer: null,
  };

  const elements = {
    headerStatusDot: document.querySelector("#header-status-dot"),
    headerStatusText: document.querySelector("#header-status-text"),
    runStatusPill: document.querySelector("#run-status-pill"),
    runStage: document.querySelector("#run-stage"),
    runUnit: document.querySelector("#run-unit"),
    runTimer: document.querySelector("#run-timer"),
    progressTrack: document.querySelector("#progress-track"),
    progressBar: document.querySelector("#progress-bar"),
    progressMessage: document.querySelector("#progress-message"),
    progressPercent: document.querySelector("#progress-percent"),
    startRunButton: document.querySelector("#start-run-button"),
    runLockNote: document.querySelector("#run-lock-note"),
    memoryLibrarySelect: document.querySelector("#memory-library-select"),
    newMemoryButton: document.querySelector("#new-memory-button"),
    deleteMemoryButton: document.querySelector("#delete-memory-button"),
    memoryLibraryMeta: document.querySelector("#memory-library-meta"),
    uploadButton: document.querySelector("#upload-button"),
    clearInputsButton: document.querySelector("#clear-inputs-button"),
    fileInput: document.querySelector("#file-input"),
    dropZone: document.querySelector("#drop-zone"),
    inputMetrics: document.querySelector("#input-metrics"),
    inputGrid: document.querySelector("#input-grid"),
    eventFreshness: document.querySelector("#event-freshness"),
    stagePanel: document.querySelector("#stage-panel"),
    memoryOverview: document.querySelector("#memory-overview"),
    memoryContent: document.querySelector("#memory-content"),
    memoryContext: document.querySelector("#memory-context"),
    summaryStatus: document.querySelector("#summary-status"),
    summaryContent: document.querySelector("#summary-content"),
    toastRegion: document.querySelector("#toast-region"),
  };

  const renderedHtml = new WeakMap();

  function updateHtml(element, html) {
    if (renderedHtml.get(element) === html) return false;
    element.innerHTML = html;
    renderedHtml.set(element, html);
    return true;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, Number(value) || 0));
  }

  function array(value) {
    return Array.isArray(value) ? value : [];
  }

  function errorValueText(value) {
    if (value == null || value === false || value === "") return "";
    if (Array.isArray(value)) return value.map(errorValueText).filter(Boolean).join("\n");
    if (typeof value === "object") {
      const message = value.message || value.detail;
      if (message) return value.type ? `${value.type}: ${message}` : String(message);
      if (value.error) return errorValueText(value.error);
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  }

  function processErrorMessage(value) {
    if (value == null || value === false || value === "") return "";
    if (typeof value !== "object") return String(value);
    const message = value.message || value.detail || value.kind;
    if (message) return value.type ? `${value.type}: ${message}` : String(message);
    if (value.error) return processErrorMessage(value.error);
    return value.type ? String(value.type) : "";
  }

  function runExitCode(run) {
    return run?.exit_code ?? run?.process_error?.exit_code ?? null;
  }

  function runLogTail(run) {
    return errorValueText(run?.process_error?.log_tail ?? run?.log_tail);
  }

  function terminalRunMessages(run) {
    const messages = [];
    const stateMessage = processErrorMessage(run?.message);
    const exitCode = runExitCode(run);
    if (stateMessage) messages.push(stateMessage);
    if (exitCode != null) messages.push(`子进程退出码 ${exitCode}`);
    return [...new Set(messages)];
  }

  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatInteger(value) {
    return new Intl.NumberFormat("zh-CN").format(number(value));
  }

  function formatBytes(value) {
    const bytes = number(value);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  }

  function formatSeconds(value) {
    const total = Math.max(0, Math.floor(number(value)));
    const hours = String(Math.floor(total / 3600)).padStart(2, "0");
    const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
    const seconds = String(total % 60).padStart(2, "0");
    return `${hours}:${minutes}:${seconds}`;
  }

  function formatDate(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(date);
  }

  function shortId(value, length = 12) {
    const text = String(value ?? "");
    return text.length > length ? `${text.slice(0, length)}…` : text || "—";
  }

  function isRunningStatus(status) {
    return [
      "starting",
      "running",
      "hashing",
      "input",
      "sam3",
      "dinov3",
      "clustering",
      "cluster_review",
      "memory",
      "report",
    ].includes(String(status));
  }

  function memoryControlsLocked() {
    return (
      state.serverCatalogLocked
      || state.catalogMutationInFlight
      || isRunningStatus(state.run.status)
    );
  }

  function normalizedStage(stage) {
    if (stage === "input_registration") return "input";
    if (stage === "run" || stage === "cli") {
      return number(latestProgressEvent()?.overall_percent) >= 100 ? "report" : "input";
    }
    if (stage === "startup") return "input";
    return stage || "input";
  }

  function statusClass(status) {
    const value = String(status || "neutral").toLowerCase();
    if (["passed", "completed", "new", "existing", "match"].includes(value)) return value;
    if (["failed", "error", "interrupted"].includes(value)) return "failed";
    if (["completed_with_errors", "uncertain", "pending", "ambiguous"].includes(value)) return "warning";
    if (["ignored", "filtered", "duplicate"].includes(value)) return value;
    if (isRunningStatus(value)) return "running";
    return "neutral";
  }

  function statusPill(status, text = null) {
    return `<span class="status-pill ${statusClass(status)}">${escapeHtml(
      text || statusLabels[status] || status || "未知"
    )}</span>`;
  }

  function inputAssetUrl(path) {
    return `/api/input-asset?path=${encodeURIComponent(path || "")}`;
  }

  function memoryAssetUrl(path, memoryId = state.selectedMemoryId) {
    return `/api/memory-asset?memory_id=${encodeURIComponent(memoryId || "default")}&path=${encodeURIComponent(path || "")}`;
  }

  function auditJsonUrl(path, memoryId = state.selectedMemoryId) {
    return `/api/audit-json?memory_id=${encodeURIComponent(memoryId || "default")}&path=${encodeURIComponent(path || "")}`;
  }

  function imageUrl(path, fallbackScope = "memory") {
    if (!path) return "";
    if (String(path).startsWith("/api/")) return path;
    return fallbackScope === "input" ? inputAssetUrl(path) : memoryAssetUrl(path);
  }

  async function apiJson(url, options = {}) {
    const method = String(options.method || "GET").toUpperCase();
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    if (!["GET", "HEAD"].includes(method)) {
      headers["X-Object-Memory-Request"] = "web-ui";
    }
    const response = await fetch(url, {
      ...options,
      method,
      headers,
    });
    const contentType = response.headers.get("content-type") || "";
    let payload = null;
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      payload = { detail: await response.text() };
    }
    if (!response.ok) {
      const message = payload?.detail || payload?.message || `请求失败（${response.status}）`;
      throw new Error(Array.isArray(message) ? JSON.stringify(message) : String(message));
    }
    return payload;
  }

  function toast(message, kind = "info") {
    const node = document.createElement("div");
    node.className = `toast ${kind === "error" ? "error" : ""}`;
    node.textContent = message;
    elements.toastRegion.append(node);
    window.setTimeout(() => node.remove(), 4200);
  }

  function selectedLibrary() {
    return state.memoryLibraries.find((item) => item.id === state.selectedMemoryId) || null;
  }

  function runMemoryLabel() {
    const memoryId = String(state.run.memory_id || "");
    if (!memoryId) return "尚无运行归属";
    const library = state.memoryLibraries.find((item) => item.id === memoryId);
    const label = state.run.memory_label || library?.label || (memoryId === "default" ? "默认记忆库" : memoryId);
    return state.run.memory_deleted_at_utc ? `${label}（已删除）` : label;
  }

  function memoryStatusText(status) {
    return {
      empty: "空白",
      ready: "可继续写入",
      running: "正在写入",
      review_only: "仅供复核",
      unreadable: "读取失败",
    }[status] || "状态未知";
  }

  function memoryIssueText(library) {
    const messages = {
      database_symlink: "数据库文件不是受管的普通文件，已停止读取。",
      database_unreadable: "SQLite 数据库无法安全读取，请在技术环境中检查。",
      invalid_library_root: "记忆库位置不是可管理的目录，已停止读取。",
      incomplete_run: "库内存在未完成或失败的运行，目前仅供复核。",
      missing_asset_directories: "对象资产目录不完整，目前仅供复核。",
      partial_without_database: "目录中只有部分资产，但没有 SQLite 数据库。",
    };
    return messages[library?.issue_code] || "该记忆库当前不能继续写入，请先复核或新建空白库。";
  }

  function renderMemoryLibraryControl() {
    const libraries = state.memoryLibraries;
    const locked = memoryControlsLocked();
    if (!libraries.length) {
      updateHtml(elements.memoryLibrarySelect, "<option>暂无可用记忆库</option>");
      elements.memoryLibrarySelect.disabled = true;
      elements.newMemoryButton.disabled = locked;
      elements.deleteMemoryButton.disabled = true;
      elements.memoryLibraryMeta.textContent = "无法读取记忆库列表。";
      return;
    }
    const current = selectedLibrary();
    updateHtml(
      elements.memoryLibrarySelect,
      libraries
        .map(
          (item) => `<option value="${escapeHtml(item.id)}" ${item.id === state.selectedMemoryId ? "selected" : ""}>${escapeHtml(item.label)} · ${escapeHtml(memoryStatusText(item.status))}</option>`
        )
        .join("")
    );
    elements.memoryLibrarySelect.disabled = locked;
    elements.newMemoryButton.disabled = locked;
    elements.deleteMemoryButton.disabled = locked || !current || current.deletable === false;
    if (!current) {
      elements.memoryLibraryMeta.textContent = "所选记忆库已不存在，正在恢复默认选择。";
      return;
    }
    const counts = current.counts || {};
    const base = current.status === "empty"
      ? "空白库，首次运行会创建 SQLite 与对象资产。"
      : `${formatInteger(counts.active_objects)} 个活跃对象 · ${formatInteger(counts.observations)} 条观测 · ${formatInteger(counts.runs)} 次运行`;
    const hasIssue = Boolean(current.issue_code || current.issue);
    elements.memoryLibraryMeta.textContent = hasIssue
      ? `${memoryStatusText(current.status)}：${memoryIssueText(current)}`
      : `${base} 系统会把本次新观测写回此库。`;
    elements.memoryContext.textContent = `正在查看：${current.label}`;
  }

  async function refreshMemoryLibraries({ quiet = false, preferServerSelection = false } = {}) {
    const requestSerial = ++state.catalogRequestSerial;
    const requestedEpoch = state.viewEpoch;
    try {
      const payload = await apiJson("/api/memories");
      if (requestSerial !== state.catalogRequestSerial || requestedEpoch !== state.viewEpoch) return;
      state.memoryLibraries = array(payload.items);
      state.serverCatalogLocked = Boolean(payload.locked);
      const activeId = payload.active_id;
      const knownIds = new Set(state.memoryLibraries.map((item) => item.id));
      const serverSelection = activeId || payload.selected_id;
      let nextSelection = state.selectedMemoryId;
      if (activeId || preferServerSelection || !knownIds.has(nextSelection)) {
        nextSelection = knownIds.has(serverSelection)
          ? serverSelection
          : state.memoryLibraries[0]?.id || "default";
      }
      const selectionChanged = setSelectedMemoryId(nextSelection, { clear: true });
      renderMemoryLibraryControl();
      renderControlAvailability();
      if (selectionChanged) {
        await Promise.all([
          refreshMemory({ quiet: true }),
          refreshResults({ quiet: true }),
        ]);
      }
    } catch (error) {
      if (!quiet) toast(`无法读取记忆库：${error.message}`, "error");
      renderMemoryLibraryControl();
    }
  }

  async function refreshInputs({ quiet = false } = {}) {
    try {
      const payload = await apiJson("/api/inputs");
      state.inputs = array(payload.items || payload.inputs);
      state.inputSummary = {
        total: number(payload.total ?? payload.count, state.inputs.length),
        unique: number(payload.unique ?? payload.unique_count, state.inputs.length),
        duplicates: number(payload.duplicates ?? payload.duplicate_count, 0),
      };
      renderInputs();
      renderControlAvailability();
    } catch (error) {
      if (!quiet) toast(`无法读取输入：${error.message}`, "error");
      elements.inputGrid.innerHTML = emptyState("无法读取输入图片", error.message, "!");
    }
  }

  async function refreshRun({ initial = false } = {}) {
    if (state.pollInFlight) return;
    const requestSerial = ++state.runRequestSerial;
    const requestedEpoch = state.viewEpoch;
    state.pollInFlight = true;
    try {
      const wasRunning = isRunningStatus(state.run.status);
      const previousWebRunId = state.run.web_run_id;
      const requestedSequence = state.latestSequence;
      const suffix = requestedSequence ? `?after_sequence=${requestedSequence}` : "";
      let payload = await apiJson(`/api/runs/current${suffix}`);
      if (requestSerial !== state.runRequestSerial || requestedEpoch !== state.viewEpoch) return;
      let incomingRun = payload.state || payload.run || payload;
      const incomingWebRunId = incomingRun.web_run_id;
      const runChanged = Boolean(incomingWebRunId && incomingWebRunId !== previousWebRunId);
      if (runChanged && requestedSequence > 0) {
        payload = await apiJson("/api/runs/current?after_sequence=0");
        if (requestSerial !== state.runRequestSerial || requestedEpoch !== state.viewEpoch) return;
        incomingRun = payload.state || payload.run || payload;
      }
      const runMemoryId = incomingRun.memory_id;
      const memoryChanged = Boolean(runMemoryId && runMemoryId !== state.selectedMemoryId);
      if (memoryChanged && isRunningStatus(incomingRun.status)) {
        setSelectedMemoryId(runMemoryId, { clear: true });
        state.serverCatalogLocked = true;
      }
      if (runChanged) {
        Object.assign(state, {
          events: [],
          latestSequence: 0,
          lastDataRefresh: 0,
        });
        const runBelongsToSelection = !runMemoryId || runMemoryId === state.selectedMemoryId;
        if (runBelongsToSelection) {
          Object.assign(state, {
            report: null,
            serverSummary: null,
            memory: null,
            candidateAssetKinds: new Map(),
            lastIntermediateRenderKey: "",
            pendingIntermediateRender: false,
          });
        }
      }
      state.run = runChanged ? { status: "idle", ...incomingRun } : { ...state.run, ...incomingRun };
      if (runChanged) renderSummary();
      const incomingEvents = array(payload.events);
      if (incomingEvents.length) {
        const known = new Set(state.events.map((event) => event.sequence));
        for (const event of incomingEvents) {
          if (!known.has(event.sequence)) state.events.push(event);
          state.latestSequence = Math.max(state.latestSequence, number(event.sequence));
        }
        state.events.sort((a, b) => number(a.sequence) - number(b.sequence));
        if (state.events.length > 1200) state.events = state.events.slice(-1200);
      }
      renderRun();
      renderIntermediates();

      const now = Date.now();
      const isRunning = isRunningStatus(state.run.status);
      const transitionedToTerminal = wasRunning && !isRunning;
      if (transitionedToTerminal || now - state.lastDataRefresh >= DATA_REFRESH_INTERVAL_MS) {
        state.lastDataRefresh = now;
        await Promise.all([
          refreshResults({ quiet: true }),
          refreshMemory({ quiet: true }),
          refreshMemoryLibraries({ quiet: true }),
        ]);
      }
      if (transitionedToTerminal && !initial) await refreshInputs({ quiet: true });
    } catch (error) {
      elements.headerStatusDot.className = "status-dot failed";
      elements.headerStatusText.textContent = "实验服务连接失败";
      if (initial) toast(`无法连接实验服务：${error.message}`, "error");
    } finally {
      state.pollInFlight = false;
    }
  }

  async function refreshResults({ quiet = false } = {}) {
    const requestedMemoryId = state.selectedMemoryId;
    const requestedEpoch = state.viewEpoch;
    const requestSerial = ++state.resultRequestSerial;
    try {
      const payload = await apiJson(`/api/results?memory_id=${encodeURIComponent(requestedMemoryId)}`);
      if (
        requestSerial !== state.resultRequestSerial
        || state.viewEpoch !== requestedEpoch
        || state.selectedMemoryId !== requestedMemoryId
        || (payload.memory_id && payload.memory_id !== requestedMemoryId)
      ) return;
      if (payload.state?.web_run_id && typeof payload.state === "object") {
        const resultWebRunId = payload.state.web_run_id;
        const resultRunChanged = Boolean(
          resultWebRunId && resultWebRunId !== state.run.web_run_id
        );
        if (resultRunChanged) {
          Object.assign(state, {
            events: [],
            latestSequence: 0,
            report: null,
            serverSummary: null,
            candidateAssetKinds: new Map(),
            lastIntermediateRenderKey: "",
            pendingIntermediateRender: false,
            lastDataRefresh: 0,
          });
          state.run = { status: "idle", ...payload.state };
          renderRun();
        } else {
          state.run = { ...state.run, ...payload.state };
        }
      }
      const incoming = payload.report || (payload.schema_version ? payload : null);
      if (payload.available === false) {
        state.report = null;
        state.serverSummary = null;
      } else if (incoming) {
        state.report = incoming;
        state.serverSummary = payload.summary || null;
      }
      renderSummary();
      renderIntermediates();
    } catch (error) {
      if (!quiet) toast(`无法读取结果摘要：${error.message}`, "error");
    }
  }

  async function refreshMemory({ quiet = false } = {}) {
    const requestedMemoryId = state.selectedMemoryId;
    const requestedEpoch = state.viewEpoch;
    const requestSerial = ++state.memoryRequestSerial;
    try {
      const payload = await apiJson(`/api/memory?memory_id=${encodeURIComponent(requestedMemoryId)}`);
      if (
        requestSerial !== state.memoryRequestSerial
        || state.viewEpoch !== requestedEpoch
        || state.selectedMemoryId !== requestedMemoryId
        || (payload.memory_id && payload.memory_id !== requestedMemoryId)
      ) return;
      state.memory = payload;
      renderMemory();
    } catch (error) {
      if (!quiet) toast(`无法读取对象记忆：${error.message}`, "error");
    }
  }

  function renderInputs() {
    const summary = state.inputSummary;
    elements.inputMetrics.innerHTML = `
      <article><span>输入图片</span><strong>${formatInteger(summary.total)}</strong></article>
      <article><span>不同内容</span><strong>${formatInteger(summary.unique)}</strong></article>
      <article><span>内容副本</span><strong>${formatInteger(summary.duplicates)}</strong></article>
      <article><span>支持格式</span><strong>JPG · PNG · WEBP</strong></article>
    `;

    if (!state.inputs.length) {
      elements.inputGrid.innerHTML = emptyState(
        "还没有实验图片",
        "添加至少一张场景图片后即可运行实验。",
        "＋"
      );
      return;
    }

    const locked = memoryControlsLocked();
    elements.inputGrid.innerHTML = state.inputs
      .map((item) => {
        const path = item.path || item.relative_path || item.name;
        const duplicate = Boolean(item.is_duplicate || item.duplicate_of);
        const dimensions = item.width && item.height ? `${item.width} × ${item.height}` : "尺寸未知";
        return `
          <article class="input-card">
            <div class="input-preview">
              <img loading="lazy" src="${escapeHtml(item.url || inputAssetUrl(path))}" alt="输入图片 ${escapeHtml(
          item.name || path
        )}" />
              ${statusPill(duplicate ? "duplicate" : "info", duplicate ? "重复内容" : "唯一内容")}
            </div>
            <div class="input-card-body">
              <div class="input-card-title">
                <div>
                  <strong title="${escapeHtml(path)}">${escapeHtml(item.name || path)}</strong>
                  <small>${escapeHtml(dimensions)} · ${formatBytes(item.size_bytes || item.size)}</small>
                </div>
                <button
                  class="quiet-button danger"
                  type="button"
                  data-delete-input="${escapeHtml(path)}"
                  ${locked ? "disabled" : ""}
                  aria-label="删除 ${escapeHtml(item.name || path)}"
                >删除</button>
              </div>
              <span class="mono-line" title="${escapeHtml(item.sha256 || "")}">SHA ${escapeHtml(
          shortId(item.sha256, 14)
        )}</span>
            </div>
          </article>
        `;
      })
      .join("");
  }

  function latestProgressEvent() {
    return state.events.length ? state.events[state.events.length - 1] : null;
  }

  function localizedProgressMessage(event, fallback) {
    if (!event) return fallback;
    const current = event.current ?? 0;
    const total = event.total ?? 0;
    const filename = event.data?.filename || String(event.data?.input_path || "").replaceAll("\\", "/").split("/").pop();
    const messages = {
      run_started: "端到端实验已启动",
      input_registration_started: "正在读取图片并计算 SHA-256",
      input_registered: `已登记 ${filename || `第 ${current} 张图片`}（${current}/${total}）`,
      input_registration_completed: "输入登记与内容去重完成",
      sam3_batch_started: "SAM3 正在对全部新图片生成点网格候选",
      sam3_image_completed: `SAM3 已完成第 ${current}/${total} 张图`,
      dinov3_batch_started: "DINOv3 正在提取全部候选的视觉指纹",
      dinov3_candidate_completed: `DINOv3 已完成 ${current}/${total} 个候选`,
      dinov3_clustering_completed: "DINOv3 跨图聚类已完成",
      cluster_review_started: "Qwen 正在按聚类批量进行语义审查",
      cluster_review_batch_completed: `Qwen 已完成第 ${current}/${total} 个聚类批次`,
      report_started: "正在汇总实验报告",
      report_completed: "运行报告已写入",
      run_completed: "端到端流程已结束",
      cli_completed: "正式实验入口已完成收尾",
      cli_failed: "实验失败，原始错误已保留",
      model_loading: `正在加载${event.stage === "sam3" ? " SAM3" : " Qwen"}模型`,
      model_loaded: `${event.stage === "sam3" ? "SAM3" : "Qwen"} 模型已加载`,
    };
    return messages[event.event] || fallback;
  }

  function latestFlowStage(event) {
    const flowStages = ["input", "sam3", "clustering", "cluster_review", "memory", "report"];
    const asFlowStage = (stage) => {
      const candidate = stage === "input_registration"
        ? "input"
        : stage === "dinov3"
          ? "clustering"
          : stage;
      return flowStages.includes(candidate) ? candidate : null;
    };
    for (const stage of [event?.stage, state.run.stage]) {
      const matched = asFlowStage(stage);
      if (matched) return matched;
    }
    for (let index = state.events.length - 1; index >= 0; index -= 1) {
      const matched = asFlowStage(state.events[index]?.stage);
      if (matched) return matched;
    }
    return normalizedStage(event?.stage || state.run.stage || "input");
  }

  function renderRun() {
    const event = latestProgressEvent();
    const runStatus = state.run.status || event?.status || "idle";
    const running = isRunningStatus(runStatus);
    const terminalFailure = ["failed", "interrupted"].includes(runStatus);
    const rawEventStage = event?.stage || state.run.stage || (isRunningStatus(runStatus) ? "starting" : "idle");
    const eventStage = normalizedStage(rawEventStage);
    const flowStage = latestFlowStage(event);
    const progress = clamp(event?.overall_percent ?? state.run.overall_percent ?? 0, 0, 100);
    const current = event?.current ?? state.run.current;
    const total = event?.total ?? state.run.total;
    const terminalMessages = terminalRunMessages(state.run);
    const rawMessage = terminalFailure
      ? terminalMessages.join(" · ") || "实验进程未正常完成"
      : event?.message || state.run.message || (runStatus === "idle" ? "尚未开始实验" : "等待下一条实验事件");
    const message = terminalFailure ? rawMessage : localizedProgressMessage(event, rawMessage);

    elements.runStatusPill.className = `status-pill ${statusClass(runStatus)}`;
    elements.runStatusPill.textContent = statusLabels[runStatus] || runStatus;
    const stageText = stageLabels[rawEventStage] || stageLabels[eventStage] || rawEventStage;
    elements.runStage.textContent = terminalFailure
      ? `${stageText} · ${statusLabels[runStatus] || runStatus}`
      : stageText || stageLabels[runStatus];
    const runDetail = terminalFailure
      ? runExitCode(state.run) != null
        ? `子进程退出码 ${runExitCode(state.run)}`
        : statusLabels[runStatus] || runStatus
      : current != null && total != null
        ? `已完成 ${current} / ${total}`
        : message;
    const ownership = state.run.memory_id
      ? `${running ? "当前" : "最近"}运行所属：${runMemoryLabel()}`
      : "尚无运行归属";
    elements.runUnit.textContent = `${ownership} · ${runDetail}`;
    elements.runUnit.title = `${ownership} · ${runDetail}`;
    elements.progressBar.style.width = `${progress}%`;
    elements.progressTrack.setAttribute("aria-valuenow", String(Math.round(progress)));
    elements.progressMessage.textContent = message;
    elements.progressPercent.textContent = `${Math.round(progress)}%`;

    elements.headerStatusDot.className = `status-dot ${running ? "running" : statusClass(runStatus) === "failed" ? "failed" : "ready"}`;
    elements.headerStatusText.textContent = running
      ? `实验运行中 · ${stageLabels[eventStage] || eventStage}`
      : runStatus === "failed" || runStatus === "interrupted"
        ? `最近实验${statusLabels[runStatus]}`
        : "实验服务已连接";

    if (event?.timestamp_utc) {
      elements.eventFreshness.textContent = `最新事件 ${formatDate(event.timestamp_utc)} · #${event.sequence}`;
    } else {
      elements.eventFreshness.textContent = state.report ? "已载入最新运行报告" : "等待实验事件";
    }

    renderFlow(flowStage, progress, terminalFailure);
    renderControlAvailability();
    updateClock();
  }

  function renderFlow(activeStage, progress, failed = false) {
    const order = ["input", "sam3", "clustering", "cluster_review", "memory", "report"];
    let activeIndex = order.indexOf(activeStage);
    if (activeIndex < 0 && progress >= 100) activeIndex = order.length - 1;
    document.querySelectorAll("[data-flow-stage]").forEach((node) => {
      const index = order.indexOf(node.dataset.flowStage);
      node.classList.toggle("active", !failed && index === activeIndex && progress < 100);
      node.classList.toggle("failed", failed && index === activeIndex);
      node.classList.toggle("done", progress >= 100 || (activeIndex >= 0 && index < activeIndex));
    });
  }

  function renderControlAvailability() {
    const locked = memoryControlsLocked();
    const experimentRunning = state.serverCatalogLocked || isRunningStatus(state.run.status);
    const hasInputs = state.inputSummary.total > 0;
    const library = selectedLibrary();
    const canUseLibrary = Boolean(library?.continuable);
    elements.startRunButton.disabled = locked || !hasInputs || !canUseLibrary;
    elements.uploadButton.disabled = locked;
    elements.clearInputsButton.disabled = locked || !hasInputs;
    elements.memoryLibrarySelect.disabled = locked || !state.memoryLibraries.length;
    elements.newMemoryButton.disabled = locked;
    elements.deleteMemoryButton.disabled = locked || !library || library.deletable === false;
    elements.inputGrid.querySelectorAll("[data-delete-input]").forEach((button) => {
      button.disabled = locked;
    });
    elements.dropZone.classList.toggle("locked", locked);
    elements.dropZone.setAttribute("aria-disabled", String(locked));
    elements.runLockNote.textContent = state.catalogMutationInFlight
      ? "正在更新实验输入或记忆库，请稍候。"
      : experimentRunning
        ? "实验正在运行，输入与记忆库管理已锁定。"
        : !canUseLibrary
          ? "所选记忆库仅供复核；请新建空白库或选择可继续写入的库。"
          : hasInputs
            ? "启动后会锁定输入；同一时间只运行一个实验。"
            : "请先上传至少一张输入图片。";
  }

  function runElapsedSeconds() {
    const event = latestProgressEvent();
    const startedAt = state.run.started_at_utc || state.run.started_at;
    if (isRunningStatus(state.run.status) && startedAt) {
      const started = new Date(startedAt).getTime();
      if (!Number.isNaN(started)) return Math.max(0, (Date.now() - started) / 1000);
    }
    return number(state.run.elapsed_seconds ?? event?.elapsed_seconds, 0);
  }

  function updateClock() {
    elements.runTimer.textContent = formatSeconds(runElapsedSeconds());
  }

  function imageStates() {
    const byKey = new Map();
    const aliases = new Map();
    const mergeImage = (image) => {
      const inputPath = String(image?.input_path || "").trim().replaceAll("\\", "/");
      const sourceId = String(image?.source_id || "").trim();
      const keys = [
        inputPath ? `input:${inputPath}` : "",
        sourceId ? `source:${sourceId}` : "",
      ].filter(Boolean);
      if (!keys.length) return;
      const existingKeys = [...new Set(keys.map((item) => aliases.get(item)).filter(Boolean))];
      const key = existingKeys[0] || keys[0];
      let previous = {};
      for (const existingKey of existingKeys) {
        previous = { ...previous, ...(byKey.get(existingKey) || {}) };
        if (existingKey !== key) byKey.delete(existingKey);
      }
      if (existingKeys.length > 1) {
        for (const [alias, existingKey] of aliases.entries()) {
          if (existingKeys.includes(existingKey)) aliases.set(alias, key);
        }
      }
      const merged = {
        ...previous,
        ...image,
        status: image.status || image.work_status || previous.status,
      };
      for (const field of ["scene_guidance", "sam", "candidate_reasoning"]) {
        if (previous[field] && image[field]) {
          merged[field] = { ...previous[field], ...image[field] };
        }
      }
      byKey.set(key, merged);
      keys.forEach((item) => aliases.set(item, key));
    };
    const visibleEvents = !state.run.memory_id || state.run.memory_id === state.selectedMemoryId
      ? state.events
      : [];
    for (const event of visibleEvents) {
      const data = event.data || {};
      const direct = data.input_path || data.source_id
        ? [{ ...data, status: data.work_status || data.status }]
        : [];
      const images = array(data.images).concat(data.image ? [data.image] : [], direct);
      for (const image of images) mergeImage(image);
    }
    // The final report is authoritative for terminal status and errors. Merge it
    // last while preserving event-only proposal details accumulated above.
    for (const image of array(state.report?.images)) mergeImage(image);
    return [...byKey.values()];
  }

  function hasActiveStageSelection() {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !selection.rangeCount) return false;
    const range = selection.getRangeAt(0);
    return Boolean(
      elements.stagePanel.contains(selection.anchorNode) ||
        elements.stagePanel.contains(selection.focusNode) ||
        elements.stagePanel.contains(range.commonAncestorContainer)
    );
  }

  function stageTableScrollKey(node, index) {
    return node.dataset.scrollKey || `${state.selectedStage}:table:${index}`;
  }

  function captureStageTableScrollPositions() {
    const positions = new Map();
    elements.stagePanel.querySelectorAll(".table-wrap").forEach((node, index) => {
      positions.set(stageTableScrollKey(node, index), node.scrollLeft);
    });
    return positions;
  }

  function restoreStageTableScrollPositions(positions) {
    elements.stagePanel.querySelectorAll(".table-wrap").forEach((node, index) => {
      const scrollLeft = positions.get(stageTableScrollKey(node, index));
      if (scrollLeft != null) node.scrollLeft = scrollLeft;
    });
  }

  function hasSamEvidence(image) {
    const sam = image?.sam;
    if (!sam) return false;
    return Boolean(
      number(sam.kept) ||
        number(sam.filtered) ||
        number(sam.raw_candidates) ||
        number(sam.grid_points) ||
        number(sam.above_confidence_threshold_candidates)
    );
  }

  function clusterStates() {
    const merged = new Map();
    const merge = (cluster) => {
      const id = cluster?.cluster_id;
      if (!id) return;
      merged.set(id, { ...(merged.get(id) || {}), ...cluster });
    };
    const visibleEvents = !state.run.memory_id || state.run.memory_id === state.selectedMemoryId
      ? state.events
      : [];
    for (const event of visibleEvents) {
      array(event.data?.clusters).forEach(merge);
    }
    array(state.report?.clusters).forEach(merge);
    return [...merged.values()].sort(
      (first, second) => number(second.member_count) - number(first.member_count) || String(first.cluster_id).localeCompare(String(second.cluster_id))
    );
  }

  function intermediateRenderKey() {
    const report = state.report || {};
    return JSON.stringify([
      state.selectedStage,
      state.selectedMemoryId,
      state.run.web_run_id || "",
      state.latestSequence,
      report.run?.run_id || "",
      report.status || "",
      report.generated_at_utc || report.run?.finished_at_utc || "",
      array(report.images).length,
      array(report.clusters).length,
      state.inputSummary.total,
      state.inputSummary.unique,
    ]);
  }

  function renderIntermediates({ force = false } = {}) {
    const renderKey = intermediateRenderKey();
    if (!force && renderKey === state.lastIntermediateRenderKey) return;
    if (state.stagePointerActive || (!force && hasActiveStageSelection())) {
      state.pendingIntermediateRender = true;
      return;
    }
    const tableScrollPositions = captureStageTableScrollPositions();
    state.pendingIntermediateRender = false;
    const images = imageStates();
    const clusters = clusterStates();
    const uniqueImages = images.filter((image) => !image.duplicate && image.source_id);
    const samImages = images.filter(hasSamEvidence);
    const keptCount = images.reduce((sum, image) => sum + number(image.sam?.kept, 0), 0);
    const reviewedClusters = clusters.filter((cluster) => cluster.qwen_review || cluster.final_decision).length;
    document.querySelector("#tab-count-input").textContent = images.length
      ? `${images.length} 个输入文件 · ${uniqueImages.length} 张本次登记源图`
      : "等待结果";
    document.querySelector("#tab-count-scene").textContent = clusters.length
      ? `${clusters.length} 个跨图候选聚类`
      : "等待结果";
    document.querySelector("#tab-count-sam").textContent = samImages.length
      ? `${samImages.length} 张 SAM3 结果图片 · ${keptCount} 个保留候选区域`
      : "等待结果";
    document.querySelector("#tab-count-reasoning").textContent = reviewedClusters
      ? `${reviewedClusters} 个聚类已有语义结果`
      : "等待结果";

    if (state.selectedStage === "input") renderDedupStage(images);
    if (state.selectedStage === "sam3") renderSamStage(images);
    if (state.selectedStage === "clustering") renderClusterStage(clusters);
    if (state.selectedStage === "cluster_review") renderReasoningStage(clusters);
    restoreStageTableScrollPositions(tableScrollPositions);
    state.lastIntermediateRenderKey = renderKey;
  }

  function renderDedupStage(images) {
    if (!images.length) {
      updateHtml(
        elements.stagePanel,
        `${stagePanelHeader(
          "STEP 01 · INPUT PREPARATION",
          "输入整理",
          stagePurposes.input,
          []
        )}${emptyState("等待输入整理结果", "实验开始后，这里会显示哪些图片进入推理、哪些相同内容被跳过。", "◎")}`
      );
      return;
    }
    const duplicates = images.filter((image) => image.duplicate);
    const uniqueCount = images.filter((image) => !image.duplicate && image.source_id).length;
    updateHtml(elements.stagePanel, `
      ${stagePanelHeader(
        "STEP 01 · INPUT PREPARATION",
        "输入整理",
        stagePurposes.input,
        [
          ["输入文件", images.length],
          ["本次登记源图", uniqueCount],
          ["重复文件", duplicates.length],
        ]
      )}
      <div class="table-wrap" data-scroll-key="input-lineage">
        <table class="lineage-table">
          <thead><tr><th>输入图片</th><th>内容指纹</th><th>处理图片 ID</th><th>处理状态</th><th>源图资产</th><th>错误</th></tr></thead>
          <tbody>
            ${images
              .map(
                (image) => `
                  <tr>
                    <td class="mono">${escapeHtml(fileName(image.input_path) || "—")}</td>
                    <td class="mono" title="${escapeHtml(image.sha256 || "")}">${escapeHtml(shortId(image.sha256, 16))}</td>
                    <td class="mono" title="${escapeHtml(image.source_id || "")}">${escapeHtml(shortId(image.source_id, 17))}</td>
                    <td>${statusPill(image.duplicate ? "duplicate" : image.status, image.duplicate ? "重复跳过" : statusLabels[image.status] || image.status)}</td>
                    <td>${sourcePreview(image) ? `<a class="quiet-button" href="${escapeHtml(sourcePreview(image))}" target="_blank" rel="noopener">查看源图</a>` : "—"}</td>
                    <td>${escapeHtml(errorValueText(image.error) || "—")}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `);
  }

  function sourcePreview(image) {
    if (image.stored_source) return memoryAssetUrl(image.stored_source);
    const inputName = String(image.input_path || "").replaceAll("\\", "/").split("/").pop();
    const matched = state.inputs.find((item) => (item.name || item.path) === inputName);
    return matched ? matched.url || inputAssetUrl(matched.path || matched.name) : "";
  }

  function fileName(path) {
    return String(path || "").replaceAll("\\", "/").split("/").pop();
  }

  function renderClusterStage(clusters) {
    if (!clusters.length) {
      updateHtml(
        elements.stagePanel,
        `${stagePanelHeader(
          "STEP 03 · VISUAL CLUSTERING",
          "DINOv3 跨图片聚类",
          stagePurposes.clustering,
          []
        )}${emptyState("等待视觉聚类结果", "DINOv3 完成全部候选指纹后，这里会显示跨图片的对象假设。", "03")}`
      );
      return;
    }
    const members = clusters.reduce((sum, cluster) => sum + number(cluster.member_count), 0);
    const multiView = clusters.filter((cluster) => number(cluster.source_count) > 1).length;
    updateHtml(elements.stagePanel, `
      ${stagePanelHeader(
        "STEP 03 · VISUAL CLUSTERING",
        "DINOv3 跨图片聚类",
        stagePurposes.clustering,
        [
          ["候选聚类", clusters.length],
          ["聚类成员", members],
          ["多视角聚类", multiView],
        ]
      )}
      <div class="candidate-grid">
        ${clusters.map((cluster) => {
          const similarity = cluster.global_similarity || {};
          return `
            <article class="candidate-card">
              <div class="candidate-asset">${previewImage(imageUrl(cluster.contact_sheet), `聚类 ${cluster.cluster_id}`)}</div>
              <div class="candidate-body">
                <div class="card-title-row">
                  <strong class="mono">${escapeHtml(cluster.cluster_id)}</strong>
                  ${statusPill(number(cluster.source_count) > 1 ? "match" : "neutral", `${formatInteger(cluster.member_count)} 个候选 / ${formatInteger(cluster.source_count)} 张图`)}
                </div>
                <p>同组候选来自不同源图；同一张图中的相似候选不会自动合并。</p>
                <p class="meta-line">CLS 相似度 · min ${number(similarity.min, 1).toFixed(3)} · mean ${number(similarity.mean, 1).toFixed(3)} · max ${number(similarity.max, 1).toFixed(3)}</p>
                <span class="mono-line">members ${array(cluster.member_proposal_ids).map((id) => shortId(id, 13)).join(" · ")}</span>
              </div>
            </article>
          `;
        }).join("")}
      </div>
    `);
  }

  function candidatesForImage(image) {
    const direct = array(
      image.kept_proposals || image.proposals || image.sam?.proposals
    ).concat(array(image.filtered_proposals));
    const fromDecisions = array(image.decisions)
      .map((decision) => {
        if (!decision.candidate) return null;
        return { id: decision.proposal_id, ...decision.candidate, decision };
      })
      .filter(Boolean);
    const merged = new Map();
    for (const candidate of [...direct, ...fromDecisions]) {
      const key = candidate.id || candidate.proposal_id || candidate.raw_candidate_id;
      if (key) merged.set(key, { ...(merged.get(key) || {}), ...candidate });
    }
    return [...merged.values()];
  }

  function renderSamStage(images) {
    const relevant = images.filter(hasSamEvidence);
    if (!relevant.length) {
      updateHtml(
        elements.stagePanel,
        `${stagePanelHeader(
          "STEP 03 · SEGMENTATION",
          "SAM3 自动点网格分割",
          stagePurposes.sam3,
          []
        )}${emptyState("等待对象分割结果", "SAM3 完成自动点网格分割后，这里会显示候选区域与脚本筛选结果。", "02")}`
      );
      return;
    }
    const kept = relevant.reduce((sum, image) => sum + number(image.sam?.kept), 0);
    const filtered = relevant.reduce((sum, image) => sum + number(image.sam?.filtered), 0);
    const rawCount = relevant.reduce((sum, image) => sum + number(image.sam?.raw_candidates ?? image.sam?.grid_points), 0);
    updateHtml(elements.stagePanel, `
      ${stagePanelHeader(
        "STEP 02 · AUTOMATIC SEGMENTATION",
        "SAM3 自动点网格分割",
        stagePurposes.sam3,
        [
          ["SAM3 结果图片", relevant.length],
          ["点网格原始候选", rawCount],
          ["保留候选区域", kept],
          ["过滤候选区域", filtered],
        ]
      )}
      <div class="evidence-list">
        ${relevant
          .map((image) => {
            const sam = image.sam || {};
            const candidates = candidatesForImage(image);
            return `
              <article class="evidence-card" style="grid-template-columns: 1fr">
                <div class="evidence-body">
                  <div class="card-title-row">
                    <div>
                      <h4>${escapeHtml(image.source_id || image.input_path || "未知源图")}</h4>
                      <span class="mono-line">SAM3 ${escapeHtml(sam.inference_seconds ?? "—")}s · ${number(sam.raw_candidates ?? sam.grid_points)} 个点网格原始候选</span>
                    </div>
                    <div class="chip-list" style="margin-top:0">
                      ${statusPill("info", `保留 ${number(sam.kept)} 个候选区域`)}
                      ${number(sam.filtered) ? statusPill("filtered", `过滤 ${number(sam.filtered)} 个候选区域`) : ""}
                    </div>
                  </div>
                  <div class="chip-list"><span class="chip prompt">候选来源 · <b>automatic_point_grid</b></span></div>
                  ${image.error ? `<p class="meta-line">阶段错误 · ${escapeHtml(image.error)}</p>` : ""}
                  ${candidates.length ? `<div class="candidate-grid">${candidates.map(candidateCard).join("")}</div>` : ""}
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    `);
    restoreCandidateAssetViews();
  }

  function candidateCard(candidate) {
    const id = candidate.id || candidate.proposal_id || candidate.raw_candidate_id || "candidate";
    const crop = candidate.crop || candidate.crop_path;
    const mask = candidate.mask || candidate.mask_path;
    const overlay = candidate.overlay || candidate.overlay_path;
    const assets = [
      ["overlay", overlay, "定位"],
      ["crop", crop, "裁剪"],
      ["mask", mask, "掩码"],
    ].filter(([, path]) => path);
    const initialAsset = assets[0]?.[1];
    const initialKind = assets[0]?.[0];
    return `
      <article class="candidate-card" data-candidate-id="${escapeHtml(id)}">
        <div class="candidate-asset">
          ${previewImage(imageUrl(initialAsset), `候选 ${id}`)}
          ${
            assets.length > 1
              ? `<div class="candidate-asset-switch">${assets
                  .map(
                    ([kind, path, label]) => `<button class="asset-toggle ${kind === initialKind ? "active" : ""}" type="button" data-asset-url="${escapeHtml(
                      imageUrl(path)
                    )}" data-asset-kind="${kind}">${label}</button>`
                  )
                  .join("")}</div>`
              : ""
          }
        </div>
        <div class="candidate-body">
          <div class="card-title-row">
            <strong class="mono">${escapeHtml(shortId(id, 20))}</strong>
            ${candidate.status ? statusPill(candidate.status) : ""}
          </div>
          <p>候选来源 · <b>${escapeHtml(candidate.candidate_source || candidate.prompt || "automatic_point_grid")}</b></p>
          <p class="meta-line">SAM3 点提示质量分 · ${number(candidate.score).toFixed(3)}</p>
          ${candidate.filter_reason ? `<p class="meta-line">过滤原因 · ${escapeHtml(candidate.filter_reason)}</p>` : ""}
          ${candidate.error ? `<p class="meta-line">错误 · ${escapeHtml(candidate.error)}</p>` : ""}
          <span class="mono-line">bbox ${escapeHtml(bboxText(candidate.bbox))} · area ${number(candidate.mask_area_ratio) ? `${(number(candidate.mask_area_ratio) * 100).toFixed(2)}%` : "—"}</span>
        </div>
      </article>
    `;
  }

  function restoreCandidateAssetViews() {
    elements.stagePanel.querySelectorAll("[data-candidate-id]").forEach((card) => {
      const selectedKind = state.candidateAssetKinds.get(card.dataset.candidateId);
      if (!selectedKind) return;
      const buttons = [...card.querySelectorAll("[data-asset-kind]")];
      const selectedButton = buttons.find((button) => button.dataset.assetKind === selectedKind);
      if (!selectedButton) return;
      const image = card.querySelector(".candidate-asset > img");
      if (image) image.src = selectedButton.dataset.assetUrl;
      buttons.forEach((button) => button.classList.toggle("active", button === selectedButton));
    });
  }

  function bboxText(bbox) {
    if (!bbox) return "—";
    if (Array.isArray(bbox)) return bbox.map((value) => number(value).toFixed(0)).join(", ");
    return [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
      .map((value) => number(value).toFixed(0))
      .join(", ");
  }

  function renderReasoningStage(clusters) {
    const relevant = clusters.filter((cluster) => cluster.qwen_review || cluster.final_decision || cluster.error);
    if (!relevant.length) {
      updateHtml(
        elements.stagePanel,
        `${stagePanelHeader(
          "STEP 04 · SEMANTIC REVIEW",
          "Qwen3-VL 聚类语义审查",
          stagePurposes.cluster_review,
          []
        )}${emptyState("等待语义审查结果", "Qwen完成聚类批次后，这里会解释候选组如何形成或未形成对象。", "04")}`
      );
      return;
    }
    updateHtml(elements.stagePanel, `
      ${stagePanelHeader(
        "STEP 04 · SEMANTIC REVIEW",
        "聚类证据 → Qwen判断 → 最终对象",
        stagePurposes.cluster_review,
        [
          ["已审查聚类", relevant.length],
          ["形成对象", relevant.filter((cluster) => ["new", "existing"].includes(cluster.final_decision)).length],
          ["过滤 / 待定", relevant.filter((cluster) => ["ignored", "uncertain"].includes(cluster.final_decision)).length],
        ]
      )}
      <div class="candidate-grid">
        ${relevant
          .map((cluster) => {
            const review = cluster.qwen_review || {};
            const visual = cluster.historical_visual_evidence || {};
            const similarity = cluster.global_similarity || {};
            const summary = review.object_summary || {};
            return `
              <article class="candidate-card">
                <div class="candidate-asset">${previewImage(
                  imageUrl(cluster.contact_sheet),
                  `聚类 ${cluster.cluster_id}`
                )}</div>
                <div class="candidate-body">
                  <div class="card-title-row">
                    <strong class="mono">${escapeHtml(cluster.cluster_id)}</strong>
                    <div class="chip-list" style="margin-top:0">
                      ${cluster.final_decision ? statusPill(cluster.final_decision) : statusPill("pending")}
                      ${cluster.raw_response ? `<a class="quiet-button" href="${escapeHtml(auditJsonUrl(cluster.raw_response))}" target="_blank" rel="noopener">原始响应</a>` : ""}
                    </div>
                  </div>
                  <div class="reasoning-steps">
                    <section class="reasoning-step">
                      <div class="reasoning-step-heading">
                        <span>1 · DINOv3 聚类依据</span>
                        ${statusPill(number(cluster.source_count) > 1 ? "match" : "neutral", `${formatInteger(cluster.member_count)} 个候选`)}
                      </div>
                      <p>跨图CLS相似度 min ${number(similarity.min, 1).toFixed(3)} · mean ${number(similarity.mean, 1).toFixed(3)} · max ${number(similarity.max, 1).toFixed(3)}</p>
                      <span class="mono-line">sources ${formatInteger(cluster.source_count)} · representatives ${formatInteger(array(cluster.representative_proposal_ids).length)}</span>
                    </section>
                    <section class="reasoning-step">
                      <div class="reasoning-step-heading">
                        <span>2 · 历史对象匹配</span>
                        ${visual.result ? statusPill(visual.result) : statusPill(cluster.final_decision || "pending")}
                      </div>
                      <p>global ${visual.global_similarity == null ? "—" : number(visual.global_similarity).toFixed(3)} · local ${visual.local_match_ratio == null ? "—" : number(visual.local_match_ratio).toFixed(3)} · combined ${visual.visual_score == null ? "—" : number(visual.visual_score).toFixed(3)}</p>
                      <span class="mono-line">best object ${escapeHtml(shortId(visual.matched_object_id, 16))} · margin ${visual.score_margin == null ? "—" : number(visual.score_margin).toFixed(3)}</span>
                    </section>
                    <section class="reasoning-step">
                      <div class="reasoning-step-heading">
                        <span>3 · Qwen语义判断与最终决定</span>
                        ${review.verdict ? statusPill(review.verdict, `Qwen：${review.verdict}`) : statusPill("pending", "等待Qwen")}
                      </div>
                      <p><b>${escapeHtml(summary.object_name_zh || "未形成对象名称")}</b> · ${escapeHtml(review.short_reason || cluster.error || "尚无判断原因")}</p>
                      <span class="mono-line">identity ${escapeHtml(review.identity_hypothesis || "—")} · final ${escapeHtml(cluster.final_decision || "—")}</span>
                      ${cluster.object_id ? `<span class="chip">object <b class="mono">${escapeHtml(cluster.object_id)}</b></span>` : ""}
                    </section>
                  </div>
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    `);
  }

  function stagePanelHeader(step, title, description, metrics) {
    return `
      <div class="stage-panel-header">
        <div>
          <span class="stage-step-label">${escapeHtml(step)}</span>
          <h3>${escapeHtml(title)}</h3>
          <p>${escapeHtml(description)}</p>
        </div>
        <div class="stage-stat-list" aria-label="本阶段结果数量">
          ${array(metrics)
            .map(
              ([label, value]) => `
                <span class="stage-stat">
                  <strong>${formatInteger(value)}</strong>
                  <small>${escapeHtml(label)}</small>
                </span>
              `
            )
            .join("")}
        </div>
      </div>
    `;
  }

  function previewImage(url, alt) {
    return url
      ? `<img loading="lazy" src="${escapeHtml(url)}" alt="${escapeHtml(alt)}" />`
      : `<div class="empty-state" style="min-height:100%;border:0;border-radius:0"><small>暂无图片资产</small></div>`;
  }

  function renderMemory() {
    const memory = state.memory || {};
    const libraryLabel = memory.memory_label || selectedLibrary()?.label || "默认记忆库";
    const counts = memory.counts || {};
    const objects = array(memory.objects);
    const candidates = array(memory.candidates || memory.proposals);
    const initialized = Boolean(
      memory.initialized ?? memory.database_exists ?? (objects.length > 0 || Object.keys(counts).length > 0)
    );
    const observationCount = number(counts.observations, objects.reduce((sum, object) => sum + array(object.observations).length, 0));
    elements.memoryContext.textContent = `正在查看：${libraryLabel}`;
    updateHtml(elements.memoryOverview, `
      <article><span>活跃对象</span><strong>${formatInteger(counts.active_objects ?? objects.length)}</strong><small>长期档案</small></article>
      <article><span>观测记录</span><strong>${formatInteger(observationCount)}</strong><small>跨视角证据</small></article>
      <article><span>候选记录</span><strong>${formatInteger(counts.proposals ?? candidates.length)}</strong><small>保留、过滤与待定总数</small></article>
      <article><span>数据库状态</span><strong>${initialized ? "已连接" : "未初始化"}</strong><small>memory.sqlite</small></article>
    `);

    if (!initialized) {
      const library = selectedLibrary();
      if (library?.status === "unreadable" || library?.status === "review_only") {
        const title = library.status === "unreadable" ? "记忆库无法读取" : "记忆库不完整，仅供复核";
        updateHtml(elements.memoryContent, emptyState(title, memoryIssueText(library), "!"));
        return;
      }
      updateHtml(elements.memoryContent, emptyState(`“${libraryLabel}”目前为空`, "首次写入会创建 SQLite 与对象资产。", "□"));
      return;
    }
    if (state.selectedMemoryView === "lineage") {
      renderLineage(candidates);
    } else {
      renderObjects(objects);
    }
  }

  function renderObjects(objects) {
    if (!objects.length) {
      updateHtml(elements.memoryContent, emptyState("数据库已建立，但还没有对象卡", "请核对候选是否被过滤、判为不确定或运行失败。", "□"));
      return;
    }
    updateHtml(elements.memoryContent, `<div class="object-grid">${objects.map(objectCard).join("")}</div>`);
  }

  function objectCard(object) {
    const observations = array(object.observations);
    const cover = object.representative_view || object.cover_path || observations[observations.length - 1]?.crop_path;
    const identityFeatures = array(object.stable_identity_features);
    const markings = array(object.brand_or_markings);
    const parts = array(object.part_appearance);
    return `
      <article class="object-card">
        <div class="object-card-top">
          <div class="object-cover">${previewImage(imageUrl(cover), `对象 ${object.id}`)}</div>
          <div class="object-card-info">
            <div class="card-title-row">
              <h3>${escapeHtml(object.object_name_zh || object.fine_category || object.coarse_category || "未命名对象")}</h3>
              ${statusPill(object.status || "completed", object.status === "archived" ? "已归档" : `${observations.length} 观测`)}
            </div>
            <span class="mono-line" title="${escapeHtml(object.id || "")}">${escapeHtml(object.id || "—")}</span>
            <p class="object-category">${escapeHtml([object.coarse_category, object.fine_category].filter(Boolean).join(" / ") || "类别未记录")}</p>
            <small class="object-field-label">部件级外观</small>
            <div class="chip-list">
              ${parts.map((part) => `<span class="chip">${escapeHtml(part.part || "部件")} · <b>${escapeHtml([...array(part.color), ...array(part.material)].join(" / ") || "未记录外观")}</b></span>`).join("")}
              ${object.summary_confidence == null && object.annotation_confidence == null ? "" : `<span class="chip">摘要置信 · <b>${Math.round(number(object.summary_confidence ?? object.annotation_confidence) * 100)}%</b></span>`}
            </div>
            <small class="object-field-label">品牌 / 标记</small>
            <div class="chip-list">${markings.length ? markings.map((value) => `<span class="chip"><b>${escapeHtml(value)}</b></span>`).join("") : '<span class="meta-line">未记录</span>'}</div>
            <small class="object-field-label">稳定汇总描述</small>
            <p class="object-description">${escapeHtml(object.stable_description || object.description || "暂无稳定对象描述")}</p>
            <small class="object-field-label">类内区别特征</small>
            ${identityFeatures.length ? `<ul class="reasoning-errors">${identityFeatures.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ul>` : '<span class="meta-line">未记录</span>'}
          </div>
        </div>
        <div class="observation-timeline" aria-label="${escapeHtml(object.id || "对象")} 的观测时间线">
          ${
            observations.length
              ? observations
                  .map(
                    (observation, index) => `
                      <div class="observation-item" title="${escapeHtml(observation.id || "")}">
                        <div class="observation-image">${previewImage(imageUrl(observation.crop_path || observation.crop), `观测 ${observation.id || index + 1} crop`)}</div>
                        ${observation.mask_path ? `<div class="observation-image">${previewImage(imageUrl(observation.mask_path), `观测 ${observation.id || index + 1} mask`)}</div>` : ""}
                        <small>${String(index + 1).padStart(2, "0")} · ${escapeHtml(shortId(observation.source_image_id, 8))}</small>
                      </div>
                    `
                  )
                  .join("")
              : '<span class="meta-line">暂无观测资产</span>'
          }
        </div>
      </article>
    `;
  }

  function renderLineage(candidates) {
    if (!candidates.length) {
      updateHtml(elements.memoryContent, emptyState("暂无候选血缘", "SAM3 产生候选并写入数据库后，这里会连接 source、proposal、decision 与 object。", "⌁"));
      return;
    }
    updateHtml(elements.memoryContent, `
      <div class="table-wrap">
        <table class="lineage-table">
          <thead><tr><th>候选</th><th>source</th><th>DINO聚类</th><th>SAM点质量</th><th>候选状态</th><th>Qwen聚类假设</th><th>历史DINO证据</th><th>最终决定</th><th>对象</th><th>资产</th><th>审计</th></tr></thead>
          <tbody>
            ${candidates
              .map((candidate) => {
                const asset = candidate.crop_path || candidate.overlay_path;
                return `
                  <tr>
                    <td class="mono" title="${escapeHtml(candidate.id || candidate.proposal_id || "")}">${escapeHtml(shortId(candidate.id || candidate.proposal_id, 17))}</td>
                    <td class="mono" title="${escapeHtml(candidate.source_image_id || "")}">${escapeHtml(shortId(candidate.source_image_id, 14))}</td>
                    <td class="mono" title="${escapeHtml(candidate.target_id || "")}">${escapeHtml(shortId(candidate.target_id, 16) || "—")}<br><code>${escapeHtml(candidate.prompt || "automatic_point_grid")}</code></td>
                    <td>${candidate.score == null ? "—" : number(candidate.score).toFixed(3)}</td>
                    <td>${statusPill(candidate.status || "pending")}</td>
                    <td>${candidate.qwen_hypothesis ? statusPill(candidate.qwen_hypothesis) : "—"}</td>
                    <td>${candidate.visual_evidence?.result ? `${statusPill(candidate.visual_evidence.result)}<br><span class="mono-line">${candidate.visual_evidence.visual_score == null ? "—" : number(candidate.visual_evidence.visual_score).toFixed(3)}</span>` : "—"}</td>
                    <td>${candidate.decision ? statusPill(candidate.decision) : "—"}</td>
                    <td class="mono">${escapeHtml(shortId(candidate.object_id || candidate.observation_object_id || candidate.matched_object_id, 15))}</td>
                    <td>${asset ? `<img class="table-thumb" loading="lazy" src="${escapeHtml(imageUrl(asset))}" alt="候选资产" />` : "—"}</td>
                    <td>${candidate.raw_response_path ? `<a class="quiet-button" href="${escapeHtml(auditJsonUrl(candidate.raw_response_path))}" target="_blank" rel="noopener">JSON</a>` : "—"}</td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      </div>
    `);
  }

  function reportMetrics(report) {
    const images = array(report?.images);
    const run = report?.run || {};
    const sourceCounts = run.source_counts || {};
    const proposalCounts = run.proposal_counts || {};
    const decisions = run.decision_counts || {};
    const clusters = array(report?.clusters);
    const clusterCounts = report?.cluster_counts || {};
    const rawCandidates = images.reduce(
      (sum, image) => sum + number(image.sam?.raw_candidates ?? image.sam?.grid_points ?? image.sam?.above_confidence_threshold_candidates),
      0
    );
    return {
      inputs: images.length,
      uniqueSources: Object.values(sourceCounts).reduce((sum, value) => sum + number(value), 0),
      duplicates: number(run.duplicate_sources_skipped, images.filter((image) => image.duplicate).length),
      rawCandidates,
      kept: images.reduce((sum, image) => sum + number(image.sam?.kept), 0),
      scriptFiltered: images.reduce((sum, image) => sum + number(image.sam?.filtered), 0),
      filtered: number(proposalCounts.filtered),
      clusters: clusters.length,
      reviewedClusters: clusters.filter((cluster) => cluster.qwen_review).length,
      newClusters: number(clusterCounts.new),
      existingClusters: number(clusterCounts.existing),
      ignoredClusters: number(clusterCounts.ignored),
      uncertainClusters: number(clusterCounts.uncertain),
      failedClusters: number(clusterCounts.failed),
      newCount: number(decisions.new),
      existingCount: number(decisions.existing),
      ignoredCount: number(decisions.ignored),
      uncertainCount: number(decisions.uncertain),
      failedProposals: number(proposalCounts.failed),
      observations: number(run.observations_added),
      objects: number(run.active_objects_total),
    };
  }

  function reportErrorMessages(report) {
    const messages = [];
    const collect = (value) => {
      if (value == null || value === false || value === "") return;
      if (Array.isArray(value)) {
        value.forEach(collect);
        return;
      }
      if (typeof value === "object") {
        const message = value.message || value.detail;
        if (message) {
          messages.push(value.type ? `${value.type}: ${message}` : String(message));
          return;
        }
        if (value.error) {
          collect(value.error);
          return;
        }
        messages.push(JSON.stringify(value));
        return;
      }
      messages.push(String(value));
    };
    collect(report?.error);
    collect(report?.progress_error);
    collect(report?.external_errors);
    array(report?.clusters).forEach((cluster) => collect(cluster?.error));
    array(report?.images).forEach((image) => {
      collect(image?.error);
      collect(image?.candidate_reasoning?.errors);
      array(image?.decisions).forEach((decision) => {
        collect(decision?.error);
        collect(decision?.errors);
      });
    });
    return [...new Set(messages)];
  }

  function renderFailedRunSummary(run) {
    const status = run.status || "failed";
    const stateMessage = processErrorMessage(run.message) || "实验进程在生成正式报告前结束。";
    const processError = processErrorMessage(run.process_error) || "后端未提供进程错误详情。";
    const logTail = runLogTail(run) || "后端未提供进程日志尾部。";
    const resolvedExitCode = runExitCode(run);
    const exitCode = resolvedExitCode == null ? "未提供" : String(resolvedExitCode);
    elements.summaryStatus.className = `status-pill ${statusClass(status)}`;
    elements.summaryStatus.textContent = statusLabels[status] || status;
    updateHtml(elements.summaryContent, `
      <div class="summary-details failure-summary-grid">
        <article class="detail-panel">
          <p class="eyebrow">PROCESS FAILURE</p>
          <h3>实验未生成正式报告</h3>
          <ul class="error-list">
            <li><span><b>状态消息</b> · ${escapeHtml(stateMessage)}</span></li>
            <li><span><b>进程错误</b> · ${escapeHtml(processError)}</span></li>
            <li><span><b>退出码</b> · ${escapeHtml(exitCode)}</span></li>
            <li><span><b>运行编号</b> · ${escapeHtml(run.run_id || run.web_run_id || "未提供")}</span></li>
          </ul>
        </article>
        <article class="detail-panel">
          <h3>进程日志尾部</h3>
          <pre class="failure-log">${escapeHtml(logTail)}</pre>
        </article>
      </div>
    `);
  }

  function renderSummary() {
    const report = state.report;
    if (!report) {
      const reportBelongsToRun = !state.run.memory_id || state.run.memory_id === state.selectedMemoryId;
      const terminalFailure = reportBelongsToRun && !state.run.memory_deleted_at_utc && ["failed", "interrupted"].includes(state.run.status);
      if (terminalFailure) {
        renderFailedRunSummary(state.run);
        return;
      }
      elements.summaryStatus.className = "status-pill neutral";
      elements.summaryStatus.textContent = "暂无报告";
      updateHtml(elements.summaryContent, emptyState("所选记忆库暂无正式报告", "完成一次实验后，这里会按五个阶段整理本次增量、库内累计结果、运行错误和人工复核重点。", "∴"));
      return;
    }
    const metrics = reportMetrics(report);
    const status = report.status || "failed";
    elements.summaryStatus.className = `status-pill ${statusClass(status)}`;
    elements.summaryStatus.textContent = statusLabels[status] || status;

    const qwen = report.models?.qwen || {};
    const sam = report.models?.sam3 || {};
    const dino = report.models?.dinov3 || {};
    const errors = reportErrorMessages(report);
    const checks = Object.entries(report.checks || {});
    const demoCoverage = Object.entries(report.demo_coverage || {});
    const libraryLabel = selectedLibrary()?.label || state.memory?.memory_label || "默认记忆库";
    const threshold = number(sam.confidence_threshold, 0.88);
    const elapsed = state.serverSummary && "elapsed_seconds" in state.serverSummary
      ? state.serverSummary.elapsed_seconds
      : state.run.elapsed_seconds;
    const structurePassed = status === "passed";
    const completedWithErrors = status === "completed_with_errors";
    const outcomeClass = structurePassed ? "passed" : completedWithErrors ? "warning" : "failed";
    const formalTitle = structurePassed
      ? "实验完成，结果已写入对象记忆"
      : completedWithErrors
        ? "实验已结束，但部分内容处理失败"
        : "实验未形成可验收的完整结果";
    updateHtml(elements.summaryContent, `
      <section class="summary-outcome ${outcomeClass}">
        <div>
          <p class="eyebrow">RUN OUTCOME</p>
          <h3>${escapeHtml(formalTitle)}</h3>
          <div class="summary-status-row">
            ${statusPill(structurePassed ? "passed" : completedWithErrors ? "uncertain" : "failed", `流程与数据检查：${structurePassed ? "通过" : completedWithErrors ? "有错误" : "未通过"}`)}
            ${statusPill("uncertain", "物体识别质量：待人工复核")}
          </div>
        </div>
        <div class="summary-run-meta">
          <span>记忆库 · <b>${escapeHtml(libraryLabel)}</b></span>
          <span>运行编号 · <code>${escapeHtml(report.run?.run_id || "—")}</code></span>
          <span>总耗时 · <b>${elapsed == null ? "—" : formatSeconds(elapsed)}</b></span>
        </div>
        <p>${structurePassed ? "流程完整结束且结构化结果可读取；物体是否找全、粒度是否完整、颜色是否正确及跨图身份是否一致，仍需结合下方证据人工确认。" : "先查看运行错误与对应阶段证据；缺失或失败的内容不会被摘要数字包装成成功结果。"}</p>
      </section>
      <div class="summary-stage-grid">
        ${summaryStageCard("01", "输入整理", [
          ["输入文件", metrics.inputs],
          ["本次登记源图", metrics.uniqueSources],
          ["重复文件跳过", metrics.duplicates],
        ])}
        ${summaryStageCard("02", "自动分割", [
          ["点网格原始候选", metrics.rawCandidates],
          ["保留候选", metrics.kept],
          ["脚本过滤候选", metrics.scriptFiltered],
        ], `点提示质量阈值 ${threshold.toFixed(2)}；候选尚未被视为对象。`)}
        ${summaryStageCard("03", "视觉聚类", [
          ["DINO视觉指纹", number(dino.fingerprints)],
          ["候选聚类", metrics.clusters],
          ["已审查聚类", metrics.reviewedClusters],
        ])}
        ${summaryStageCard("04", "聚类语义决定", [
          ["新对象聚类", metrics.newClusters],
          ["已有对象聚类", metrics.existingClusters],
          ["背景 / 零件过滤", metrics.ignoredClusters],
          ["不确定聚类", metrics.uncertainClusters],
          ["失败聚类", metrics.failedClusters],
        ], "每个聚类的接触表、视觉相似度、Qwen判断和最终对象ID可在中间结果中追溯。")}
        ${summaryStageCard("05", "记忆写入", [
          ["本次新增观测", metrics.observations],
          ["库内活跃对象", metrics.objects],
        ], "新增观测是本次增量；活跃对象是所选记忆库的累计总数。")}
      </div>
      <div class="summary-details summary-review-grid">
        <article class="detail-panel">
          <h3>运行错误 · ${errors.length}</h3>
          <ul class="error-list">
            ${errors.length ? errors.map((error) => `<li><span>${escapeHtml(error)}</span></li>`).join("") : '<li class="no-error"><span>正式报告未记录运行错误。这不代表语义结果已经正确。</span></li>'}
          </ul>
        </article>
        <article class="detail-panel">
          <h3>人工复核清单</h3>
          <ul class="review-list">
            <li><a href="#intermediates">点网格是否覆盖了所有真实物体</a></li>
            <li><a href="#intermediates">DINO是否把同一物体跨视角聚在一起</a></li>
            <li><a href="#intermediates">背景、零件和完整物体的Qwen判断是否正确</a></li>
            <li><a href="#memory">跨图片的对象身份归并与观测时间线</a></li>
          </ul>
        </article>
      </div>
      <article class="detail-panel timing-panel">
        <h3>模型与耗时</h3>
        <div class="timing-grid">
          ${timingMetric("Qwen 加载", qwen.model_load_seconds, qwen.loaded ? "聚类后顺序驻留" : "未加载")}
          ${timingMetric("Qwen 推理", qwen.inference_seconds, `${number(qwen.calls ?? qwen.total_calls)} 个聚类批次`)}
          ${timingMetric("SAM3 推理", sam.inference_seconds, `峰值 ${number(sam.peak_memory_mib).toFixed(0)} MiB`)}
          ${timingMetric("DINOv3 推理", dino.inference_seconds, `${number(dino.fingerprints)} 个视觉指纹`)}
        </div>
      </article>
      <details class="technical-details">
        <summary>技术检查详情 · ${checks.length + demoCoverage.length} 项</summary>
        <ul class="check-list">
          ${checks.length ? checks.map(([name, ok]) => `<li class="${ok ? "ok" : "bad"}"><span>${ok ? "通过" : "未通过"} · <code>${escapeHtml(name)}</code></span></li>`).join("") : '<li class="bad"><span>报告没有结构检查字段</span></li>'}
          ${demoCoverage.map(([name, ok]) => `<li class="${ok ? "ok" : "bad"}"><span>${ok ? "覆盖" : "未覆盖"} · demo · <code>${escapeHtml(name)}</code></span></li>`).join("")}
        </ul>
      </details>
    `);
  }

  function summaryStageCard(index, title, rows, note = "") {
    return `<article class="summary-stage-card">
      <div class="summary-stage-title"><span>${escapeHtml(index)}</span><h3>${escapeHtml(title)}</h3></div>
      <dl>${rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${formatInteger(value)}</dd></div>`).join("")}</dl>
      ${note ? `<p>${escapeHtml(note)}</p>` : ""}
    </article>`;
  }

  function timingMetric(label, seconds, hint) {
    return `<div class="timing-card"><span>${escapeHtml(label)}</span><strong>${seconds == null ? "—" : formatSeconds(seconds)}</strong><small>${escapeHtml(hint)}</small></div>`;
  }

  function emptyState(title, description, symbol) {
    return `<div class="empty-state"><span class="empty-symbol" aria-hidden="true">${escapeHtml(symbol)}</span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p></div>`;
  }

  function clearSelectedMemoryViews({ invalidate = false } = {}) {
    if (invalidate) state.viewEpoch += 1;
    Object.assign(state, {
      report: null,
      serverSummary: null,
      memory: null,
      candidateAssetKinds: new Map(),
      lastIntermediateRenderKey: "",
      pendingIntermediateRender: false,
    });
    renderSummary();
    renderMemory();
    renderIntermediates({ force: true });
  }

  function setSelectedMemoryId(memoryId, { clear = false } = {}) {
    const next = String(memoryId || "default");
    if (next === state.selectedMemoryId) return false;
    state.selectedMemoryId = next;
    state.viewEpoch += 1;
    if (clear) clearSelectedMemoryViews({ invalidate: false });
    return true;
  }

  async function selectMemoryLibrary(memoryId) {
    if (memoryControlsLocked() || memoryId === state.selectedMemoryId) {
      renderMemoryLibraryControl();
      return;
    }
    const previousMemoryId = state.selectedMemoryId;
    setSelectedMemoryId(memoryId, { clear: true });
    state.catalogMutationInFlight = true;
    renderMemoryLibraryControl();
    renderControlAvailability();
    try {
      await apiJson(`/api/memories/${encodeURIComponent(memoryId)}/select`, {
        method: "POST",
      });
      state.catalogMutationInFlight = false;
      await refreshMemoryLibraries({ quiet: true, preferServerSelection: true });
      await Promise.all([
        refreshMemory({ quiet: true }),
        refreshResults({ quiet: true }),
      ]);
      renderIntermediates({ force: true });
    } catch (error) {
      setSelectedMemoryId(previousMemoryId, { clear: true });
      state.catalogMutationInFlight = false;
      await refreshMemoryLibraries({ quiet: true, preferServerSelection: true });
      await Promise.all([
        refreshMemory({ quiet: true }),
        refreshResults({ quiet: true }),
      ]);
      toast(`无法切换记忆库：${error.message}`, "error");
    }
  }

  async function createMemoryLibrary() {
    if (memoryControlsLocked()) return;
    const label = window.prompt(
      "为新的空白记忆库命名（最多 40 个字符）。\n\n名称只用于页面识别，系统会生成安全的内部编号。",
      ""
    );
    if (label == null) return;
    state.viewEpoch += 1;
    state.catalogMutationInFlight = true;
    renderControlAvailability();
    try {
      const payload = await apiJson("/api/memories", {
        method: "POST",
        body: JSON.stringify({ label }),
        headers: { "Content-Type": "application/json" },
      });
      setSelectedMemoryId(payload.selected_id, { clear: true });
      state.catalogMutationInFlight = false;
      await refreshMemoryLibraries({ quiet: false, preferServerSelection: true });
      await Promise.all([refreshMemory({ quiet: true }), refreshResults({ quiet: true })]);
      toast(`已创建并选中“${payload.item?.label || "新记忆库"}”`);
    } catch (error) {
      state.catalogMutationInFlight = false;
      renderControlAvailability();
      toast(`无法新建记忆库：${error.message}`, "error");
    }
  }

  async function deleteMemoryLibrary() {
    if (memoryControlsLocked()) return;
    const library = selectedLibrary();
    if (!library) return;
    const counts = library.counts || {};
    const action = library.id === "default" ? "清空" : "永久删除";
    const typed = window.prompt(
      `${action}“${library.label}”？\n\n这会删除该库的 SQLite、源图副本、候选资产、对象观测、原始模型回答和库内运行报告；实验输入图片不会删除。页面无法撤销此操作。\n\n当前包含 ${formatInteger(counts.active_objects)} 个活跃对象、${formatInteger(counts.observations)} 条观测、${formatInteger(counts.runs)} 次运行。\n\n请输入完整名称以确认：`,
      ""
    );
    if (typed == null || typed !== library.label) {
      if (typed != null) toast("名称不一致，未删除记忆库", "error");
      return;
    }
    state.viewEpoch += 1;
    state.catalogMutationInFlight = true;
    renderControlAvailability();
    try {
      const payload = await apiJson(
        `/api/memories/${encodeURIComponent(library.id)}?confirm=${encodeURIComponent(library.id)}`,
        { method: "DELETE" }
      );
      setSelectedMemoryId(payload.selected_id || "default", { clear: true });
      if (state.memory !== null || state.report !== null) {
        clearSelectedMemoryViews({ invalidate: false });
      }
      state.catalogMutationInFlight = false;
      await refreshMemoryLibraries({ quiet: false, preferServerSelection: true });
      await Promise.all([refreshMemory({ quiet: true }), refreshResults({ quiet: true })]);
      if (payload.cleanup_pending) {
        toast("记忆库已从页面移除，但服务器仍有一份待清理的临时隔离副本。", "error");
      } else {
        toast(payload.action === "cleared" ? "默认记忆库已清空" : "记忆库已删除");
      }
    } catch (error) {
      state.catalogMutationInFlight = false;
      renderControlAvailability();
      toast(`无法删除记忆库：${error.message}`, "error");
    }
  }

  async function uploadFiles(fileList) {
    const files = [...fileList];
    if (!files.length || memoryControlsLocked()) return;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    elements.uploadButton.disabled = true;
    try {
      const payload = await apiJson("/api/inputs", { method: "POST", body: form, headers: {} });
      toast(payload.message || `已上传 ${files.length} 个文件`);
      elements.fileInput.value = "";
      await refreshInputs();
    } catch (error) {
      toast(`上传失败：${error.message}`, "error");
    } finally {
      renderControlAvailability();
    }
  }

  async function deleteInput(path) {
    if (memoryControlsLocked()) return;
    const confirmed = window.confirm(`确定从本次实验输入中移除“${fileName(path)}”吗？\n\n已经生成的对象记忆不会受到影响。`);
    if (!confirmed) return;
    try {
      const payload = await apiJson(`/api/inputs?path=${encodeURIComponent(path)}`, { method: "DELETE" });
      toast(payload.message || "输入图片已删除");
      await refreshInputs();
    } catch (error) {
      toast(`删除失败：${error.message}`, "error");
    }
  }

  async function clearInputs() {
    if (memoryControlsLocked() || !state.inputSummary.total) return;
    const inputCount = state.inputSummary.total;
    const confirmed = window.confirm(
      `确定删除当前全部 ${inputCount} 张实验输入吗？\n\n这只会清空实验输入区；已经生成的对象记忆、源图副本和运行报告不会受到影响。`
    );
    if (!confirmed) return;
    state.catalogMutationInFlight = true;
    renderControlAvailability();
    try {
      const payload = await apiJson("/api/inputs/all", { method: "DELETE" });
      state.catalogMutationInFlight = false;
      await refreshInputs();
      toast(`已清空 ${formatInteger(payload.deleted_count)} 张实验输入`);
    } catch (error) {
      state.catalogMutationInFlight = false;
      renderControlAvailability();
      await refreshInputs({ quiet: true });
      toast(`无法清空实验输入：${error.message}`, "error");
    }
  }

  async function startRun() {
    if (memoryControlsLocked() || !state.inputSummary.total) return;
    const library = selectedLibrary();
    if (!library?.continuable) return;
    const counts = library.counts || {};
    const memoryContext = library.status === "empty"
      ? "这是空白库，本次会创建一份独立对象记忆，并执行首次 Demo 结构覆盖检查。"
      : `本次结果会并入现有 ${formatInteger(counts.active_objects)} 个对象和 ${formatInteger(counts.observations)} 条观测，不会生成独立副本，也不会重复套用空白库的首次覆盖条件。`;
    const confirmed = window.confirm(
      `将使用当前 ${state.inputSummary.total} 个输入文件启动完整端到端实验。\n\n目标记忆库：${library.label}（${memoryStatusText(library.status)}）\n${memoryContext}\n\n运行期间会锁定输入和记忆库管理，并联合加载 Qwen3-VL、SAM3 与 DINOv3，随后逐图闭环处理。是否继续？`
    );
    if (!confirmed) return;
    state.viewEpoch += 1;
    state.catalogMutationInFlight = true;
    renderMemoryLibraryControl();
    renderControlAvailability();
    try {
      const payload = await apiJson("/api/runs", {
        method: "POST",
        body: JSON.stringify({ memory_id: state.selectedMemoryId }),
        headers: { "Content-Type": "application/json" },
      });
      const incomingRun = payload.state || payload.run || payload;
      Object.assign(state, {
        events: [],
        latestSequence: 0,
        report: null,
        serverSummary: null,
        memory: null,
        candidateAssetKinds: new Map(),
        lastIntermediateRenderKey: "",
        pendingIntermediateRender: false,
        lastDataRefresh: 0,
      });
      state.run = { ...incomingRun, status: payload.status || incomingRun.status || "starting" };
      setSelectedMemoryId(incomingRun.memory_id || state.selectedMemoryId, { clear: false });
      state.catalogMutationInFlight = false;
      state.serverCatalogLocked = true;
      renderMemoryLibraryControl();
      renderRun();
      renderSummary();
      renderIntermediates({ force: true });
      toast("实验已启动，页面将持续显示真实阶段事件");
      await refreshRun();
    } catch (error) {
      state.catalogMutationInFlight = false;
      toast(`无法启动实验：${error.message}`, "error");
      renderControlAvailability();
      await Promise.all([
        refreshResults({ quiet: true }),
        refreshMemory({ quiet: true }),
        refreshMemoryLibraries({ quiet: true, preferServerSelection: true }),
      ]);
    }
  }

  function releaseStagePointer() {
    if (!state.stagePointerActive) return;
    state.stagePointerActive = false;
    window.setTimeout(() => {
      if (
        state.stagePointerActive
        || !state.pendingIntermediateRender
        || hasActiveStageSelection()
      ) return;
      renderIntermediates();
    }, 0);
  }

  function bindEvents() {
    elements.uploadButton.addEventListener("click", () => elements.fileInput.click());
    elements.clearInputsButton.addEventListener("click", clearInputs);
    elements.fileInput.addEventListener("change", () => uploadFiles(elements.fileInput.files));
    elements.dropZone.addEventListener("click", () => {
      if (!memoryControlsLocked()) elements.fileInput.click();
    });
    elements.dropZone.addEventListener("keydown", (event) => {
      if ((event.key === "Enter" || event.key === " ") && !memoryControlsLocked()) {
        event.preventDefault();
        elements.fileInput.click();
      }
    });
    ["dragenter", "dragover"].forEach((name) =>
      elements.dropZone.addEventListener(name, (event) => {
        event.preventDefault();
        if (!memoryControlsLocked()) elements.dropZone.classList.add("dragging");
      })
    );
    ["dragleave", "drop"].forEach((name) =>
      elements.dropZone.addEventListener(name, (event) => {
        event.preventDefault();
        elements.dropZone.classList.remove("dragging");
      })
    );
    elements.dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
    elements.startRunButton.addEventListener("click", startRun);
    elements.memoryLibrarySelect.addEventListener("change", () => {
      selectMemoryLibrary(elements.memoryLibrarySelect.value);
    });
    elements.newMemoryButton.addEventListener("click", createMemoryLibrary);
    elements.deleteMemoryButton.addEventListener("click", deleteMemoryLibrary);

    elements.inputGrid.addEventListener("click", (event) => {
      const button = event.target.closest("[data-delete-input]");
      if (button) deleteInput(button.dataset.deleteInput);
    });

    document.querySelectorAll("[data-stage-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedStage = button.dataset.stageTab;
        document.querySelectorAll("[data-stage-tab]").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
        renderIntermediates({ force: true });
      });
    });

    document.querySelectorAll("[data-memory-view]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedMemoryView = button.dataset.memoryView;
        document.querySelectorAll("[data-memory-view]").forEach((item) => item.classList.toggle("active", item === button));
        renderMemory();
      });
    });

    elements.stagePanel.addEventListener("pointerdown", () => {
      state.stagePointerActive = true;
    });
    window.addEventListener("pointerup", releaseStagePointer);
    window.addEventListener("pointercancel", releaseStagePointer);
    window.addEventListener("blur", releaseStagePointer);

    elements.stagePanel.addEventListener("click", (event) => {
      const button = event.target.closest("[data-asset-url]");
      if (!button) return;
      const container = button.closest(".candidate-asset");
      const card = button.closest("[data-candidate-id]");
      const image = container?.querySelector(":scope > img");
      if (card?.dataset.candidateId && button.dataset.assetKind) {
        state.candidateAssetKinds.set(card.dataset.candidateId, button.dataset.assetKind);
      }
      if (image) image.src = button.dataset.assetUrl;
      container?.querySelectorAll(".asset-toggle").forEach((item) => item.classList.toggle("active", item === button));
    });

    document.addEventListener("selectionchange", () => {
      if (state.pendingIntermediateRender && !hasActiveStageSelection()) {
        renderIntermediates();
      }
    });
  }

  async function initialize() {
    bindEvents();
    await refreshMemoryLibraries({ quiet: true, preferServerSelection: true });
    await Promise.all([
      refreshInputs({ quiet: true }),
      refreshResults({ quiet: true }),
      refreshMemory({ quiet: true }),
    ]);
    state.lastDataRefresh = Date.now();
    await refreshRun({ initial: true });
    state.pollTimer = window.setInterval(() => refreshRun({ initial: false }), POLL_INTERVAL_MS);
    state.clockTimer = window.setInterval(updateClock, 1000);
  }

  initialize();
})();
