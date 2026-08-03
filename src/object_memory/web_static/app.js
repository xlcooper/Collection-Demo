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
    scene_guidance: "Qwen 首轮目标规划",
    sam3: "SAM3 分割与候选过滤",
    candidate_reasoning: "Qwen 候选与对象判断",
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
    ignored: "已忽略",
    uncertain: "不确定",
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
    selectedStage: "input",
    selectedMemoryView: "objects",
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
    uploadButton: document.querySelector("#upload-button"),
    fileInput: document.querySelector("#file-input"),
    dropZone: document.querySelector("#drop-zone"),
    inputMetrics: document.querySelector("#input-metrics"),
    inputGrid: document.querySelector("#input-grid"),
    eventFreshness: document.querySelector("#event-freshness"),
    stagePanel: document.querySelector("#stage-panel"),
    memoryOverview: document.querySelector("#memory-overview"),
    memoryContent: document.querySelector("#memory-content"),
    summaryStatus: document.querySelector("#summary-status"),
    summaryContent: document.querySelector("#summary-content"),
    toastRegion: document.querySelector("#toast-region"),
  };

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
      "scene_guidance",
      "sam3",
      "candidate_reasoning",
      "memory",
      "report",
    ].includes(String(status));
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
    if (["passed", "completed", "new", "existing"].includes(value)) return value;
    if (["failed", "error", "interrupted"].includes(value)) return "failed";
    if (["completed_with_errors", "uncertain", "pending"].includes(value)) return "warning";
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

  function memoryAssetUrl(path) {
    return `/api/memory-asset?path=${encodeURIComponent(path || "")}`;
  }

  function auditJsonUrl(path) {
    return `/api/audit-json?path=${encodeURIComponent(path || "")}`;
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
    state.pollInFlight = true;
    try {
      const wasRunning = isRunningStatus(state.run.status);
      const previousWebRunId = state.run.web_run_id;
      const requestedSequence = state.latestSequence;
      const suffix = requestedSequence ? `?after_sequence=${requestedSequence}` : "";
      let payload = await apiJson(`/api/runs/current${suffix}`);
      let incomingRun = payload.state || payload.run || payload;
      const incomingWebRunId = incomingRun.web_run_id;
      const runChanged = Boolean(incomingWebRunId && incomingWebRunId !== previousWebRunId);
      if (runChanged) {
        Object.assign(state, {
          events: [],
          latestSequence: 0,
          report: null,
          serverSummary: null,
          lastDataRefresh: 0,
        });
        if (requestedSequence > 0) {
          payload = await apiJson("/api/runs/current?after_sequence=0");
          incomingRun = payload.state || payload.run || payload;
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
        await Promise.all([refreshResults({ quiet: true }), refreshMemory({ quiet: true })]);
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
    try {
      const payload = await apiJson("/api/results");
      if (payload.state && typeof payload.state === "object") {
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
            lastDataRefresh: 0,
          });
          state.run = { status: "idle", ...payload.state };
          renderRun();
        } else {
          state.run = { ...state.run, ...payload.state };
        }
      }
      const incoming = payload.report || (payload.schema_version ? payload : null);
      const currentEnough = payload.is_current_run !== false || !isRunningStatus(state.run.status);
      if (payload.available === false) {
        state.report = null;
        state.serverSummary = null;
      } else if (incoming && currentEnough) {
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
    try {
      state.memory = await apiJson("/api/memory");
      renderMemory();
      renderIntermediates();
    } catch (error) {
      if (!quiet) toast(`无法读取对象记忆：${error.message}`, "error");
    }
  }

  function renderInputs() {
    const summary = state.inputSummary;
    elements.inputMetrics.innerHTML = `
      <article><span>文件总数</span><strong>${formatInteger(summary.total)}</strong></article>
      <article><span>唯一内容</span><strong>${formatInteger(summary.unique)}</strong></article>
      <article><span>重复副本</span><strong>${formatInteger(summary.duplicates)}</strong></article>
      <article><span>支持格式</span><strong>JPG · PNG · WEBP</strong></article>
    `;

    if (!state.inputs.length) {
      elements.inputGrid.innerHTML = emptyState(
        "输入目录为空",
        "上传至少一张图片后才能运行实验。",
        "＋"
      );
      return;
    }

    const locked = isRunningStatus(state.run.status);
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
      scene_guidance_started: "准备进行首轮场景目标规划",
      scene_guidance_batch_started: `Qwen 正在分析第 ${current + 1}/${total} 个场景批次`,
      scene_guidance_batch_completed: `第 ${current}/${total} 个场景批次已完成`,
      scene_guidance_completed: "首轮目标规划完成",
      sam3_started: "准备进行 SAM3 文本定向分割",
      sam3_image_started: `SAM3 正在处理第 ${current + 1}/${total} 张图`,
      sam3_image_completed: `SAM3 已完成第 ${current}/${total} 张图`,
      sam3_completed: "SAM3 分割与候选过滤完成",
      candidate_reasoning_started: "准备进行候选复核与对象身份判断",
      candidate_reasoning_image_started: `Qwen 正在判断第 ${current + 1}/${total} 张图的候选`,
      candidate_reasoning_image_completed: `Qwen 已完成第 ${current}/${total} 张图的对象判断`,
      candidate_reasoning_completed: "候选复核与对象记忆写入完成",
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
    const flowStages = ["input", "scene_guidance", "sam3", "candidate_reasoning", "memory", "report"];
    const asFlowStage = (stage) => {
      const candidate = stage === "input_registration" ? "input" : stage;
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
    elements.runUnit.textContent = terminalFailure
      ? runExitCode(state.run) != null
        ? `子进程退出码 ${runExitCode(state.run)}`
        : statusLabels[runStatus] || runStatus
      : current != null && total != null
        ? `已完成 ${current} / ${total}`
        : message;
    elements.progressBar.style.width = `${progress}%`;
    elements.progressTrack.setAttribute("aria-valuenow", String(Math.round(progress)));
    elements.progressMessage.textContent = message;
    elements.progressPercent.textContent = `${Math.round(progress)}%`;

    const running = isRunningStatus(runStatus);
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
    const order = ["input", "scene_guidance", "sam3", "candidate_reasoning", "memory", "report"];
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
    const running = isRunningStatus(state.run.status);
    const hasInputs = state.inputSummary.total > 0;
    elements.startRunButton.disabled = running || !hasInputs;
    elements.uploadButton.disabled = running;
    elements.dropZone.classList.toggle("locked", running);
    elements.dropZone.setAttribute("aria-disabled", String(running));
    elements.runLockNote.textContent = running
      ? "实验正在运行，输入上传、删除和再次启动已锁定。"
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

  function imageStateKey(image) {
    const inputPath = String(image?.input_path || "").trim().replaceAll("\\", "/");
    if (inputPath) return `input:${inputPath}`;
    const sourceId = String(image?.source_id || "").trim();
    return sourceId ? `source:${sourceId}` : "";
  }

  function imageStates() {
    const byKey = new Map();
    const mergeImage = (image) => {
      const key = imageStateKey(image);
      if (!key) return;
      const previous = byKey.get(key) || {};
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
    };
    for (const event of state.events) {
      const data = event.data || {};
      const direct = data.input_path || data.source_id
        ? [
            event.event === "sam3_image_completed"
              ? {
                  input_path: data.input_path,
                  source_id: data.source_id,
                  status: data.work_status,
                  sam: {
                    above_confidence_threshold_candidates: data.above_confidence_threshold_candidates,
                    prompt_detection_counts: data.prompt_detection_counts,
                    zero_candidate_prompts: data.zero_candidate_prompts,
                    inference_seconds: data.inference_seconds,
                    kept: array(data.kept).length,
                    filtered: array(data.filtered).length,
                    filter_counts: data.filter_counts,
                  },
                  kept_proposals: data.kept,
                  filtered_proposals: data.filtered,
                  error: data.error,
                }
              : event.event === "candidate_reasoning_image_completed"
                ? {
                    input_path: data.input_path,
                    source_id: data.source_id,
                    status: data.work_status,
                    candidate_reasoning: data.candidate_reasoning,
                    decisions: data.decisions,
                    error: data.error,
                  }
                : {
                    ...data,
                    status: data.work_status || data.status,
                  },
          ]
        : [];
      const images = array(data.images).concat(data.image ? [data.image] : [], direct);
      for (const image of images) mergeImage(image);
    }
    // The final report is authoritative for terminal status and errors. Merge it
    // last while preserving event-only proposal details accumulated above.
    for (const image of array(state.report?.images)) mergeImage(image);
    return [...byKey.values()];
  }

  function renderIntermediates() {
    const images = imageStates();
    const sceneCount = images.reduce((sum, image) => sum + number(image.scene_guidance?.target_count, 0), 0);
    const keptCount = images.reduce((sum, image) => sum + number(image.sam?.kept, 0), 0);
    const decisionCount = images.reduce((sum, image) => sum + array(image.decisions).length, 0);
    document.querySelector("#tab-count-input").textContent = images.length ? `${images.length} 张` : "—";
    document.querySelector("#tab-count-scene").textContent = images.length ? `${sceneCount} 目标` : "—";
    document.querySelector("#tab-count-sam").textContent = images.length ? `${keptCount} 保留` : "—";
    document.querySelector("#tab-count-reasoning").textContent = images.length ? `${decisionCount} 判断` : "—";

    if (state.selectedStage === "input") renderDedupStage(images);
    if (state.selectedStage === "scene_guidance") renderSceneStage(images);
    if (state.selectedStage === "sam3") renderSamStage(images);
    if (state.selectedStage === "candidate_reasoning") renderReasoningStage(images);
  }

  function renderDedupStage(images) {
    if (!images.length) {
      elements.stagePanel.innerHTML = emptyState("等待去重结果", "实验登记输入后会显示每个文件的 canonical source 与重复状态。", "◎");
      return;
    }
    const duplicates = images.filter((image) => image.duplicate);
    elements.stagePanel.innerHTML = `
      ${stagePanelHeader("输入登记与去重", `${images.length} 个输入记录，${duplicates.length} 个内容副本按 SHA-256 跳过。`, `${images.length - duplicates.length} 唯一 source`)}
      <div class="table-wrap">
        <table class="lineage-table">
          <thead><tr><th>输入</th><th>SHA-256</th><th>source_id</th><th>状态</th><th>存储源图</th><th>错误</th></tr></thead>
          <tbody>
            ${images
              .map(
                (image) => `
                  <tr>
                    <td class="mono">${escapeHtml(image.input_path || "—")}</td>
                    <td class="mono" title="${escapeHtml(image.sha256 || "")}">${escapeHtml(shortId(image.sha256, 16))}</td>
                    <td class="mono" title="${escapeHtml(image.source_id || "")}">${escapeHtml(shortId(image.source_id, 17))}</td>
                    <td>${statusPill(image.duplicate ? "duplicate" : image.status, image.duplicate ? "重复跳过" : statusLabels[image.status] || image.status)}</td>
                    <td class="mono">${escapeHtml(image.stored_source || "—")}</td>
                    <td>${escapeHtml(image.error || "—")}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function sourcePreview(image) {
    if (image.stored_source) return memoryAssetUrl(image.stored_source);
    const inputName = String(image.input_path || "").replaceAll("\\", "/").split("/").pop();
    const matched = state.inputs.find((item) => (item.name || item.path) === inputName);
    return matched ? matched.url || inputAssetUrl(matched.path || matched.name) : "";
  }

  function renderSceneStage(images) {
    const guided = images.filter((image) => image.scene_guidance);
    if (!guided.length) {
      elements.stagePanel.innerHTML = emptyState("等待首轮目标", "Qwen 完成一个场景批次后会显示原图、场景摘要与 SAM3 文本概念。", "02");
      return;
    }
    const targetCount = guided.reduce((sum, image) => sum + number(image.scene_guidance?.target_count, 0), 0);
    elements.stagePanel.innerHTML = `
      ${stagePanelHeader("Qwen 首轮目标规划", `按每批最多 4 张图生成完整物体概念；首轮漏掉的物体不会进入 SAM3。`, `${targetCount} 个目标`)}
      <div class="evidence-list">
        ${guided
          .map((image) => {
            const guidance = image.scene_guidance || {};
            const targets = array(guidance.targets);
            return `
              <article class="evidence-card">
                <div class="evidence-image">${previewImage(sourcePreview(image), `源图 ${image.source_id || ""}`)}</div>
                <div class="evidence-body">
                  <div class="card-title-row">
                    <div>
                      <h4>${escapeHtml(image.source_id || image.input_path || "未知源图")}</h4>
                      <span class="mono-line">${escapeHtml(guidance.scope_id || "—")} · batch ${escapeHtml(guidance.batch_index ?? "—")}</span>
                    </div>
                    <div class="chip-list" style="margin-top:0">
                      ${guidance.errors?.length ? statusPill("failed", "有错误") : statusPill("info", `${targets.length} 目标`)}
                      ${guidance.raw_response ? `<a class="quiet-button" href="${escapeHtml(auditJsonUrl(guidance.raw_response))}" target="_blank" rel="noopener">原始响应</a>` : ""}
                    </div>
                  </div>
                  <p>${escapeHtml(guidance.scene_summary || guidance.no_target_reason || guidance.errors?.join("；") || "尚无可用场景摘要")}</p>
                  <div class="chip-list">
                    ${targets.length
                      ? targets
                          .map(
                            (target) => `
                              <span class="chip prompt" title="${escapeHtml(target.selection_short_reason || "")}">
                                <b>${escapeHtml(target.object_name_zh || target.target_id)}</b>
                                ${escapeHtml(target.sam_text_prompt)}
                                · ${Math.round(number(target.confidence) * 100)}%
                              </span>
                            `
                          )
                          .join("")
                      : '<span class="chip">本图无目标</span>'}
                  </div>
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
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
    const relevant = images.filter(
      (image) =>
        image.sam &&
        (image.sam.kept ||
          image.sam.filtered ||
          image.sam.above_confidence_threshold_candidates ||
          Object.keys(image.sam.prompt_detection_counts || {}).length ||
          (image.error && number(image.scene_guidance?.target_count) > 0))
    );
    if (!relevant.length) {
      elements.stagePanel.innerHTML = emptyState("等待 SAM3 结果", "每张图完成文字查找与脚本过滤后，会显示检测数、0 候选提示和图像资产。", "03");
      return;
    }
    const kept = relevant.reduce((sum, image) => sum + number(image.sam?.kept), 0);
    const filtered = relevant.reduce((sum, image) => sum + number(image.sam?.filtered), 0);
    elements.stagePanel.innerHTML = `
      ${stagePanelHeader("SAM3 分割与脚本过滤", `文字阈值后的区域再经过面积、重复、包含关系与数量上限检查。`, `${kept} 保留 · ${filtered} 过滤`)}
      <div class="evidence-list">
        ${relevant
          .map((image) => {
            const sam = image.sam || {};
            const candidates = candidatesForImage(image);
            const promptCounts = Object.entries(sam.prompt_detection_counts || {});
            return `
              <article class="evidence-card" style="grid-template-columns: 1fr">
                <div class="evidence-body">
                  <div class="card-title-row">
                    <div>
                      <h4>${escapeHtml(image.source_id || image.input_path || "未知源图")}</h4>
                      <span class="mono-line">SAM ${escapeHtml(sam.inference_seconds ?? "—")}s · ${number(sam.above_confidence_threshold_candidates)} 个阈值上候选</span>
                    </div>
                    <div class="chip-list" style="margin-top:0">
                      ${statusPill("info", `${number(sam.kept)} 保留`)}
                      ${number(sam.filtered) ? statusPill("filtered", `${number(sam.filtered)} 过滤`) : ""}
                    </div>
                  </div>
                  <div class="chip-list">
                    ${promptCounts
                      .map(([prompt, count]) => `<span class="chip ${number(count) === 0 ? "" : "prompt"}">${escapeHtml(prompt)} · <b>${formatInteger(count)}</b></span>`)
                      .join("") || '<span class="chip">暂无提示统计</span>'}
                  </div>
                  ${image.error ? `<p class="meta-line">阶段错误 · ${escapeHtml(image.error)}</p>` : ""}
                  ${candidates.length ? `<div class="candidate-grid">${candidates.map(candidateCard).join("")}</div>` : ""}
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function candidateCard(candidate) {
    const id = candidate.id || candidate.proposal_id || candidate.raw_candidate_id || "candidate";
    const crop = candidate.crop || candidate.crop_path;
    const mask = candidate.mask || candidate.mask_path;
    const overlay = candidate.overlay || candidate.overlay_path;
    const initialAsset = overlay || crop || mask;
    const assets = [
      ["overlay", overlay, "定位"],
      ["crop", crop, "裁剪"],
      ["mask", mask, "掩码"],
    ].filter(([, path]) => path);
    return `
      <article class="candidate-card">
        <div class="candidate-asset">
          ${previewImage(imageUrl(initialAsset), `候选 ${id}`)}
          ${
            assets.length > 1
              ? `<div class="candidate-asset-switch">${assets
                  .map(
                    ([kind, path, label], index) => `<button class="asset-toggle ${index === 0 ? "active" : ""}" type="button" data-asset-url="${escapeHtml(
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
          <p><b>${escapeHtml(candidate.sam_text_prompt || candidate.prompt || "unknown")}</b> · score ${number(candidate.score).toFixed(3)}</p>
          ${candidate.filter_reason ? `<p class="meta-line">过滤原因 · ${escapeHtml(candidate.filter_reason)}</p>` : ""}
          ${candidate.error ? `<p class="meta-line">错误 · ${escapeHtml(candidate.error)}</p>` : ""}
          <span class="mono-line">bbox ${escapeHtml(bboxText(candidate.bbox))} · area ${number(candidate.mask_area_ratio) ? `${(number(candidate.mask_area_ratio) * 100).toFixed(2)}%` : "—"}</span>
        </div>
      </article>
    `;
  }

  function bboxText(bbox) {
    if (!bbox) return "—";
    if (Array.isArray(bbox)) return bbox.map((value) => number(value).toFixed(0)).join(", ");
    return [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
      .map((value) => number(value).toFixed(0))
      .join(", ");
  }

  function renderReasoningStage(images) {
    const relevant = images.filter((image) => array(image.decisions).length || image.candidate_reasoning);
    if (!relevant.length) {
      elements.stagePanel.innerHTML = emptyState("等待对象判断", "第二轮 Qwen 会对每个正式候选给出有效性、临时标注与对象身份决定。", "04");
      return;
    }
    const decisions = relevant.flatMap((image) => array(image.decisions).map((decision) => ({ image, decision })));
    elements.stagePanel.innerHTML = `
      ${stagePanelHeader("Qwen 候选与对象判断", `crop 判断外观，原色定位图辅助边界与附着；上游文本只作为可错的检索假设。`, `${decisions.length} 个决定`)}
      <div class="candidate-grid">
        ${decisions
          .map(({ image, decision }) => {
            const candidate = decision.candidate || {};
            const annotation = decision.temporary_annotation;
            const finalAnnotation = decision.final_annotation;
            const errors = array(decision.errors).map(errorValueText).filter(Boolean);
            const validity = decision.validity;
            const validityLabel =
              validity === "valid"
                ? "有效候选"
                : validity === "ignored"
                  ? "无效 / 忽略"
                  : decision.status === "failed"
                    ? "有效性判断失败"
                    : "未返回有效性";
            const validityStatus = validity === "valid" ? "passed" : validity === "ignored" ? "ignored" : decision.status || "pending";
            const validityReason = decision.validity_short_reason || errors.join("；") || "未提供有效性理由";
            const objectReason = decision.short_reason || (decision.decision ? "未提供对象决定理由" : "未形成对象身份决定");
            return `
              <article class="candidate-card">
                <div class="candidate-asset">${previewImage(
                  imageUrl(candidate.crop || candidate.crop_path || candidate.overlay || candidate.overlay_path),
                  `候选 ${decision.proposal_id}`
                )}</div>
                <div class="candidate-body">
                  <div class="card-title-row">
                    <strong class="mono">${escapeHtml(shortId(decision.proposal_id, 21))}</strong>
                    <div class="chip-list" style="margin-top:0">
                      ${decision.status ? statusPill(decision.status) : ""}
                      ${decision.raw_response ? `<a class="quiet-button" href="${escapeHtml(auditJsonUrl(decision.raw_response))}" target="_blank" rel="noopener">原始响应</a>` : ""}
                    </div>
                  </div>
                  <div class="reasoning-steps">
                    <section class="reasoning-step">
                      <div class="reasoning-step-heading">
                        <span>1 · 候选有效性</span>
                        ${statusPill(validityStatus, validityLabel)}
                      </div>
                      <p>${escapeHtml(validityReason)}</p>
                      <span class="mono-line">validity_confidence ${decision.validity_confidence == null ? "—" : `${Math.round(number(decision.validity_confidence) * 100)}%`} · ${escapeHtml(decision.validity_reason_code || "无 reason code")}</span>
                      ${errors.length ? `<ul class="reasoning-errors">${errors.map((error) => `<li>${escapeHtml(error)}</li>`).join("")}</ul>` : ""}
                    </section>
                    <section class="reasoning-step">
                      <div class="reasoning-step-heading">
                        <span>2 · 对象身份决定</span>
                        ${decision.decision ? statusPill(decision.decision) : statusPill(decision.status || "pending", "未形成决定")}
                      </div>
                      <p>${escapeHtml(objectReason)}</p>
                      <span class="mono-line">source ${escapeHtml(shortId(image.source_id, 16))} · confidence ${decision.confidence == null ? "—" : `${Math.round(number(decision.confidence) * 100)}%`}</span>
                      ${decision.object_id || decision.matched_object_id ? `<span class="chip">object <b class="mono">${escapeHtml(decision.object_id || decision.matched_object_id)}</b></span>` : ""}
                    </section>
                  </div>
                  ${
                    annotation || finalAnnotation
                      ? `<div class="annotation-compare">
                          ${annotationBox("本次观测", annotation)}
                          ${annotationBox("稳定对象卡", finalAnnotation)}
                        </div>`
                      : ""
                  }
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function annotationBox(label, annotation) {
    if (!annotation) return `<div class="annotation-box"><span>${label}</span><p>无</p></div>`;
    const title = annotation.fine_category || annotation.coarse_category || "未命名对象";
    return `<div class="annotation-box"><span>${label}</span><p><b>${escapeHtml(title)}</b><br>${escapeHtml(annotation.description || "—")}</p></div>`;
  }

  function stagePanelHeader(title, description, count) {
    return `
      <div class="stage-panel-header">
        <div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></div>
        ${statusPill("info", count)}
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
    const counts = memory.counts || {};
    const objects = array(memory.objects);
    const candidates = array(memory.candidates || memory.proposals);
    const initialized = Boolean(
      memory.initialized ?? memory.database_exists ?? (objects.length > 0 || Object.keys(counts).length > 0)
    );
    const observationCount = number(counts.observations, objects.reduce((sum, object) => sum + array(object.observations).length, 0));
    elements.memoryOverview.innerHTML = `
      <article><span>活跃对象</span><strong>${formatInteger(counts.active_objects ?? objects.length)}</strong><small>长期档案</small></article>
      <article><span>观测记录</span><strong>${formatInteger(observationCount)}</strong><small>跨视角证据</small></article>
      <article><span>正式候选</span><strong>${formatInteger(counts.proposals ?? candidates.length)}</strong><small>含过滤与待定</small></article>
      <article><span>数据库状态</span><strong>${initialized ? "已连接" : "未初始化"}</strong><small>memory.sqlite</small></article>
    `;

    if (!initialized) {
      elements.memoryContent.innerHTML = emptyState("当前还没有对象记忆", "首次实验会自动创建 SQLite 与对象资产。", "□");
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
      elements.memoryContent.innerHTML = emptyState("数据库已建立，但还没有对象卡", "请核对候选是否被忽略、不确定或运行失败。", "□");
      return;
    }
    elements.memoryContent.innerHTML = `<div class="object-grid">${objects.map(objectCard).join("")}</div>`;
  }

  function objectCard(object) {
    const observations = array(object.observations);
    const cover = object.representative_view || object.cover_path || observations[observations.length - 1]?.crop_path;
    const colors = array(object.color || object.colors);
    const materials = array(object.material || object.materials);
    return `
      <article class="object-card">
        <div class="object-card-top">
          <div class="object-cover">${previewImage(imageUrl(cover), `对象 ${object.id}`)}</div>
          <div class="object-card-info">
            <div class="card-title-row">
              <h3>${escapeHtml(object.fine_category || object.coarse_category || "未命名对象")}</h3>
              ${statusPill(object.status || "completed", object.status === "archived" ? "已归档" : `${observations.length} 观测`)}
            </div>
            <span class="mono-line" title="${escapeHtml(object.id || "")}">${escapeHtml(object.id || "—")}</span>
            <div class="chip-list">
              ${colors.map((value) => `<span class="chip">颜色 · <b>${escapeHtml(value)}</b></span>`).join("")}
              ${materials.map((value) => `<span class="chip">材质 · <b>${escapeHtml(value)}</b></span>`).join("")}
              ${object.shape ? `<span class="chip">形状 · <b>${escapeHtml(object.shape)}</b></span>` : ""}
              ${object.annotation_confidence == null ? "" : `<span class="chip">标注置信 · <b>${Math.round(number(object.annotation_confidence) * 100)}%</b></span>`}
            </div>
            <p class="object-description">${escapeHtml(object.description || "暂无稳定对象描述")}</p>
          </div>
        </div>
        <div class="observation-timeline" aria-label="${escapeHtml(object.id || "对象")} 的观测时间线">
          ${
            observations.length
              ? observations
                  .map(
                    (observation, index) => `
                      <div class="observation-item" title="${escapeHtml(observation.description || observation.id || "")}">
                        <div class="observation-image">${previewImage(imageUrl(observation.crop_path || observation.crop), `观测 ${observation.id || index + 1}`)}</div>
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
      elements.memoryContent.innerHTML = emptyState("暂无候选血缘", "SAM3 产生候选并写入数据库后，这里会连接 source、proposal、decision 与 object。", "⌁");
      return;
    }
    elements.memoryContent.innerHTML = `
      <div class="table-wrap">
        <table class="lineage-table">
          <thead><tr><th>候选</th><th>source</th><th>文本提示</th><th>分数</th><th>候选状态</th><th>Qwen 决策</th><th>对象</th><th>资产</th><th>审计</th></tr></thead>
          <tbody>
            ${candidates
              .map((candidate) => {
                const asset = candidate.crop_path || candidate.overlay_path;
                return `
                  <tr>
                    <td class="mono" title="${escapeHtml(candidate.id || candidate.proposal_id || "")}">${escapeHtml(shortId(candidate.id || candidate.proposal_id, 17))}</td>
                    <td class="mono" title="${escapeHtml(candidate.source_image_id || "")}">${escapeHtml(shortId(candidate.source_image_id, 14))}</td>
                    <td>${escapeHtml(candidate.prompt || candidate.sam_text_prompt || "—")}</td>
                    <td>${candidate.score == null ? "—" : number(candidate.score).toFixed(3)}</td>
                    <td>${statusPill(candidate.status || "pending")}</td>
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
    `;
  }

  function reportMetrics(report) {
    const images = array(report?.images);
    const run = report?.run || {};
    const sourceCounts = run.source_counts || {};
    const proposalCounts = run.proposal_counts || {};
    const decisions = run.decision_counts || {};
    const sceneTargets = images.reduce((sum, image) => sum + number(image.scene_guidance?.target_count), 0);
    const rawCandidates = images.reduce((sum, image) => sum + number(image.sam?.above_confidence_threshold_candidates), 0);
    const zeroPrompts = images.reduce((sum, image) => sum + array(image.sam?.zero_candidate_prompts).length, 0);
    return {
      inputs: images.length,
      uniqueSources: Object.values(sourceCounts).reduce((sum, value) => sum + number(value), 0),
      duplicates: number(run.duplicate_sources_skipped, images.filter((image) => image.duplicate).length),
      sceneTargets,
      rawCandidates,
      kept: number(proposalCounts.decided) + number(proposalCounts.pending) + number(proposalCounts.failed),
      filtered: number(proposalCounts.filtered),
      newCount: number(decisions.new),
      existingCount: number(decisions.existing),
      ignoredCount: number(decisions.ignored),
      uncertainCount: number(decisions.uncertain),
      observations: number(run.observations_added),
      objects: number(run.active_objects_total),
      zeroPrompts,
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
    elements.summaryContent.innerHTML = `
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
    `;
  }

  function renderSummary() {
    const report = state.report;
    if (!report) {
      const terminalFailure = ["failed", "interrupted"].includes(state.run.status);
      if (terminalFailure) {
        renderFailedRunSummary(state.run);
        return;
      }
      elements.summaryStatus.className = "status-pill neutral";
      elements.summaryStatus.textContent = "暂无报告";
      elements.summaryContent.innerHTML = emptyState("完成一次实验后生成摘要", "摘要会覆盖输入、目标、候选、四类决策、对象、观测、耗时和错误。", "∴");
      return;
    }
    const metrics = reportMetrics(report);
    const status = report.status || "failed";
    elements.summaryStatus.className = `status-pill ${statusClass(status)}`;
    elements.summaryStatus.textContent = statusLabels[status] || status;

    const qwen = report.models?.qwen || {};
    const sam = report.models?.sam3 || {};
    const errors = reportErrorMessages(report);
    const checks = Object.entries(report.checks || {});
    const demoCoverage = Object.entries(report.demo_coverage || {});
    const serverNarrative = state.serverSummary?.narrative || state.serverSummary?.headline;
    const verdict = status === "passed" ? "流程结构通过" : status === "completed_with_errors" ? "完成但存在错误" : "本次流程未通过";
    elements.summaryContent.innerHTML = `
      <div class="summary-hero">
        <article class="summary-verdict">
          <div>
            <p class="eyebrow">RUN VERDICT</p>
            <h3>${escapeHtml(serverNarrative || verdict)}</h3>
            <p>run_id · <span class="mono">${escapeHtml(report.run?.run_id || "—")}</span></p>
          </div>
          <div class="summary-warning">passed 只表示程序与结构检查通过；物体召回、完整粒度、颜色和身份归并仍需人工逐图复核。</div>
        </article>
        <div class="funnel-grid">
          ${summaryMetric("输入文件", metrics.inputs, `${metrics.duplicates} 个重复副本`)}
          ${summaryMetric("首轮目标", metrics.sceneTargets, "Qwen 提出的物体概念")}
          ${summaryMetric("阈值上候选", metrics.rawCandidates, `${metrics.zeroPrompts} 条提示为 0 候选`)}
          ${summaryMetric("保留候选", metrics.kept, `${metrics.filtered} 个被脚本过滤`)}
          ${summaryMetric("对象决定", metrics.newCount + metrics.existingCount + metrics.ignoredCount + metrics.uncertainCount, `${metrics.newCount} new · ${metrics.existingCount} existing`)}
          ${summaryMetric("长期记忆", metrics.objects, `${metrics.observations} 条观测`)}
        </div>
      </div>
      <div class="summary-details">
        <article class="detail-panel">
          <h3>耗时与模型</h3>
          <div class="timing-grid">
            ${timingMetric("总墙钟", state.run.elapsed_seconds, "Web 运行记录")}
            ${timingMetric("Qwen 加载", qwen.model_load_seconds, `${number(qwen.load_count)} 次驻留`)}
            ${timingMetric("Qwen 推理", qwen.inference_seconds, `${number(qwen.total_calls)} 次调用`)}
            ${timingMetric("SAM3 推理", sam.inference_seconds, `峰值 ${number(sam.peak_memory_mib).toFixed(0)} MiB`)}
          </div>
        </article>
        <article class="detail-panel">
          <h3>结构检查</h3>
          <ul class="check-list">
            ${
              checks.length
                ? checks.map(([name, ok]) => `<li class="${ok ? "ok" : "bad"}"><span><b>${escapeHtml(name)}</b> · ${ok ? "通过" : "未通过"}</span></li>`).join("")
                : '<li class="bad"><span>报告没有结构检查字段</span></li>'
            }
            ${demoCoverage.map(([name, ok]) => `<li class="${ok ? "ok" : "bad"}"><span><b>demo · ${escapeHtml(name)}</b> · ${ok ? "覆盖" : "未覆盖"}</span></li>`).join("")}
          </ul>
        </article>
      </div>
      <div class="summary-details">
        <article class="detail-panel">
          <h3>四类决定</h3>
          <div class="chip-list">
            ${statusPill("new", `new ${metrics.newCount}`)}
            ${statusPill("existing", `existing ${metrics.existingCount}`)}
            ${statusPill("ignored", `ignored ${metrics.ignoredCount}`)}
            ${statusPill("uncertain", `uncertain ${metrics.uncertainCount}`)}
          </div>
        </article>
        <article class="detail-panel">
          <h3>错误与人工复核提示</h3>
          <ul class="error-list">
            ${
              errors.length
                ? errors.map((error) => `<li><span>${escapeHtml(error)}</span></li>`).join("")
                : '<li style="background:var(--green-soft)"><span>报告未记录外部错误；仍需人工审查语义效果。</span></li>'
            }
          </ul>
        </article>
      </div>
    `;
  }

  function summaryMetric(label, value, hint) {
    return `<article class="funnel-card"><span>${escapeHtml(label)}</span><strong>${formatInteger(value)}</strong><small>${escapeHtml(hint)}</small></article>`;
  }

  function timingMetric(label, seconds, hint) {
    return `<div class="timing-card"><span>${escapeHtml(label)}</span><strong>${seconds == null ? "—" : formatSeconds(seconds)}</strong><small>${escapeHtml(hint)}</small></div>`;
  }

  function emptyState(title, description, symbol) {
    return `<div class="empty-state"><span class="empty-symbol" aria-hidden="true">${escapeHtml(symbol)}</span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p></div>`;
  }

  async function uploadFiles(fileList) {
    const files = [...fileList];
    if (!files.length || isRunningStatus(state.run.status)) return;
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
    if (isRunningStatus(state.run.status)) return;
    const confirmed = window.confirm(`确定从实验输入中删除“${path}”吗？\n\n此操作只影响 data/input，不会回删已有对象记忆。`);
    if (!confirmed) return;
    try {
      const payload = await apiJson(`/api/inputs?path=${encodeURIComponent(path)}`, { method: "DELETE" });
      toast(payload.message || "输入图片已删除");
      await refreshInputs();
    } catch (error) {
      toast(`删除失败：${error.message}`, "error");
    }
  }

  async function startRun() {
    if (isRunningStatus(state.run.status) || !state.inputSummary.total) return;
    const confirmed = window.confirm(
      `将使用当前 ${state.inputSummary.total} 个输入文件启动完整端到端实验。\n\n运行期间会锁定输入，并顺序加载 Qwen、SAM3、Qwen。是否继续？`
    );
    if (!confirmed) return;
    elements.startRunButton.disabled = true;
    try {
      const payload = await apiJson("/api/runs", {
        method: "POST",
        body: JSON.stringify({ validate_demo: true }),
        headers: { "Content-Type": "application/json" },
      });
      const incomingRun = payload.state || payload.run || payload;
      Object.assign(state, {
        events: [],
        latestSequence: 0,
        report: null,
        serverSummary: null,
        lastDataRefresh: 0,
      });
      state.run = { ...incomingRun, status: payload.status || incomingRun.status || "starting" };
      renderSummary();
      toast("实验已启动，页面将持续显示真实阶段事件");
      await refreshRun();
    } catch (error) {
      toast(`无法启动实验：${error.message}`, "error");
      renderControlAvailability();
    }
  }

  function bindEvents() {
    elements.uploadButton.addEventListener("click", () => elements.fileInput.click());
    elements.fileInput.addEventListener("change", () => uploadFiles(elements.fileInput.files));
    elements.dropZone.addEventListener("click", () => {
      if (!isRunningStatus(state.run.status)) elements.fileInput.click();
    });
    elements.dropZone.addEventListener("keydown", (event) => {
      if ((event.key === "Enter" || event.key === " ") && !isRunningStatus(state.run.status)) {
        event.preventDefault();
        elements.fileInput.click();
      }
    });
    ["dragenter", "dragover"].forEach((name) =>
      elements.dropZone.addEventListener(name, (event) => {
        event.preventDefault();
        if (!isRunningStatus(state.run.status)) elements.dropZone.classList.add("dragging");
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

    elements.inputGrid.addEventListener("click", (event) => {
      const button = event.target.closest("[data-delete-input]");
      if (button) deleteInput(button.dataset.deleteInput);
    });

    document.querySelectorAll("[data-stage-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedStage = button.dataset.stageTab;
        document.querySelectorAll("[data-stage-tab]").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
        renderIntermediates();
      });
    });

    document.querySelectorAll("[data-memory-view]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedMemoryView = button.dataset.memoryView;
        document.querySelectorAll("[data-memory-view]").forEach((item) => item.classList.toggle("active", item === button));
        renderMemory();
      });
    });

    elements.stagePanel.addEventListener("click", (event) => {
      const button = event.target.closest("[data-asset-url]");
      if (!button) return;
      const container = button.closest(".candidate-asset");
      const image = container?.querySelector(":scope > img");
      if (image) image.src = button.dataset.assetUrl;
      container?.querySelectorAll(".asset-toggle").forEach((item) => item.classList.toggle("active", item === button));
    });
  }

  async function initialize() {
    bindEvents();
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
