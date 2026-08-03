from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "src" / "object_memory" / "web_static"


class WebStaticContractTests(unittest.TestCase):
    def test_single_page_contains_requested_experiment_views(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        for expected in (
            "实验输入",
            "逐阶段中间结果",
            "SQLite 对象记忆",
            "结果摘要",
            "运行完整实验",
        ):
            self.assertIn(expected, html)

        for stage in (
            "input",
            "scene_guidance",
            "sam3",
            "candidate_reasoning",
            "memory",
            "report",
        ):
            self.assertIn(f'data-flow-stage="{stage}"', html)

    def test_frontend_uses_only_project_local_assets(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn('href="/static/app.css"', html)
        self.assertIn('src="/static/app.js"', html)
        self.assertIn(
            'href="https://github.com/xlcooper/Collection-Demo"',
            html,
        )
        self.assertEqual(html.count("https://"), 1)
        self.assertNotIn("http://", html)

    def test_project_introduction_names_models_and_inference_environment(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        for expected in (
            "Collection Demo",
            "Qwen3-VL-8B-Instruct-FP8",
            "SAM3 0.1.0 · BF16 · 文本阈值 0.4",
            "AutoDL · RTX 4090 24 GB · 单卡顺序加载",
            "详细过程与阶段结果在下方展开。",
        ):
            self.assertIn(expected, html)

        self.assertNotIn("这里直接对应服务器", html)
        self.assertNotIn("data/input/", html)
        self.assertIn("默认 Demo 配置与服务器基线", html)

    def test_frontend_calls_only_fixed_web_api_routes(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        for route in (
            "/api/inputs",
            "/api/runs",
            "/api/runs/current",
            "/api/results",
            "/api/memory",
            "/api/input-asset",
            "/api/memory-asset",
            "/api/audit-json",
        ):
            self.assertIn(route, javascript)

        self.assertNotIn("localStorage", javascript)
        self.assertNotIn("sessionStorage", javascript)
        self.assertNotIn('startsWith("http://")', javascript)
        self.assertNotIn('startsWith("https://")', javascript)

    def test_intermediate_images_are_keyed_by_input_path(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            'const inputPath = String(image?.input_path || "")',
            javascript,
        )
        self.assertIn('if (inputPath) return `input:${inputPath}`;', javascript)
        self.assertIn("const direct = data.input_path || data.source_id", javascript)
        self.assertNotIn("const key = image.source_id || image.input_path", javascript)
        self.assertIn(".concat(array(image.filtered_proposals))", javascript)
        image_states = javascript[
            javascript.index("function imageStates") : javascript.index(
                "function renderIntermediates"
            )
        ]
        self.assertGreater(
            image_states.index("array(state.report?.images)"),
            image_states.index("for (const event of state.events)"),
        )

    def test_polling_is_serial_and_data_refreshes_are_throttled(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        refresh_run = javascript[
            javascript.index("async function refreshRun") : javascript.index(
                "async function refreshResults"
            )
        ]

        self.assertIn("pollInFlight: false", javascript)
        self.assertIn("if (state.pollInFlight) return;", refresh_run)
        self.assertIn("state.pollInFlight = false;", refresh_run)
        self.assertIn(
            "const transitionedToTerminal = wasRunning && !isRunning;",
            refresh_run,
        )
        self.assertIn(
            "now - state.lastDataRefresh >= DATA_REFRESH_INTERVAL_MS",
            refresh_run,
        )
        self.assertIn(
            "if (transitionedToTerminal && !initial) await refreshInputs",
            refresh_run,
        )
        self.assertEqual(refresh_run.count("refreshInputs"), 1)

    def test_polling_does_not_replace_unchanged_evidence_dom(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        refresh_memory = javascript[
            javascript.index("async function refreshMemory") : javascript.index(
                "function renderInputs"
            )
        ]
        candidate_card = javascript[
            javascript.index("function candidateCard") : javascript.index(
                "function bboxText"
            )
        ]

        self.assertIn("const renderedHtml = new WeakMap();", javascript)
        self.assertIn("if (renderedHtml.get(element) === html) return false;", javascript)
        self.assertNotIn("renderIntermediates();", refresh_memory)
        self.assertIn("renderKey === state.lastIntermediateRenderKey", javascript)
        self.assertIn("const samImages = images.filter(hasSamEvidence);", javascript)
        self.assertIn("!image.duplicate && image.source_id", javascript)
        self.assertIn("hasActiveStageSelection()", javascript)
        self.assertIn("state.pendingIntermediateRender = true;", javascript)
        self.assertIn('document.addEventListener("selectionchange"', javascript)
        self.assertIn("candidateAssetKinds: new Map()", javascript)
        self.assertIn("function restoreCandidateAssetViews", javascript)
        self.assertIn("state.candidateAssetKinds.get(card.dataset.candidateId)", javascript)
        self.assertIn(
            "state.candidateAssetKinds.set(card.dataset.candidateId, button.dataset.assetKind);",
            javascript,
        )

    def test_intermediate_stage_counts_have_explicit_units_and_purpose(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        for expected in (
            "个输入文件",
            "张实际处理图片",
            "个首轮文本目标",
            "个保留候选区域",
            "条候选判断记录",
            "识别内容完全相同的输入文件",
            "一个首轮文本目标不等于最终对象",
            "保留结果仍是候选区域，不是最终对象",
            "Qwen3-VL 先判断候选是否为完整、可独立建档的物体",
        ):
            self.assertIn(expected, javascript)

    def test_new_web_run_resets_old_cursor_and_evidence(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        refresh_run = javascript[
            javascript.index("async function refreshRun") : javascript.index(
                "async function refreshResults"
            )
        ]

        self.assertIn("const previousWebRunId = state.run.web_run_id;", refresh_run)
        self.assertIn("incomingWebRunId !== previousWebRunId", refresh_run)
        for field in ("events: []", "latestSequence: 0", "report: null", "serverSummary: null"):
            self.assertIn(field, refresh_run)
        self.assertIn('apiJson("/api/runs/current?after_sequence=0")', refresh_run)
        self.assertIn("if (runChanged) renderSummary();", refresh_run)

    def test_unavailable_result_clears_stale_report(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        refresh_results = javascript[
            javascript.index("async function refreshResults") : javascript.index(
                "async function refreshMemory"
            )
        ]

        self.assertIn("if (payload.available === false)", refresh_results)
        self.assertIn("state.report = null;", refresh_results)
        self.assertIn("state.serverSummary = null;", refresh_results)
        self.assertIn("resultWebRunId !== state.run.web_run_id", refresh_results)
        self.assertIn("events: []", refresh_results)
        self.assertIn("latestSequence: 0", refresh_results)

    def test_summary_includes_all_report_error_channels(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        for field in ("error", "progress_error", "external_errors"):
            self.assertIn(f"collect(report?.{field});", javascript)
        self.assertIn("collect(image?.error);", javascript)
        self.assertIn("collect(image?.candidate_reasoning?.errors);", javascript)
        self.assertIn("collect(decision?.error);", javascript)
        self.assertIn("collect(decision?.errors);", javascript)
        self.assertIn("const errors = reportErrorMessages(report);", javascript)

    def test_terminal_run_message_ignores_stale_progress_message(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        render_run = javascript[
            javascript.index("function renderRun") : javascript.index(
                "function renderFlow"
            )
        ]

        self.assertIn(
            'const terminalFailure = ["failed", "interrupted"].includes(runStatus);',
            render_run,
        )
        self.assertIn("const terminalMessages = terminalRunMessages(state.run);", render_run)
        self.assertIn("const message = terminalFailure ? rawMessage", render_run)
        self.assertIn("run?.exit_code ?? run?.process_error?.exit_code", javascript)
        self.assertIn("const flowStage = latestFlowStage(event);", render_run)
        self.assertIn("renderFlow(flowStage, progress, terminalFailure);", render_run)
        self.assertIn("const rawEventStage = event?.stage || state.run.stage", render_run)
        terminal_messages = javascript[
            javascript.index("function terminalRunMessages") : javascript.index(
                "function number"
            )
        ]
        self.assertNotIn("process_error?.log_tail", terminal_messages)
        self.assertNotIn("errorValueText", terminal_messages)

    def test_failed_flow_keeps_last_real_stage(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        render_flow = javascript[
            javascript.index("function renderFlow") : javascript.index(
                "function renderControlAvailability"
            )
        ]

        self.assertIn('node.classList.toggle("failed", failed && index === activeIndex);', render_flow)
        self.assertIn("index < activeIndex", render_flow)
        self.assertIn("state.events.length - 1", javascript)

    def test_missing_report_renders_process_failure_summary(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
        render_summary = javascript[
            javascript.index("function renderSummary") : javascript.index(
                "function summaryMetric"
            )
        ]

        self.assertIn("renderFailedRunSummary(state.run);", render_summary)
        self.assertIn("processErrorMessage(run.process_error)", javascript)
        self.assertIn("run?.process_error?.log_tail ?? run?.log_tail", javascript)
        self.assertNotIn("errorValueText(run.process_error)", javascript)
        self.assertIn('${escapeHtml(logTail)}', javascript)
        self.assertIn(".failure-log", stylesheet)

    def test_reasoning_shows_validity_before_object_decision(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        reasoning = javascript[
            javascript.index("function renderReasoningStage") : javascript.index(
                "function annotationBox"
            )
        ]

        for field in (
            "decision.validity",
            "decision.validity_confidence",
            "decision.validity_short_reason",
            "decision.errors",
            "decision.decision",
        ):
            self.assertIn(field, reasoning)
        self.assertLess(reasoning.index("1 · 候选有效性"), reasoning.index("2 · 对象身份决定"))

    def test_nullish_memory_fallback_is_parenthesized(self) -> None:
        javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "memory.database_exists ?? (objects.length > 0 || Object.keys(counts).length > 0)",
            javascript,
        )
        self.assertNotIn("memory.database_exists ?? objects.length ||", javascript)


if __name__ == "__main__":
    unittest.main()
