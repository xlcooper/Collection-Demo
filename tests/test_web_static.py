from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "src" / "object_memory" / "web_static"


class WebStaticContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        self.javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        self.stylesheet = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")

    def test_page_explains_the_automatic_cluster_workflow(self) -> None:
        for expected in (
            "Collection Demo",
            "Qwen3-VL-8B-Instruct-FP8",
            "SAM3 0.1.0 · BF16 · 16×16 自动点网格",
            "DINOv3 ViT-B/16 · CLS + mask 内 patch",
            "SAM → DINO → Qwen 顺序驻留",
            "语义审查",
            "SQLite · 对象与观测",
            "运行完整实验",
        ):
            self.assertIn(expected, self.html)

        for stage in (
            "input",
            "sam3",
            "clustering",
            "cluster_review",
            "memory",
            "report",
        ):
            self.assertIn(f'data-flow-stage="{stage}"', self.html)

    def test_frontend_uses_only_project_local_assets(self) -> None:
        self.assertIn('href="/static/app.css"', self.html)
        self.assertIn('src="/static/app.js"', self.html)
        self.assertIn(
            'href="https://github.com/xlcooper/Collection-Demo"',
            self.html,
        )
        self.assertEqual(self.html.count("https://"), 1)
        self.assertNotIn("http://", self.html)

    def test_frontend_calls_only_fixed_web_api_routes(self) -> None:
        for route in (
            "/api/inputs",
            "/api/memories",
            "/api/runs",
            "/api/runs/current",
            "/api/results",
            "/api/memory",
            "/api/input-asset",
            "/api/memory-asset",
            "/api/audit-json",
        ):
            self.assertIn(route, self.javascript)
        self.assertNotIn("localStorage", self.javascript)
        self.assertNotIn("sessionStorage", self.javascript)

    def test_intermediate_images_merge_input_and_source_aliases(self) -> None:
        self.assertIn(
            'const inputPath = String(image?.input_path || "")',
            self.javascript,
        )
        self.assertIn('inputPath ? `input:${inputPath}` : ""', self.javascript)
        self.assertIn("const aliases = new Map();", self.javascript)
        self.assertIn("const direct = data.input_path || data.source_id", self.javascript)
        self.assertIn("function samKeptCandidatesForImage(image)", self.javascript)
        self.assertNotIn(".concat(array(image.filtered_proposals))", self.javascript)

    def test_sam_results_are_collapsed_by_source_image(self) -> None:
        for expected in (
            'class="sam-result-group"',
            'data-sam-group="${escapeHtml(groupId)}"',
            "state.openSamGroups.has(groupId)",
            "展开查看全部保留候选",
            "const candidates = samKeptCandidatesForImage(image);",
            "不受后续Qwen判断影响",
            'statusPill("info", "SAM阶段保留")',
        ):
            self.assertIn(expected, self.javascript)
        sam_card = self.javascript[
            self.javascript.index("function samCandidateCard") : self.javascript.index(
                "function restoreCandidateAssetViews"
            )
        ]
        self.assertNotIn("candidate.status", sam_card)
        self.assertNotIn("candidate.filter_reason", sam_card)
        self.assertIn(".sam-result-group[open]", self.stylesheet)

    def test_cluster_stage_explains_identifiers_and_similarity_once(self) -> None:
        for expected in (
            "clu_…",
            "prop_…",
            "CLS相似度",
            "最低 / 平均 / 最高",
            "单成员组没有可比较对象",
        ):
            self.assertIn(expected, self.javascript)
        self.assertEqual(
            self.javascript.count("同一源图的候选不会自动进入同一组"),
            1,
        )

    def test_legacy_demo_coverage_failure_is_explained_as_pipeline_success(self) -> None:
        for expected in (
            'report.pipeline_status !== "passed"',
            'failedNames[0] === "exact_duplicate_image_was_skipped"',
            'const runStatus = legacyDemoCoverageFailure ? "completed" : reportedRunStatus;',
            "legacyDemoCoverageFailure",
            "实验流程成功，旧固定 Demo 覆盖条件未满足",
            "本次输入均不重复",
        ):
            self.assertIn(expected, self.javascript)

    def test_polling_is_serial_and_evidence_refresh_is_throttled(self) -> None:
        refresh_run = self.javascript[
            self.javascript.index("async function refreshRun") : self.javascript.index(
                "async function refreshResults"
            )
        ]
        self.assertIn("pollInFlight: false", self.javascript)
        self.assertIn("if (state.pollInFlight) return;", refresh_run)
        self.assertIn("state.pollInFlight = false;", refresh_run)
        self.assertIn("const transitionedToTerminal = wasRunning && !isRunning;", refresh_run)
        self.assertIn("now - state.lastDataRefresh >= DATA_REFRESH_INTERVAL_MS", refresh_run)
        self.assertEqual(refresh_run.count("refreshInputs"), 1)

    def test_memory_library_controls_are_explicit(self) -> None:
        for expected in (
            'id="memory-library-select"',
            'id="new-memory-button"',
            'id="delete-memory-button"',
            "本次实验使用的对象记忆库",
        ):
            self.assertIn(expected, self.html)
        self.assertIn("memory_id: state.selectedMemoryId", self.javascript)
        self.assertIn("function memoryControlsLocked()", self.javascript)
        self.assertIn("state.serverCatalogLocked = Boolean(payload.locked);", self.javascript)

    def test_input_collection_has_guarded_clear_action(self) -> None:
        self.assertIn('id="clear-inputs-button"', self.html)
        self.assertIn(
            'apiJson("/api/inputs/all", { method: "DELETE" })',
            self.javascript,
        )
        self.assertIn(
            "已经生成的对象记忆、源图副本和运行报告不会受到影响",
            self.javascript,
        )
        self.assertIn(
            "elements.clearInputsButton.disabled = locked || !hasInputs;",
            self.javascript,
        )

    def test_reasoning_stage_shows_qwen_dino_and_final_decision(self) -> None:
        reasoning = self.javascript[
            self.javascript.index("function renderReasoningStage") : self.javascript.index(
                "function renderMemory()"
            )
        ]
        for expected in (
            "1 · DINOv3 聚类依据",
            "2 · 历史对象匹配",
            "3 · Qwen语义判断与最终决定",
            "cluster.qwen_review",
            "cluster.historical_visual_evidence",
            "visual.global_similarity",
            "visual.local_match_ratio",
            "cluster.final_decision",
        ):
            self.assertIn(expected, reasoning)

    def test_object_card_has_one_summary_and_observation_assets(self) -> None:
        for expected in (
            "稳定汇总描述",
            "类内区别特征",
            "品牌 / 标记",
            "部件级外观",
            "observation.mask",
            "observation.crop",
        ):
            self.assertIn(expected, self.javascript)
        self.assertNotIn("observation.description", self.javascript)

    def test_lineage_table_exposes_three_stage_identity_evidence(self) -> None:
        for expected in (
            "DINO聚类",
            "Qwen聚类假设",
            "历史DINO证据",
            "最终决定",
            "candidate.visual_evidence",
        ):
            self.assertIn(expected, self.javascript)

    def test_terminal_failure_and_nullish_fallback_contracts_remain(self) -> None:
        self.assertIn(
            'const terminalFailure = ["failed", "interrupted"].includes(runStatus);',
            self.javascript,
        )
        self.assertIn("renderFailedRunSummary(state.run);", self.javascript)
        self.assertIn(
            "memory.database_exists ?? (objects.length > 0 || Object.keys(counts).length > 0)",
            self.javascript,
        )
        self.assertNotIn("memory.database_exists ?? objects.length ||", self.javascript)


if __name__ == "__main__":
    unittest.main()
