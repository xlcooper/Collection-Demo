from __future__ import annotations

import copy
import unittest
from typing import Any

from scripts.run_object_memory import add_demo_coverage, failure_report


def passing_report() -> dict[str, Any]:
    return {
        "status": "passed",
        "run": {
            "decision_counts": {
                "new": 2,
                "existing": 1,
                "uncertain": 0,
            },
            "proposal_counts": {
                "filtered": 0,
                "pending": 0,
                "failed": 0,
            },
            "active_objects_total": 2,
            "duplicate_sources_skipped": 1,
        },
    }


class DemoCoverageTests(unittest.TestCase):
    def test_startup_failure_uses_current_report_schema(self) -> None:
        report = failure_report(ValueError("bad configuration"))

        self.assertEqual(report["schema_version"], 7)
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("run", report)

    def test_started_run_failure_keeps_internal_report_identity(self) -> None:
        report = failure_report(
            RuntimeError("model interrupted"),
            run_id="run_20260803_200000",
        )

        self.assertEqual(report["run"]["run_id"], "run_20260803_200000")
        self.assertEqual(
            report["run_report"],
            "run_reports/run_20260803_200000.json",
        )

    def test_clean_candidate_run_can_pass(self) -> None:
        report = passing_report()

        add_demo_coverage(report)

        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(report["demo_coverage"].values()))
        self.assertEqual(
            report["demo_observations"],
            {"filtered_candidate_observed": False},
        )

    def test_each_required_coverage_item_still_gates_validation(self) -> None:
        mutations = {
            "new_objects": lambda report: report["run"]["decision_counts"].update(
                {"new": 1}
            ),
            "active_objects": lambda report: report["run"].update(
                {"active_objects_total": 1}
            ),
            "existing_match": lambda report: report["run"][
                "decision_counts"
            ].update({"existing": 0}),
            "duplicate_source": lambda report: report["run"].update(
                {"duplicate_sources_skipped": 0}
            ),
            "pending_proposal": lambda report: report["run"][
                "proposal_counts"
            ].update({"pending": 1}),
            "failed_proposal": lambda report: report["run"][
                "proposal_counts"
            ].update({"failed": 1}),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name):
                report = copy.deepcopy(passing_report())
                mutate(report)

                add_demo_coverage(report)

                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["pipeline_status"], "passed")

    def test_filter_is_observed_without_becoming_a_requirement(self) -> None:
        report = copy.deepcopy(passing_report())
        report["run"]["proposal_counts"]["filtered"] = 1
        add_demo_coverage(report)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["demo_observations"]["filtered_candidate_observed"])

    def test_pipeline_failure_cannot_be_overridden_by_coverage(self) -> None:
        report = passing_report()
        report["status"] = "completed_with_errors"

        add_demo_coverage(report)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["pipeline_status"], "completed_with_errors")


if __name__ == "__main__":
    unittest.main()
