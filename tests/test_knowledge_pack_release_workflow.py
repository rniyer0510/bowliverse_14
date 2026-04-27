import unittest
import uuid
from types import SimpleNamespace

from app.persistence.knowledge_pack_releases import (
    apply_release_action,
    create_release_candidate,
)


class _FakeQuery:
    def __init__(self, result=None):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeDb:
    def __init__(self, existing_candidate=None):
        self.added = []
        self.flushed = 0
        self._existing_candidate = existing_candidate

    def add(self, row):
        self.added.append(row)

    def flush(self):
        self.flushed += 1

    def query(self, model):
        return _FakeQuery(self._existing_candidate)


class KnowledgePackReleaseWorkflowTests(unittest.TestCase):
    def test_create_release_candidate_requires_resolved_clusters(self):
        db = _FakeDb()
        unresolved_cluster = SimpleNamespace(
            learning_case_cluster_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            status="UNDER_REVIEW",
        )

        with self.assertRaises(ValueError):
            create_release_candidate(
                knowledge_pack_id="actionlab_deterministic_expert",
                base_pack_version="2026-04-22.v1",
                candidate_pack_version="2026-05-01.v1",
                summary="Tighten ambiguity handling.",
                created_by_account_id="22222222-2222-2222-2222-222222222222",
                cluster_rows=[unresolved_cluster],
                case_rows=[],
                change_summary={"why": "Ambiguous clusters kept recurring."},
                tests_added=["regression_case_ambiguous_block_leak"],
                reinterpret_run_ids=[],
                supersedes_pack_version="2026-04-22.v1",
                db=db,
            )

    def test_create_release_candidate_records_cluster_and_case_lineage(self):
        db = _FakeDb()
        resolved_cluster = SimpleNamespace(
            learning_case_cluster_id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
            status="RESOLVED",
        )
        case_row = SimpleNamespace(
            learning_case_id=uuid.UUID("44444444-4444-4444-4444-444444444444"),
        )

        candidate = create_release_candidate(
            knowledge_pack_id="actionlab_deterministic_expert",
            base_pack_version="2026-04-22.v1",
            candidate_pack_version="2026-05-01.v1",
            summary="Add a new contradiction rule for recurring ambiguous clusters.",
            created_by_account_id="55555555-5555-5555-5555-555555555555",
            cluster_rows=[resolved_cluster],
            case_rows=[case_row],
            change_summary={"what_changed": ["Added contradiction thresholds"]},
            tests_added=["test_ambiguous_cluster_regression_fixture"],
            reinterpret_run_ids=["run-1"],
            supersedes_pack_version="2026-04-22.v1",
            db=db,
        )

        self.assertEqual(candidate.status, "DRAFT")
        self.assertEqual(candidate.motivating_cluster_ids, ["33333333-3333-3333-3333-333333333333"])
        self.assertEqual(candidate.motivating_case_ids, ["44444444-4444-4444-4444-444444444444"])
        self.assertEqual(candidate.tests_added, ["test_ambiguous_cluster_regression_fixture"])
        self.assertEqual(db.flushed, 2)
        self.assertEqual(len(db.added), 2)

    def test_release_actions_enforce_gate_order(self):
        db = _FakeDb()
        candidate = SimpleNamespace(
            knowledge_pack_release_candidate_id=uuid.UUID("66666666-6666-6666-6666-666666666666"),
            status="DRAFT",
            current_environment=None,
            schema_validated=False,
            referential_integrity_validated=False,
            regression_suite_passed=False,
            staging_evaluation_passed=False,
            approval_granted=False,
            updated_at=None,
            updated_by_account_id=None,
            promoted_at=None,
        )

        with self.assertRaises(ValueError):
            apply_release_action(
                candidate_row=candidate,
                action="promote_staging",
                account_id="77777777-7777-7777-7777-777777777777",
                notes="Too early.",
                metadata={},
                db=db,
            )

        apply_release_action(
            candidate_row=candidate,
            action="record_schema_validation",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="Schema valid.",
            metadata={},
            db=db,
        )
        apply_release_action(
            candidate_row=candidate,
            action="record_referential_integrity",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="References valid.",
            metadata={},
            db=db,
        )
        apply_release_action(
            candidate_row=candidate,
            action="record_regression_pass",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="Regression suite passed.",
            metadata={},
            db=db,
        )
        apply_release_action(
            candidate_row=candidate,
            action="promote_dev",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="Promoted to dev.",
            metadata={},
            db=db,
        )
        apply_release_action(
            candidate_row=candidate,
            action="promote_staging",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="Promoted to staging.",
            metadata={},
            db=db,
        )
        apply_release_action(
            candidate_row=candidate,
            action="record_staging_evaluation",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="Staging evaluation passed.",
            metadata={},
            db=db,
        )
        apply_release_action(
            candidate_row=candidate,
            action="approve_production",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="Approved for production.",
            metadata={},
            db=db,
        )
        event = apply_release_action(
            candidate_row=candidate,
            action="promote_production",
            account_id="77777777-7777-7777-7777-777777777777",
            notes="Promoted to production.",
            metadata={},
            db=db,
        )

        self.assertEqual(candidate.status, "PROMOTED")
        self.assertEqual(candidate.current_environment, "production")
        self.assertTrue(candidate.schema_validated)
        self.assertTrue(candidate.referential_integrity_validated)
        self.assertTrue(candidate.regression_suite_passed)
        self.assertTrue(candidate.staging_evaluation_passed)
        self.assertTrue(candidate.approval_granted)
        self.assertEqual(event.action, "promote_production")


if __name__ == "__main__":
    unittest.main()
