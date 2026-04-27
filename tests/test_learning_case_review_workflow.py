import unittest
import uuid
from types import SimpleNamespace

from app.persistence.learning_cases import (
    apply_cluster_review_action,
    apply_learning_case_review_action,
    derive_cluster_status_from_case_statuses,
    review_target_status,
)


class _FakeDb:
    def __init__(self):
        self.added = []
        self.flushed = 0

    def add(self, row):
        self.added.append(row)

    def flush(self):
        self.flushed += 1


class LearningCaseReviewWorkflowTests(unittest.TestCase):
    def test_review_target_status_maps_supported_actions(self):
        self.assertEqual(review_target_status("triage"), "CLUSTERED")
        self.assertEqual(review_target_status("queue"), "QUEUED")
        self.assertEqual(review_target_status("start_review"), "UNDER_REVIEW")
        self.assertEqual(review_target_status("resolve"), "RESOLVED")
        self.assertEqual(review_target_status("reject"), "REJECTED")
        self.assertEqual(review_target_status("reopen"), "OPEN")

    def test_derive_cluster_status_from_case_statuses_returns_under_review_for_mixed_terminal_states(self):
        status = derive_cluster_status_from_case_statuses(["RESOLVED", "REJECTED"])
        self.assertEqual(status, "UNDER_REVIEW")

    def test_apply_cluster_review_action_updates_cluster_and_all_cases(self):
        db = _FakeDb()
        cluster_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        cluster_row = SimpleNamespace(
            learning_case_cluster_id=cluster_id,
            status="CLUSTERED",
            cluster_payload={"case_type": "NO_MATCH"},
            updated_at=None,
        )
        case_rows = [
            SimpleNamespace(
                learning_case_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
                status="CLUSTERED",
            ),
            SimpleNamespace(
                learning_case_id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
                status="QUEUED",
            ),
        ]

        event = apply_cluster_review_action(
            cluster_row=cluster_row,
            case_rows=case_rows,
            action="start_review",
            account_id="44444444-4444-4444-4444-444444444444",
            notes="This cluster needs manual review.",
            metadata={"target_type": "cluster"},
            db=db,
        )

        self.assertEqual(cluster_row.status, "UNDER_REVIEW")
        self.assertEqual([row.status for row in case_rows], ["UNDER_REVIEW", "UNDER_REVIEW"])
        self.assertEqual(event.action, "start_review")
        self.assertEqual(event.from_status, "CLUSTERED")
        self.assertEqual(event.to_status, "UNDER_REVIEW")
        self.assertEqual(cluster_row.cluster_payload["latest_review"]["action"], "start_review")
        self.assertEqual(db.flushed, 1)

    def test_apply_learning_case_review_action_recomputes_cluster_status_from_siblings(self):
        db = _FakeDb()
        cluster_row = SimpleNamespace(
            learning_case_cluster_id=uuid.UUID("55555555-5555-5555-5555-555555555555"),
            status="UNDER_REVIEW",
            cluster_payload={"case_type": "COACH_FEEDBACK"},
            updated_at=None,
        )
        case_a = SimpleNamespace(
            learning_case_id=uuid.UUID("66666666-6666-6666-6666-666666666666"),
            status="UNDER_REVIEW",
        )
        case_b = SimpleNamespace(
            learning_case_id=uuid.UUID("77777777-7777-7777-7777-777777777777"),
            status="CLUSTERED",
        )
        sibling_rows = [case_a, case_b]

        event = apply_learning_case_review_action(
            cluster_row=cluster_row,
            case_row=case_a,
            sibling_case_rows=sibling_rows,
            action="resolve",
            account_id="88888888-8888-8888-8888-888888888888",
            notes="One case is resolved but the cluster still has open work.",
            metadata={"target_type": "case"},
            db=db,
        )

        self.assertEqual(case_a.status, "RESOLVED")
        self.assertEqual(case_b.status, "CLUSTERED")
        self.assertEqual(cluster_row.status, "CLUSTERED")
        self.assertEqual(event.learning_case_id, case_a.learning_case_id)
        self.assertEqual(event.to_status, "RESOLVED")
        self.assertEqual(cluster_row.cluster_payload["latest_review"]["learning_case_id"], str(case_a.learning_case_id))
        self.assertEqual(db.flushed, 1)


if __name__ == "__main__":
    unittest.main()
