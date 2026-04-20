import unittest
from unittest.mock import patch

from app.workers.screening.video_screen import (
    _is_competing_primary,
    run_preanalysis_screen,
)


class VideoScreenTests(unittest.TestCase):
    def test_minor_bystander_is_allowed(self):
        primary = {
            "x": 80.0,
            "y": 20.0,
            "w": 180.0,
            "h": 520.0,
            "area": 93600.0,
        }
        umpire = {
            "x": 280.0,
            "y": 160.0,
            "w": 70.0,
            "h": 180.0,
            "area": 12600.0,
        }

        self.assertFalse(_is_competing_primary(umpire, primary))

    def test_similar_sized_secondary_person_is_rejected(self):
        primary = {
            "x": 80.0,
            "y": 20.0,
            "w": 180.0,
            "h": 520.0,
            "area": 93600.0,
        }
        competing = {
            "x": 260.0,
            "y": 24.0,
            "w": 170.0,
            "h": 470.0,
            "area": 79900.0,
        }

        self.assertTrue(_is_competing_primary(competing, primary))

    def test_screening_reports_multiple_prominent_people(self):
        with patch(
            "app.workers.screening.video_screen.detect_delivery_candidates",
            return_value={
                "delivery_count": 1,
                "method": "wrist_velocity",
                "candidate_frames": [70],
            },
        ), patch(
            "app.workers.screening.video_screen.assess_primary_subject",
            return_value={
                "passed": False,
                "status": "fail",
                "frames_with_competing_primary": [30, 60],
                "frames_with_minor_people": [],
            },
        ):
            result = run_preanalysis_screen(
                video={"fps": 30.0, "total_frames": 150, "path": "/tmp/fake.mp4"},
                pose_frames=[],
                hand="R",
            )

        self.assertFalse(result["passed"])
        self.assertEqual(
            result["blocking_issues"][0]["code"],
            "multiple_prominent_people",
        )

    def test_screening_allows_minor_people_with_warning(self):
        with patch(
            "app.workers.screening.video_screen.detect_delivery_candidates",
            return_value={
                "delivery_count": 1,
                "method": "wrist_velocity",
                "candidate_frames": [70],
            },
        ), patch(
            "app.workers.screening.video_screen.assess_primary_subject",
            return_value={
                "passed": True,
                "status": "pass",
                "frames_with_competing_primary": [],
                "frames_with_minor_people": [40],
            },
        ):
            result = run_preanalysis_screen(
                video={"fps": 30.0, "total_frames": 150, "path": "/tmp/fake.mp4"},
                pose_frames=[],
                hand="R",
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["warnings"][0]["code"], "minor_bystanders_present")

    def test_screening_allows_threshold_competing_frames_with_warning(self):
        with patch(
            "app.workers.screening.video_screen.detect_delivery_candidates",
            return_value={
                "delivery_count": 1,
                "method": "wrist_velocity",
                "candidate_frames": [70],
            },
        ), patch(
            "app.workers.screening.video_screen.assess_primary_subject",
            return_value={
                "passed": True,
                "status": "pass",
                "frames_with_competing_primary": [30, 60],
                "frames_with_minor_people": [],
                "detector_frames": 6,
                "frame_threshold": 2,
            },
        ):
            result = run_preanalysis_screen(
                video={"fps": 30.0, "total_frames": 150, "path": "/tmp/fake.mp4"},
                pose_frames=[],
                hand="R",
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["warnings"][0]["code"], "primary_subject_competition_seen")

    def test_screening_prefers_raw_window_delivery_count_when_available(self):
        with patch(
            "app.workers.screening.video_screen.detect_delivery_candidates",
            return_value={
                "delivery_count": 1,
                "method": "wrist_velocity",
                "candidate_frames": [70],
            },
        ), patch(
            "app.workers.screening.video_screen.assess_primary_subject",
            return_value={
                "passed": True,
                "status": "pass",
                "frames_with_competing_primary": [],
                "frames_with_minor_people": [],
            },
        ):
            result = run_preanalysis_screen(
                video={
                    "fps": 30.0,
                    "total_frames": 240,
                    "path": "/tmp/fake.mp4",
                    "coarse_delivery_window": {
                        "available": True,
                        "method": "coarse_motion_scan",
                        "delivery_count": 2,
                        "peak_frames": [60, 170],
                    },
                },
                pose_frames=[],
                hand="R",
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"]["delivery"]["delivery_count"], 1)
        self.assertEqual(result["checks"]["delivery"]["candidate_frames"], [70])
        self.assertEqual(result["checks"]["delivery"]["raw_delivery_count"], 2)
        self.assertEqual(result["checks"]["delivery"]["raw_candidate_frames"], [60, 170])

    def test_screening_falls_back_to_raw_window_when_pose_guard_is_inconclusive(self):
        with patch(
            "app.workers.screening.video_screen.detect_delivery_candidates",
            return_value={
                "delivery_count": 0,
                "method": "wrist_velocity",
                "candidate_frames": [],
            },
        ), patch(
            "app.workers.screening.video_screen.assess_primary_subject",
            return_value={
                "passed": True,
                "status": "pass",
                "frames_with_competing_primary": [],
                "frames_with_minor_people": [],
            },
        ):
            result = run_preanalysis_screen(
                video={
                    "fps": 30.0,
                    "total_frames": 240,
                    "path": "/tmp/fake.mp4",
                    "coarse_delivery_window": {
                        "available": True,
                        "method": "subject_local_motion_scan",
                        "delivery_count": 2,
                        "peak_frames": [60, 170],
                    },
                },
                pose_frames=[],
                hand="R",
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["blocking_issues"][0]["code"], "multiple_deliveries")
        self.assertEqual(result["checks"]["delivery"]["delivery_count"], 2)
        self.assertEqual(result["checks"]["delivery"]["candidate_frames"], [60, 170])


if __name__ == "__main__":
    unittest.main()
