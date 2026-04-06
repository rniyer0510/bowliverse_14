import unittest

from app.persistence.read_api import (
    _build_rating_heatmap_v2,
    _build_score_heatmap,
    _extract_rating_summary_v2,
    _extract_score_summary,
)


class ScoreHeatmapTests(unittest.TestCase):
    def test_extract_score_summary_from_clinician_scorecard(self):
        result_json = {
            "clinician": {
                "scorecard": {
                    "overall": {"score": 82, "label": "Strong base"},
                    "pillars": {
                        "balance": {"score": 79},
                        "carry": {"score": 84},
                        "body_load": {"score": 80},
                    },
                    "confidence": {"score": 68},
                }
            }
        }

        summary = _extract_score_summary(result_json)

        self.assertEqual(summary["overall"]["score"], 82)
        self.assertEqual(summary["overall"]["band"], "strong")
        self.assertEqual(summary["balance"]["score"], 79)
        self.assertEqual(summary["carry"]["band"], "strong")
        self.assertEqual(summary["confidence"]["band"], "watch")

    def test_build_score_heatmap_keeps_latest_first_cells(self):
        heatmap = _build_score_heatmap(
            [
                {
                    "run_id": "run-2",
                    "created_at": "2026-03-31T10:00:00",
                    "score_summary": {
                        "overall": {"score": 84, "band": "strong"},
                        "balance": {"score": 80, "band": "strong"},
                        "carry": {"score": 86, "band": "strong"},
                        "body_load": {"score": 82, "band": "strong"},
                        "confidence": {"score": 69, "band": "watch"},
                    },
                },
                {
                    "run_id": "run-1",
                    "created_at": "2026-03-30T10:00:00",
                    "score_summary": {
                        "overall": {"score": 74, "band": "watch"},
                        "balance": {"score": 70, "band": "watch"},
                        "carry": {"score": 76, "band": "strong"},
                        "body_load": {"score": 72, "band": "watch"},
                        "confidence": {"score": 61, "band": "watch"},
                    },
                },
            ]
        )

        self.assertEqual(heatmap["version"], "score_heatmap_v1")
        self.assertTrue(heatmap["latest_first"])
        self.assertEqual(heatmap["rows"][0]["metric"], "overall")
        self.assertEqual(heatmap["rows"][0]["cells"][0]["run_id"], "run-2")
        self.assertEqual(heatmap["rows"][0]["cells"][1]["band"], "watch")

    def test_extract_rating_summary_v2_from_clinician_payload(self):
        result_json = {
            "clinician": {
                "rating_system_v2": {
                    "overall": {"score": 77, "label": "Good base"},
                    "player_view": {
                        "metrics": {
                            "upper_body_alignment": {"score": 74},
                            "lower_body_alignment": {"score": 79},
                            "whole_body_alignment": {"score": 72},
                            "momentum_forward": {"score": 80},
                        }
                    },
                    "coach_view": {
                        "metrics": {
                            "safety": {"score": 69},
                        }
                    },
                    "confidence": {"score": 71},
                }
            }
        }

        summary = _extract_rating_summary_v2(result_json)

        self.assertEqual(summary["overall"]["score"], 77)
        self.assertEqual(summary["upper_body_alignment"]["band"], "watch")
        self.assertEqual(summary["momentum_forward"]["score"], 80)
        self.assertEqual(summary["safety"]["band"], "watch")

    def test_build_rating_heatmap_v2_keeps_latest_first_cells(self):
        heatmap = _build_rating_heatmap_v2(
            [
                {
                    "run_id": "run-2",
                    "created_at": "2026-03-31T10:00:00",
                    "rating_summary_v2": {
                        "overall": {"score": 82, "band": "strong"},
                        "upper_body_alignment": {"score": 76, "band": "strong"},
                        "lower_body_alignment": {"score": 78, "band": "strong"},
                        "whole_body_alignment": {"score": 74, "band": "watch"},
                        "momentum_forward": {"score": 84, "band": "strong"},
                        "safety": {"score": 69, "band": "watch"},
                        "confidence": {"score": 71, "band": "watch"},
                    },
                },
                {
                    "run_id": "run-1",
                    "created_at": "2026-03-30T10:00:00",
                    "rating_summary_v2": {
                        "overall": {"score": 70, "band": "watch"},
                        "upper_body_alignment": {"score": 64, "band": "watch"},
                        "lower_body_alignment": {"score": 68, "band": "watch"},
                        "whole_body_alignment": {"score": 62, "band": "watch"},
                        "momentum_forward": {"score": 73, "band": "watch"},
                        "safety": {"score": 60, "band": "watch"},
                        "confidence": {"score": 65, "band": "watch"},
                    },
                },
            ]
        )

        self.assertEqual(heatmap["version"], "rating_heatmap_v2")
        self.assertTrue(heatmap["latest_first"])
        self.assertEqual(heatmap["rows"][1]["metric"], "upper_body_alignment")
        self.assertEqual(heatmap["rows"][0]["cells"][0]["run_id"], "run-2")
        self.assertEqual(heatmap["rows"][4]["cells"][1]["band"], "watch")


if __name__ == "__main__":
    unittest.main()
