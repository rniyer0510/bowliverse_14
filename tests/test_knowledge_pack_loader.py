import unittest

from app.clinician.knowledge_pack import (
    DEFAULT_KNOWLEDGE_PACK_VERSION,
    clear_knowledge_pack_cache,
    load_knowledge_pack,
    validate_default_knowledge_pack,
)


class KnowledgePackLoaderTests(unittest.TestCase):
    def setUp(self):
        clear_knowledge_pack_cache()

    def test_default_pack_loads_expected_sections(self):
        pack = load_knowledge_pack(DEFAULT_KNOWLEDGE_PACK_VERSION)

        self.assertEqual(pack["pack_version"], DEFAULT_KNOWLEDGE_PACK_VERSION)
        self.assertEqual(pack["pack_id"], "actionlab_deterministic_expert")
        self.assertIn("weak_approach_build", pack["symptoms"])
        self.assertIn("soft_block_with_trunk_carry", pack["mechanisms"])
        self.assertIn("pace_created_late_and_expensively", pack["trajectories"])
        self.assertIn("stack_over_landing_leg", pack["prescriptions"])
        self.assertIn("more_stable_release_timing", pack["followup_checks"])
        self.assertIn("soft_block_trunk_carry_story", pack["render_stories"])
        self.assertIn("transfer_block_stability", pack["history_bindings"])
        self.assertIn("soft_block_leakage_bowler", pack["archetypes"])
        self.assertIn("hedges", pack["wording"])
        self.assertIn("surfaces", pack["wording"])
        self.assertIn("internal_metrics", pack["globals"])
        self.assertIn("followup_defaults", pack["globals"])
        self.assertIn("history_uncertainty", pack["globals"])
        self.assertEqual(
            pack["prescriptions"]["stack_over_landing_leg"]["review_window_type"],
            "next_3_runs",
        )

    def test_render_and_history_links_are_resolved(self):
        pack = load_knowledge_pack(DEFAULT_KNOWLEDGE_PACK_VERSION)

        mechanism = pack["mechanisms"]["soft_block_with_trunk_carry"]
        self.assertEqual(
            mechanism["render_story_ids"],
            ["soft_block_trunk_carry_story"],
        )
        self.assertIn(
            "reduced_trunk_drift_after_ffc",
            pack["trajectories"]["pace_leakage_at_block"]["followup_signals"],
        )
        self.assertEqual(
            pack["followup_checks"]["reduced_trunk_drift_after_ffc"]["history_graph_binding"],
            "transfer_block_stability",
        )
        self.assertIn(
            "soft_block_with_trunk_carry",
            pack["archetypes"]["soft_block_leakage_bowler"]["dominant_mechanisms"],
        )
        self.assertIn(
            "coach",
            pack["wording"]["surfaces"],
        )

    def test_validate_default_pack_succeeds(self):
        validate_default_knowledge_pack()


if __name__ == "__main__":
    unittest.main()
