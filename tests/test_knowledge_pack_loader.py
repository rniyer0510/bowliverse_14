import unittest

from app.clinician.knowledge_pack import (
    ACTIONLAB_EXPERT_ENGINE_VERSION,
    DEFAULT_KNOWLEDGE_PACK_VERSION,
    _validate_manifest,
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
        self.assertIn("release_window_instability_under_load", pack["mechanisms"])
        self.assertIn("neck_tilt_left_bfc", pack["contributors"])
        self.assertIn("front_leg_support_score", pack["contributors"])
        self.assertIn("pace_created_late_and_expensively", pack["trajectories"])
        self.assertIn("release_window_fragility_under_load", pack["trajectories"])
        self.assertIn("stack_over_landing_leg", pack["prescriptions"])
        self.assertIn("stabilize_release_window_under_load", pack["prescriptions"])
        self.assertIn("more_stable_release_timing", pack["followup_checks"])
        self.assertIn("calmer_release_window_under_load", pack["followup_checks"])
        self.assertIn("soft_block_trunk_carry_story", pack["render_stories"])
        self.assertIn("release_window_under_load_story", pack["render_stories"])
        self.assertIn("transfer_block_stability", pack["history_bindings"])
        self.assertIn("release_window_control", pack["history_bindings"])
        self.assertIn("workable_but_leaking", pack["coach_judgments"]["chain_statuses"])
        self.assertIn("transfer_and_block", pack["coach_judgments"]["break_points"])
        self.assertIn("micro", pack["coach_judgments"]["change_size_bands"])
        self.assertIn("next_3_balls", pack["capture_templates"]["outcome_windows"])
        self.assertIn(
            "classification_before_interpretation",
            pack["architecture_principles"],
        )
        self.assertIn("glazier_wheat_2014", pack["research_sources"])
        self.assertIn("agastya_blog_2", pack["research_sources"])
        self.assertIn("evidence_soft_block_ffc", pack["knowledge_evidence"])
        self.assertIn(
            "canonical_front_leg_transfer_support",
            pack["reconciliation"]["canonical_concepts"],
        )
        self.assertIn(
            "primary_mechanism_id",
            pack["capture_templates"]["coach_review_questionnaire"],
        )
        self.assertIn(
            "prescription_id",
            pack["capture_templates"]["intervention_outcome_fields"],
        )
        self.assertIn("soft_block_leakage_bowler", pack["archetypes"])
        self.assertIn("release_fragility_bowler", pack["archetypes"])
        self.assertIn("hedges", pack["wording"])
        self.assertIn("surfaces", pack["wording"])
        self.assertIn("internal_metrics", pack["globals"])
        self.assertIn("confidence_bands", pack["globals"])
        self.assertIn("severity_bands", pack["globals"])
        self.assertIn("followup_defaults", pack["globals"])
        self.assertIn("history_uncertainty", pack["globals"])
        self.assertIn("presentation_downgrade_rules", pack["globals"])
        self.assertEqual(
            pack["prescriptions"]["stack_over_landing_leg"]["review_window_type"],
            "next_3_runs",
        )
        self.assertIn(
            "change_reaction",
            pack["prescriptions"]["stack_over_landing_leg"],
        )
        self.assertEqual(
            pack["prescriptions"]["stabilize_release_window_under_load"]["change_reaction"]["match_pressure_risk"],
            "high",
        )
        self.assertEqual(
            pack["manifest"]["min_engine_version"],
            ACTIONLAB_EXPERT_ENGINE_VERSION,
        )
        self.assertFalse(pack["manifest"]["breaking_changes"])

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
            pack["contributors"]["neck_tilt_left_bfc"]["phase"],
            "BFC",
        )
        self.assertEqual(
            pack["research_sources"]["glazier_wheat_2014"]["evidence_tier"],
            "A",
        )
        self.assertEqual(
            pack["research_sources"]["agastya_blog_2"]["source_type"],
            "architecture_reference",
        )
        self.assertEqual(
            pack["knowledge_evidence"]["evidence_stack_over_landing"]["target_type"],
            "prescription",
        )
        self.assertEqual(
            pack["architecture_principles"]["strict_validation_gate"]["principle_type"],
            "validation_phase",
        )
        self.assertEqual(
            pack["coach_judgments"]["break_points"]["transfer_and_block"]["title"],
            "Transfer and block",
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
        self.assertIn(
            "reviewer",
            pack["wording"]["surfaces"],
        )
        self.assertIn(
            "clinician",
            pack["wording"]["surfaces"],
        )

    def test_validate_default_pack_succeeds(self):
        validate_default_knowledge_pack()

    def test_manifest_rejects_newer_engine_requirement(self):
        with self.assertRaises(ValueError):
            _validate_manifest(
                DEFAULT_KNOWLEDGE_PACK_VERSION,
                {
                    "schema_version": "actionlab.knowledge_pack.manifest.v1",
                    "pack_id": "actionlab_deterministic_expert",
                    "pack_version": DEFAULT_KNOWLEDGE_PACK_VERSION,
                    "release_date": "2026-04-22",
                    "min_engine_version": "9.9.9",
                    "breaking_changes": False,
                    "supersedes": None,
                    "changelog_ref": "docs/fake.md",
                    "runtime": {
                        "static_at_runtime": True,
                        "deterministic_only": True,
                    },
                    "index": {
                        "globals": "globals.yaml",
                        "mechanism_families": "mechanism_families.yaml",
                        "symptoms": "symptoms.yaml",
                        "mechanisms": "mechanisms.yaml",
                        "contributors": "contributors.yaml",
                        "archetypes": "archetypes.yaml",
                        "trajectories": "trajectories.yaml",
                        "prescriptions": "prescriptions.yaml",
                        "followup_checks": "followup_checks.yaml",
                        "render_stories": "render_stories.yaml",
                        "history_bindings": "history_bindings.yaml",
                        "coach_judgments": "coach_judgments.yaml",
                        "capture_templates": "capture_templates.yaml",
                        "architecture_principles": "architecture_principles.yaml",
                        "research_sources": "research_sources.yaml",
                        "knowledge_evidence": "knowledge_evidence.yaml",
                        "reconciliation": "reconciliation.yaml",
                        "wording": "wording.yaml",
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
