import json
import os
import tempfile
import unittest

from scripts.ablation_runner_utils import deep_update
from scripts.run_final_combo_ablation import (
    build_planned_runs,
    build_run_manifest,
    combo_ids_for_set,
    compute_baseline_deltas,
    get_combos,
)


class FinalComboRunnerTests(unittest.TestCase):
    def test_all_combo_ids_are_unique(self):
        ids = [combo.combo_id for combo in get_combos()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_safe_combo_set_excludes_transductive_combo(self):
        safe_ids = combo_ids_for_set("safe")
        self.assertNotIn("C12_TRANSDUCTIVE_LABEL_PRIOR", safe_ids)

    def test_risky_combo_set_includes_dynamic_beta_combo(self):
        risky_ids = combo_ids_for_set("risky")
        self.assertIn("C11_DYNAMIC_BETA_FULL", risky_ids)

    def test_all_combo_set_includes_c0_through_c12(self):
        all_ids = combo_ids_for_set("all")
        expected = {f"C{i}" for i in range(13)}
        observed = {combo_id.split("_", 1)[0] for combo_id in all_ids}
        self.assertEqual(observed, expected)

    def test_transductive_flag_is_true_only_for_c12(self):
        transductive_ids = [combo.combo_id for combo in get_combos() if combo.transductive]
        self.assertEqual(transductive_ids, ["C12_TRANSDUCTIVE_LABEL_PRIOR"])

    def test_deep_update_works(self):
        base = {"A": {"B": 1, "C": 2}, "D": 3}
        overrides = {"A": {"B": 9}, "E": 4}
        updated = deep_update(base, overrides)

        self.assertEqual(updated, {"A": {"B": 9, "C": 2}, "D": 3, "E": 4})
        self.assertEqual(base, {"A": {"B": 1, "C": 2}, "D": 3})

    def test_baseline_delta_computation_works_with_missing_keys(self):
        rows = [
            {
                "dataset": "cifar100",
                "combo_id": "C0_ORIGINAL",
                "average_accuracy": 50.0,
                "final_accuracy": None,
                "performance_degradation": 10.0,
            },
            {
                "dataset": "cifar100",
                "combo_id": "C1_SAFE_PROTOTYPE",
                "average_accuracy": 52.5,
                "final_accuracy": 40.0,
                "performance_degradation": None,
            },
        ]

        updated = compute_baseline_deltas(rows)
        candidate = updated[1]
        self.assertEqual(candidate["delta_avg_vs_original"], 2.5)
        self.assertIsNone(candidate["delta_final_vs_original"])
        self.assertIsNone(candidate["delta_pd_vs_original"])

    def test_dry_run_manifest_can_be_created(self):
        combos = get_combos()[:1]
        manifest = build_run_manifest(
            timestamp="20260519_120000",
            datasets=["cifar100"],
            config="configs/trainers/bimc.yaml",
            combo_set="all",
            device="cuda",
            gpu_id="0;",
            dry_run=True,
            combos=combos,
            planned_runs=[],
            git_info={"branch": "test-4", "commit": "abc"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "run_manifest.json")
            with open(path, "w") as handle:
                json.dump(manifest, handle, indent=2)
            self.assertTrue(os.path.exists(path))

        self.assertTrue(manifest["dry_run"])
        self.assertEqual(manifest["combos"][0]["combo_id"], "C0_ORIGINAL")

    def test_combo_overrides_map_to_current_config(self):
        plans = build_planned_runs(["cifar100"], "configs/trainers/bimc.yaml", get_combos())
        errors = [error for plan in plans for error in plan["validation_errors"]]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
