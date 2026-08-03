from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import trainer_core as studio


class CloverStudioTests(unittest.TestCase):
    def test_local_dataset_derives_standard_training_values(self) -> None:
        values = studio.local_training_values("data/example-monet")

        self.assertEqual(values["style"], "example-monet")
        self.assertTrue(values["dataset"].endswith("data/example-monet"))
        self.assertEqual(values["output_dir"], "outputs/example-monet-lora")
        self.assertEqual(values["rank"], 16)
        self.assertIn("Monet Style", values["validation_prompt"])

        command = studio.training_preview(
            values,
            studio.train_lora.BASE_MODEL,
            "Full training",
            False,
        )
        self.assertIn("--train_data_dir data/example-monet", command)
        self.assertIn("--max_train_steps 1000", command)
        self.assertIn("--output_dir outputs/example-monet-lora", command)

    def test_training_recipe_builds_smoke_command(self) -> None:
        values = studio.config_values("monet")
        command = studio.training_preview(
            values,
            studio.train_lora.BASE_MODEL,
            "5-step smoke test",
            False,
        )

        self.assertIn("accelerate launch", command)
        self.assertIn("--max_train_steps 5", command)
        self.assertIn("--max_train_samples 4", command)
        self.assertNotIn("--push_to_hub", command)

    def test_coreml_preview_is_copyable_and_machine_independent(self) -> None:
        command = studio.coreml_preview(
            studio.COREML_ACTIONS[0],
            "/path/to/Clover-Image-Tiny",
            "outputs/monet-lora/Monet.safetensors",
            "coreml-models/clover-stateful",
            35,
        )

        self.assertTrue(command.startswith("coreml/export_stateful.sh"))
        self.assertIn("coreml-models/clover-stateful", command)
        self.assertNotIn(str(studio.ROOT), command)

    def test_coreml_runner_resolves_paths_for_subprocess(self) -> None:
        command = studio.coreml_command(
            studio.COREML_ACTIONS[2],
            "/tmp/Clover-Image-Tiny",
            "/tmp/Monet.safetensors",
            "coreml-models/test",
            38.5,
        )

        self.assertTrue(Path(command[0]).is_absolute())
        self.assertIn("--minimum-psnr", command)
        self.assertEqual(command[-1], "38.5")

    def test_preflight_reports_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = studio.coreml_requirements(
                studio.COREML_ACTIONS[0],
                str(Path(directory) / "missing-model"),
                str(Path(directory) / "missing-style.safetensors"),
                str(Path(directory) / "output"),
            )

        missing = {item.name: item.detail for item in report if item.status == "Missing"}
        self.assertIn("Clover model", missing)
        self.assertIn("Style weights", missing)
        self.assertIn("Diffusers folder", missing["Clover model"])
        self.assertIn(".safetensors", missing["Style weights"])

    def test_stateful_package_detects_exported_apple_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "Clovers2_unet.mlpackage"
            package.mkdir()

            self.assertEqual(studio.stateful_package(Path(directory)), package)


if __name__ == "__main__":
    unittest.main()
