from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import trainer_gui as studio


class CloverStudioTests(unittest.TestCase):
    def test_training_recipe_builds_smoke_command(self) -> None:
        command = studio.preview_training_command(
            studio.train_lora.BASE_MODEL,
            "5-step smoke test",
            False,
            *studio.config_values("monet"),
        )

        self.assertIn("accelerate launch", command)
        self.assertIn("--max_train_steps 5", command)
        self.assertIn("--max_train_samples 4", command)
        self.assertNotIn("--push_to_hub", command)

    def test_coreml_preview_is_copyable_and_machine_independent(self) -> None:
        command = studio.preview_coreml_command(
            studio.COREML_ACTIONS[0],
            "/path/to/Clover-Image-Tiny",
            "outputs/monet-lora/Monet.safetensors",
            "coreml-models/clover-stateful",
            35,
        )

        self.assertTrue(command.startswith("./coreml/export_stateful.sh"))
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
            report = studio.coreml_readiness(
                studio.COREML_ACTIONS[0],
                str(Path(directory) / "missing-model"),
                str(Path(directory) / "missing-style.safetensors"),
                str(Path(directory) / "output"),
                35,
            )

        self.assertIn("Setup needs attention", report)
        self.assertIn("Choose a local Diffusers model folder", report)
        self.assertIn("Choose a .safetensors file", report)


if __name__ == "__main__":
    unittest.main()
