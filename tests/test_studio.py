from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trainer_core as studio
from safetensors.torch import load_file, save_file
import torch


class CloverStudioTests(unittest.TestCase):
    def test_local_dataset_loads_every_training_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "storybook-style"
            images = root / "images"
            images.mkdir(parents=True)
            rows = []
            for index in range(100):
                image = images / f"{index:04d}.png"
                image.touch()
                rows.append(
                    {
                        "file_name": f"images/{image.name}",
                        "text": f"storybook style, scene {index}",
                    }
                )
            (root / "metadata.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )

            status, gallery = studio.dataset_preview(str(root))
            values = studio.local_training_values(str(root))

        self.assertEqual(len(gallery), 100)
        self.assertEqual(status, "Loaded 100 local training pairs.")
        self.assertEqual(values["style"], "storybook-style")
        self.assertEqual(values["trigger"], "storybook style")

    def test_user_trigger_phrase_is_used_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "custom-style"
            images = root / "images"
            images.mkdir(parents=True)
            (images / "0001.png").touch()
            (root / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "file_name": "images/0001.png",
                        "text": "My Trigger, a quiet garden",
                    }
                )
                + "\n"
            )

            values = studio.local_training_values(str(root), "My Trigger")
            with self.assertRaisesRegex(ValueError, "start of every caption"):
                studio.local_training_values(str(root), "Wrong Trigger")

        self.assertEqual(values["trigger"], "My Trigger")
        self.assertEqual(values["validation_prompt"], "My Trigger, a quiet garden")

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

    def test_macos_training_uses_zero_workers_and_project_accelerate(self) -> None:
        values = studio.local_training_values("data/example-monet")
        config = studio.make_config(values)
        with patch("train_lora.platform.system", return_value="Darwin"):
            command = studio.training_command(
                config,
                studio.train_lora.BASE_MODEL,
                "5-step smoke test",
                False,
                fetch=False,
            )

        worker_index = command.index("--dataloader_num_workers")
        self.assertEqual(command[worker_index + 1], "0")
        self.assertNotEqual(command[0], "accelerate")
        self.assertEqual(command[command.index("--num_processes") + 1], "1")
        self.assertNotIn("--validation_prompt", command)

    def test_full_training_resumes_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "style-lora"
            (output / "checkpoint-500").mkdir(parents=True)
            values = studio.local_training_values("data/example-monet")
            values["output_dir"] = str(output)
            command = studio.training_command(
                studio.make_config(values),
                studio.train_lora.BASE_MODEL,
                "Full training",
                False,
                fetch=False,
            )

        resume_index = command.index("--resume_from_checkpoint")
        self.assertEqual(command[resume_index + 1], "latest")

    def test_coreml_style_package_matches_ios_repository_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pytorch_lora_weights.safetensors"
            tensors = {}
            for index in range(72):
                prefix = f"unet.block_{index}.to_q"
                tensors[f"{prefix}.lora_A.weight"] = torch.zeros(16, 32)
                tensors[f"{prefix}.lora_B.weight"] = torch.zeros(32, 16)
            save_file(tensors, source, metadata={"format": "pt"})
            output = root / "storybook-anime-coreml"

            subprocess.run(
                studio.coreml_style_command(str(source), str(output)),
                check=True,
                capture_output=True,
                text=True,
            )

            weights = output / "Storybook-Anime.safetensors"
            schema = json.loads(
                (output / "coreml-state-schema.json").read_text()
            )
            packaged = load_file(str(weights), device="cpu")
            model_card = (output / "README.md").read_text()

            self.assertEqual(len(packaged), 144)
            self.assertIn("unet.block_0.to_q.lora.down.weight", packaged)
            self.assertNotIn("unet.block_0.to_q.lora_A.weight", packaged)
            self.assertEqual(schema["state_count"], 144)
            self.assertEqual(schema["states"][0]["shape"][-2:], [1, 1])
            self.assertIn("pipeline_tag: text-to-image", model_card)
            self.assertIn("Clover-Image-Tiny-CoreML", model_card)
            self.assertTrue((output / "LICENSE").is_file())
            self.assertTrue((output / ".gitattributes").is_file())

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
