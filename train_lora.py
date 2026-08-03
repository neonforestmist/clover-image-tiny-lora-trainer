#!/usr/bin/env python3
"""Train a Clover Image Tiny style LoRA locally.

This is a thin, reproducible wrapper around the official Diffusers
`train_text_to_image_lora.py` example. It reads a style config from
`configs/`, downloads the pinned trainer once, and launches it through
`accelerate` on whatever device you have (CUDA, Apple MPS, or CPU).

Examples
--------
    # Dry run: print the exact accelerate command without training.
    python train_lora.py configs/monet.json --dry-run

    # Local smoke test: 5 steps on a tiny sample, no Hub push.
    python train_lora.py configs/monet.json --smoke

    # Full local run.
    python train_lora.py configs/monet.json

    # Full run and push the adapter to the Hub when finished.
    python train_lora.py configs/monet.json --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINERS_DIR = ROOT / ".trainers"

# Pinned to the Diffusers release the Clover LoRAs were trained with so a local
# run reproduces the published adapters bit-for-bit given the same hardware.
DIFFUSERS_VERSION = "v0.39.0"
TRAINER_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    f"{DIFFUSERS_VERSION}/examples/text_to_image/train_text_to_image_lora.py"
)

BASE_MODEL = "neonforestmist/Clover-Image-Tiny"
SEED = 20260730


def accelerate_command() -> list[str]:
    """Run Accelerate from this trainer's environment, never another install."""
    candidates = [
        ROOT / ".venv" / "bin" / "accelerate",
        ROOT / ".venv" / "Scripts" / "accelerate.exe",
        Path(sys.executable).resolve().parent / "accelerate",
        Path(sys.executable).resolve().parent / "accelerate.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]
    executable = shutil.which("accelerate")
    if executable:
        return [executable]
    return [sys.executable, "-m", "accelerate.commands.accelerate_cli"]


def load_config(path: Path) -> dict:
    with path.open() as handle:
        config = json.load(handle)
    required = {"style", "dataset", "trigger", "validation_prompt"}
    missing = required - config.keys()
    if missing:
        raise SystemExit(f"{path} is missing required keys: {sorted(missing)}")
    return config


def fetch_trainer() -> Path:
    """Download the pinned official trainer once and cache it locally."""
    TRAINERS_DIR.mkdir(exist_ok=True)
    trainer = TRAINERS_DIR / f"train_text_to_image_lora_{DIFFUSERS_VERSION}.py"
    if not trainer.exists():
        print(f"Fetching pinned trainer {DIFFUSERS_VERSION} -> {trainer}")
        with urllib.request.urlopen(TRAINER_URL) as response:  # noqa: S310
            trainer.write_bytes(response.read())
    return trainer


def build_command(
    trainer: Path,
    config: dict,
    *,
    base_model: str,
    smoke: bool,
    push_to_hub: bool,
) -> list[str]:
    output_dir = config.get("output_dir", f"outputs/{config['style']}-lora")
    steps = 5 if smoke else int(config.get("max_train_steps", 1000))
    warmup = 0 if smoke else int(config.get("warmup_steps", 100))

    workers = int(config.get("dataloader_num_workers", 2))
    # Diffusers defines its image preprocessing closure inside main(). Python's
    # spawn mode on macOS cannot pickle that closure for worker processes.
    if platform.system() == "Darwin":
        workers = 0

    command = accelerate_command() + [
        "launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        str(trainer),
        "--pretrained_model_name_or_path",
        base_model,
        "--dataset_name",
        config["dataset"],
        "--image_column",
        config.get("image_column", "image"),
        "--caption_column",
        config.get("caption_column", "text"),
        "--resolution",
        str(config.get("resolution", 512)),
        "--random_flip",
        "--train_batch_size",
        str(config.get("train_batch_size", 1)),
        "--gradient_accumulation_steps",
        str(config.get("gradient_accumulation_steps", 1)),
        "--max_train_steps",
        str(steps),
        "--learning_rate",
        str(config.get("learning_rate", 1e-4)),
        "--lr_scheduler",
        config.get("lr_scheduler", "cosine"),
        "--lr_warmup_steps",
        str(warmup),
        "--snr_gamma",
        str(config.get("snr_gamma", 5.0)),
        "--rank",
        str(config.get("rank", 16)),
        "--mixed_precision",
        config.get("mixed_precision", "fp16"),
        "--dataloader_num_workers",
        str(workers),
        "--seed",
        str(config.get("seed", SEED)),
        "--output_dir",
        output_dir,
        "--checkpointing_steps",
        str(config.get("checkpointing_steps", 250)),
        "--checkpoints_total_limit",
        "2",
        "--report_to",
        "tensorboard",
    ]

    # Rebuilding the full inference pipeline every epoch can push an Apple
    # Silicon machine into many gigabytes of swap. Keep the training process
    # bounded on macOS; preview generation can run separately after training.
    validation_enabled = bool(
        config.get("enable_validation", platform.system() != "Darwin")
    )
    if validation_enabled:
        command += [
            "--validation_prompt",
            config["validation_prompt"],
            "--num_validation_images",
            "4",
        ]

    resume = config.get("resume_from_checkpoint")
    output_path = Path(output_dir).expanduser()
    if not smoke and resume is None and output_path.is_dir():
        resume = "latest" if any(output_path.glob("checkpoint-*")) else None
    if not smoke and resume:
        command += ["--resume_from_checkpoint", str(resume)]

    # A local imagefolder dataset needs `--train_data_dir`; a Hub id uses
    # `--dataset_name` (already added above).
    if Path(config["dataset"]).exists():
        command[command.index(config["dataset"]) - 1] = "--train_data_dir"

    if smoke:
        command += ["--max_train_samples", "4"]
    if push_to_hub:
        if not config.get("hub_model_id"):
            raise SystemExit("--push-to-hub requires 'hub_model_id' in the config")
        command += ["--push_to_hub", "--hub_model_id", config["hub_model_id"]]

    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a style config JSON")
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base checkpoint to adapt (default: %(default)s)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="5-step run on a 4-image sample; verifies the pipeline end to end",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the finished adapter to the config's hub_model_id",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the accelerate command and exit without training",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = ROOT / ".trainers" / "placeholder" if args.dry_run else fetch_trainer()
    if args.dry_run and not trainer.exists():
        # Show a representative path without downloading anything.
        trainer = TRAINERS_DIR / f"train_text_to_image_lora_{DIFFUSERS_VERSION}.py"

    command = build_command(
        trainer,
        config,
        base_model=args.base_model,
        smoke=args.smoke,
        push_to_hub=args.push_to_hub,
    )

    print("Style     :", config["style"])
    print("Trigger   :", config["trigger"])
    print("Dataset   :", config["dataset"])
    print("Base model:", args.base_model)
    print("\nCommand:\n  " + " \\\n  ".join(command) + "\n")

    if args.dry_run:
        return

    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
