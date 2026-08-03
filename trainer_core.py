"""GUI-independent workflows for Clover LoRA training and Core ML export."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import train_lora


ROOT = Path(__file__).resolve().parent
COREML_DIR = ROOT / "coreml"
CONFIGS = {
    path.stem: path for path in sorted((ROOT / "configs").glob("*.json"))
}
STEP_PATTERN = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")
COREML_ACTIONS = (
    "Export stateful U-Net",
    "Compile for Xcode",
    "Validate parity",
)


@dataclass(frozen=True)
class Requirement:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class Artifact:
    name: str
    detail: str
    path: Path


def resolve_path(value: str) -> Path:
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def config_values(name: str) -> dict[str, Any]:
    config = train_lora.load_config(CONFIGS[name])
    return {
        "style": config["style"],
        "dataset": config["dataset"],
        "trigger": config["trigger"],
        "validation_prompt": config["validation_prompt"],
        "output_dir": config.get("output_dir", f"outputs/{config['style']}-lora"),
        "max_train_steps": int(config.get("max_train_steps", 1000)),
        "rank": int(config.get("rank", 16)),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "train_batch_size": int(config.get("train_batch_size", 1)),
        "gradient_accumulation_steps": int(
            config.get("gradient_accumulation_steps", 1)
        ),
        "mixed_precision": config.get("mixed_precision", "fp16"),
        "checkpointing_steps": int(config.get("checkpointing_steps", 250)),
        "seed": int(config.get("seed", train_lora.SEED)),
        "hub_model_id": config.get("hub_model_id", ""),
    }


def make_config(values: dict[str, Any]) -> dict[str, Any]:
    steps = int(values["max_train_steps"])
    return {
        "style": str(values["style"]).strip(),
        "dataset": str(values["dataset"]).strip(),
        "trigger": str(values["trigger"]).strip(),
        "validation_prompt": str(values["validation_prompt"]).strip(),
        "output_dir": str(values["output_dir"]).strip(),
        "max_train_steps": steps,
        "rank": int(values["rank"]),
        "learning_rate": float(values["learning_rate"]),
        "train_batch_size": int(values["train_batch_size"]),
        "gradient_accumulation_steps": int(values["gradient_accumulation_steps"]),
        "mixed_precision": str(values["mixed_precision"]),
        "checkpointing_steps": int(values["checkpointing_steps"]),
        "warmup_steps": max(steps // 10, 0),
        "snr_gamma": 5.0,
        "dataloader_num_workers": 2,
        "seed": int(values["seed"]),
        "hub_model_id": str(values.get("hub_model_id", "")).strip(),
    }


def training_command(
    config: dict[str, Any],
    base_model: str,
    mode: str,
    push_to_hub: bool,
    *,
    fetch: bool,
) -> list[str]:
    trainer = (
        train_lora.fetch_trainer()
        if fetch
        else train_lora.TRAINERS_DIR
        / f"train_text_to_image_lora_{train_lora.DIFFUSERS_VERSION}.py"
    )
    return train_lora.build_command(
        trainer,
        config,
        base_model=base_model.strip(),
        smoke=mode == "5-step smoke test",
        push_to_hub=push_to_hub,
    )


def display_command(command: list[str]) -> str:
    display: list[str] = []
    for argument in command:
        try:
            display.append(str(Path(argument).relative_to(ROOT)))
        except (TypeError, ValueError):
            display.append(argument)
    return shlex.join(display)


def training_preview(
    values: dict[str, Any],
    base_model: str,
    mode: str,
    push_to_hub: bool,
) -> str:
    return display_command(
        training_command(
            make_config(values),
            base_model,
            mode,
            push_to_hub,
            fetch=False,
        )
    )


def local_dataset_preview(root: Path) -> list[tuple[Path, str]]:
    metadata = root / "metadata.jsonl"
    rows = [
        json.loads(line)
        for line in metadata.read_text().splitlines()
        if line.strip()
    ]
    return [
        (root / row["file_name"], str(row.get("text", "")))
        for row in rows[:12]
    ]


def local_training_values(dataset_name: str) -> dict[str, Any]:
    """Derive the standard training configuration from one imagefolder dataset."""
    root = resolve_path(dataset_name)
    if not root.is_dir():
        raise ValueError("Choose a dataset folder.")
    if not (root / "images").is_dir():
        raise ValueError("The dataset folder must contain an images folder.")
    if not (root / "metadata.jsonl").is_file():
        raise ValueError("The dataset folder must contain metadata.jsonl.")

    samples = local_dataset_preview(root)
    if not samples:
        raise ValueError("metadata.jsonl does not contain any training rows.")
    missing = [path.name for path, _caption in samples if not path.is_file()]
    if missing:
        raise ValueError(f"Training image not found: {missing[0]}")

    style = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-")
    style = style or "custom-style"
    first_caption = samples[0][1].strip()
    trigger = first_caption.split(",", 1)[0].strip() or root.name.replace("-", " ")
    validation_prompt = first_caption or f"{trigger}, a small blue cat"
    return {
        "style": style,
        "dataset": str(root),
        "trigger": trigger,
        "validation_prompt": validation_prompt,
        "output_dir": f"outputs/{style}-lora",
        "max_train_steps": 1000,
        "rank": 16,
        "learning_rate": 1e-4,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "fp16",
        "checkpointing_steps": 250,
        "seed": train_lora.SEED,
        "hub_model_id": "",
    }


def dataset_preview(dataset_name: str) -> tuple[str, list[tuple[Any, str]]]:
    local = resolve_path(dataset_name)
    if local.exists():
        gallery = local_dataset_preview(local)
        return f"Loaded {len(gallery)} local training pairs.", gallery

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    gallery: list[tuple[Any, str]] = []
    for index, row in enumerate(dataset):
        if index >= 12:
            break
        gallery.append((row["image"], str(row.get("text", ""))))
    return f"Loaded {len(gallery)} streamed Hub samples.", gallery


def sample_images(output_dir: str) -> list[Path]:
    root = resolve_path(output_dir)
    if not root.exists():
        return []
    return sorted(
        root.rglob("*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:12]


def training_progress(line: str, current: int) -> int:
    matches = STEP_PATTERN.findall(line)
    if not matches:
        return current
    step, total = map(int, matches[-1])
    return current if total <= 0 else min(max(round(step / total * 100), 0), 100)


def coreml_paths(
    model_dir: str,
    style_file: str,
    output_dir: str,
) -> tuple[Path, Path, Path]:
    return (
        resolve_path(model_dir),
        resolve_path(style_file),
        resolve_path(output_dir),
    )


def stateful_package(output: Path) -> Path:
    preferred = output / "Unet.mlpackage"
    if preferred.is_dir():
        return preferred
    matches = sorted(output.glob("*_unet.mlpackage")) if output.is_dir() else []
    return matches[0] if matches else preferred


def coreml_command(
    action: str,
    model_dir: str,
    style_file: str,
    output_dir: str,
    minimum_psnr: float,
) -> list[str]:
    model, style, output = coreml_paths(model_dir, style_file, output_dir)
    package = stateful_package(output)
    if action == COREML_ACTIONS[0]:
        return [str(COREML_DIR / "export_stateful.sh"), str(model), str(output), str(style)]
    if action == COREML_ACTIONS[1]:
        return [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(package),
            str(output / "compiled"),
        ]
    if action == COREML_ACTIONS[2]:
        return [
            str(ROOT / ".venv-coreml" / "bin" / "python"),
            str(COREML_DIR / "validate_stateful_lora.py"),
            "--model-version",
            str(model),
            "--coreml-model",
            str(package),
            "--adapter-schema",
            str(output / "coreml-state-schema.json"),
            "--lora-weights",
            str(style),
            "--minimum-psnr",
            str(float(minimum_psnr)),
        ]
    raise ValueError(f"Unknown Core ML action: {action}")


def coreml_preview(
    action: str,
    model_dir: str,
    style_file: str,
    output_dir: str,
    minimum_psnr: float,
) -> str:
    return display_command(
        coreml_command(action, model_dir, style_file, output_dir, minimum_psnr)
    )


def coreml_requirements(
    action: str,
    model_dir: str,
    style_file: str,
    output_dir: str,
) -> list[Requirement]:
    model, style, output = coreml_paths(model_dir, style_file, output_dir)
    package = stateful_package(output)
    requirements: list[Requirement] = []

    def add(name: str, ok: bool, success: str, failure: str) -> None:
        requirements.append(Requirement(name, "Ready" if ok else "Missing", success if ok else failure))

    add("macOS", platform.system() == "Darwin", platform.system(), "Core ML export requires macOS")
    python_name = os.environ.get("PYTHON_BIN", "python3.11")
    python_path = shutil.which(python_name)
    add("Python 3.11", bool(python_path), python_path or "", f"{python_name} was not found")
    xcrun = shutil.which("xcrun")
    add("Xcode tools", bool(xcrun), xcrun or "", "Install Xcode command-line tools")

    if action in (COREML_ACTIONS[0], COREML_ACTIONS[2]):
        model_ok = model.is_dir() and (model / "model_index.json").is_file()
        add("Clover model", model_ok, str(model), "Choose a Diffusers folder containing model_index.json")
        style_ok = style.is_file() and style.suffix.lower() == ".safetensors"
        style_detail = f"{style.name} · {format_bytes(style.stat().st_size)}" if style_ok else "Choose a .safetensors file"
        add("Style weights", style_ok, style_detail, style_detail)

    if action in (COREML_ACTIONS[1], COREML_ACTIONS[2]):
        add("Stateful U-Net", package.is_dir(), str(package), "Run Export stateful U-Net first")
    if action == COREML_ACTIONS[2]:
        schema = output / "coreml-state-schema.json"
        add("State schema", schema.is_file(), str(schema), "coreml-state-schema.json was not found")
        converter_python = ROOT / ".venv-coreml/bin/python"
        add("Converter environment", converter_python.is_file(), str(converter_python), "Created automatically during export")

    free = shutil.disk_usage(ROOT).free
    if free >= 15 * 1024**3:
        disk_status = "Ready"
    elif free >= 8 * 1024**3:
        disk_status = "Warning"
    else:
        disk_status = "Missing"
    requirements.append(Requirement("Free disk space", disk_status, format_bytes(free)))
    return requirements


def coreml_artifacts(output_dir: str) -> list[Artifact]:
    output = resolve_path(output_dir)
    package = stateful_package(output)
    candidates = [
        package,
        output / "coreml-state-schema.json",
        output / "compiled",
    ]
    artifacts: list[Artifact] = []
    for path in candidates:
        if not path.exists():
            continue
        detail = format_bytes(path.stat().st_size) if path.is_file() else "Folder"
        artifacts.append(Artifact(path.name, detail, path))
    return artifacts


def coreml_progress(line: str, current: int) -> int:
    markers = (
        ("clone", 8),
        ("install", 18),
        ("loading", 32),
        ("tracing", 48),
        ("convert", 68),
        ("state schema", 90),
        ("psnr", 75),
        ("complete", 98),
    )
    lowered = line.lower()
    for marker, progress in markers:
        if marker in lowered:
            current = max(current, progress)
    return current
