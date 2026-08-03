#!/usr/bin/env python3
"""Clover Studio: local LoRA training and stateful Core ML export."""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Iterator

import gradio as gr

import train_lora


ROOT = Path(__file__).resolve().parent
COREML_DIR = ROOT / "coreml"
CONFIGS = {
    path.stem: path for path in sorted((ROOT / "configs").glob("*.json"))
}
PROCESS_LOCK = threading.Lock()
ACTIVE_PROCESS: subprocess.Popen[str] | None = None
ACTIVE_WORKFLOW = ""
STEP_PATTERN = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")

CSS = """
:root {
  --clover: #16a34a;
  --clover-dark: #15803d;
  --ink: #17201a;
  --muted: #66736a;
  --line: #dce4de;
  --canvas: #f5f8f5;
}

.gradio-container {
  --background-fill-primary: #f5f8f5;
  --background-fill-secondary: #eef3ef;
  --block-background-fill: #ffffff;
  --block-label-background-fill: #ffffff;
  --block-label-text-color: #435047;
  --block-info-text-color: #66736a;
  --body-background-fill: #f5f8f5;
  --body-text-color: #17201a;
  --body-text-color-subdued: #66736a;
  --border-color-primary: #dce4de;
  --button-secondary-background-fill: #ffffff;
  --button-secondary-background-fill-hover: #f2f7f3;
  --button-secondary-text-color: #233128;
  --color-accent-soft: #e5f6e9;
  --input-background-fill: #f7faf8;
  --input-background-fill-focus: #ffffff;
  --input-placeholder-color: #839087;
  background:
    radial-gradient(circle at 8% 0%, rgba(22, 163, 74, .10), transparent 28rem),
    var(--canvas) !important;
  color: var(--ink);
  max-width: none !important;
}

.app-shell { max-width: 1440px; margin: 0 auto; padding: 24px 28px 56px; }
.app-header {
  display: flex; align-items: center; justify-content: space-between; gap: 24px;
  padding: 16px 0 24px; border-bottom: 1px solid var(--line); margin-bottom: 18px;
}
.brand-lockup { display: flex; align-items: center; gap: 14px; }
.brand-mark {
  width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;
  color: white; background: linear-gradient(145deg, #22c55e, #15803d);
  box-shadow: 0 8px 24px rgba(21, 128, 61, .22); font: 700 18px/1 ui-monospace;
}
.brand-title { color: var(--ink); font-size: 22px; line-height: 1.15; font-weight: 720; letter-spacing: -.025em; }
.brand-subtitle { color: var(--muted); font-size: 13px; margin-top: 3px; }
.local-pill {
  border: 1px solid #cbd8ce; border-radius: 999px; padding: 7px 11px;
  color: #435047; background: rgba(255,255,255,.72); font-size: 12px; font-weight: 650;
}
.hero {
  border: 1px solid #d5e2d8; border-radius: 22px; padding: 26px 28px;
  background: linear-gradient(115deg, rgba(255,255,255,.98), rgba(238,248,240,.92));
  box-shadow: 0 18px 50px rgba(23, 32, 26, .06); margin: 10px 0 20px;
}
.eyebrow { color: var(--clover-dark); font: 700 11px/1.2 ui-monospace; letter-spacing: .12em; text-transform: uppercase; }
.hero h1 { color: var(--ink); margin: 10px 0 8px; font-size: clamp(28px, 4vw, 46px); line-height: 1.04; letter-spacing: -.045em; }
.hero p { max-width: 760px; color: var(--muted); font-size: 16px; line-height: 1.6; margin: 0; }
.workflow-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 22px; }
.workflow-step { display: flex; align-items: center; gap: 11px; padding: 12px; border-radius: 13px; background: rgba(255,255,255,.78); border: 1px solid #e0e8e2; }
.step-number { width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center; background: #e5f6e9; color: var(--clover-dark); font-size: 12px; font-weight: 800; }
.workflow-step strong { color: var(--ink); display: block; font-size: 13px; }
.workflow-step span { display: block; color: var(--muted); font-size: 11px; margin-top: 2px; }

.workspace-tabs [role="tablist"] { gap: 8px; border: 0 !important; margin-bottom: 16px; }
.workspace-tabs [role="tab"] { border: 1px solid var(--line) !important; border-radius: 12px !important; min-height: 44px; padding-inline: 18px; font-weight: 680; }
.workspace-tabs [role="tab"].selected { background: #183f25 !important; color: #fff !important; border-color: #183f25 !important; }
.section-card {
  border: 1px solid var(--line) !important; border-radius: 18px !important;
  background: rgba(255,255,255,.88) !important; padding: 18px !important;
  box-shadow: 0 8px 30px rgba(23,32,26,.035);
}
.section-heading { margin: 0 0 14px; }
.section-heading .kicker { color: var(--clover-dark); font: 700 11px/1.2 ui-monospace; text-transform: uppercase; letter-spacing: .09em; }
.section-heading h2 { color: var(--ink); font-size: 19px; line-height: 1.25; margin: 5px 0 4px; letter-spacing: -.02em; }
.section-heading p { color: var(--muted); font-size: 13px; line-height: 1.5; margin: 0; }
.quiet-card { border: 1px dashed #cad6cc; border-radius: 14px; padding: 14px 16px; background: #f8faf8; color: var(--muted); }
.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; margin: 4px 0 14px; }
.metric { border: 1px solid #dbe5dd; border-radius: 13px; padding: 12px; background: #fbfdfb; }
.metric strong { display: block; color: var(--ink); font-size: 14px; }
.metric span { color: var(--muted); font-size: 11px; }
.status-card { border-radius: 13px; padding: 13px 15px; border: 1px solid #dce5de; background: #f8faf8; }
.status-card strong { display: block; font-size: 13px; margin-bottom: 3px; }
.status-card span { color: var(--muted); font-size: 12px; }
.status-card.running { background: #eff8f1; border-color: #b8ddc1; }
.status-card.success { background: #ecf9ef; border-color: #a7dcb4; }
.status-card.error { background: #fff2f2; border-color: #efc0c0; }
.check-list { display: grid; gap: 7px; margin-top: 8px; }
.check { display: grid; grid-template-columns: 9px 1fr; gap: 9px; align-items: start; font-size: 12px; color: var(--muted); }
.check-dot { width: 8px; height: 8px; border-radius: 50%; background: #c5cec7; margin-top: 5px; }
.check.ok .check-dot { background: #22a447; }
.check.warn .check-dot { background: #d18b16; }
.check.bad .check-dot { background: #d64545; }
.check b { color: var(--ink); }
.artifact-list { font-size: 12px; line-height: 1.7; color: var(--muted); }
.artifact-list code { color: var(--ink); background: #edf2ee; padding: 2px 5px; border-radius: 5px; }
.footer-note { text-align: center; color: var(--muted); font-size: 11px; margin-top: 24px; }
.gradio-container footer { display: none !important; }
button.primary-action { min-height: 46px !important; font-weight: 720 !important; }
button.secondary-action { min-height: 46px !important; font-weight: 650 !important; }
.gradio-container button.secondary { color: #233128 !important; }
.console textarea, .command-box textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important; font-size: 12px !important; }

@media (max-width: 760px) {
  .app-shell { padding: 12px 12px 36px; }
  .app-header { align-items: flex-start; }
  .local-pill { display: none; }
  .hero { padding: 20px; }
  .workflow-strip, .metric-row { grid-template-columns: 1fr; }
}
"""


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


def status_card(title: str, detail: str, tone: str = "idle") -> str:
    return (
        f'<div class="status-card {html.escape(tone)}">'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(detail)}</span></div>"
    )


def section_heading(step: str, title: str, detail: str) -> str:
    return (
        '<div class="section-heading">'
        f'<div class="kicker">{html.escape(step)}</div>'
        f"<h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(detail)}</p></div>"
    )


def config_values(name: str) -> tuple:
    config = train_lora.load_config(CONFIGS[name])
    return (
        config["style"],
        config["dataset"],
        config["trigger"],
        config["validation_prompt"],
        config.get("output_dir", f"outputs/{config['style']}-lora"),
        int(config.get("max_train_steps", 1000)),
        int(config.get("rank", 16)),
        float(config.get("learning_rate", 1e-4)),
        int(config.get("train_batch_size", 1)),
        int(config.get("gradient_accumulation_steps", 1)),
        config.get("mixed_precision", "fp16"),
        int(config.get("checkpointing_steps", 250)),
        int(config.get("seed", train_lora.SEED)),
        config.get("hub_model_id", ""),
    )


def make_config(
    style: str,
    dataset: str,
    trigger: str,
    validation_prompt: str,
    output_dir: str,
    max_train_steps: float,
    rank: float,
    learning_rate: float,
    train_batch_size: float,
    gradient_accumulation_steps: float,
    mixed_precision: str,
    checkpointing_steps: float,
    seed: float,
    hub_model_id: str,
) -> dict:
    return {
        "style": style.strip(),
        "dataset": dataset.strip(),
        "trigger": trigger.strip(),
        "validation_prompt": validation_prompt.strip(),
        "output_dir": output_dir.strip(),
        "max_train_steps": int(max_train_steps),
        "rank": int(rank),
        "learning_rate": float(learning_rate),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "mixed_precision": mixed_precision,
        "checkpointing_steps": int(checkpointing_steps),
        "warmup_steps": max(int(max_train_steps) // 10, 0),
        "snr_gamma": 5.0,
        "dataloader_num_workers": 2,
        "seed": int(seed),
        "hub_model_id": hub_model_id.strip(),
    }


def training_command(
    config: dict,
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


def preview_training_command(
    base_model: str,
    mode: str,
    push_to_hub: bool,
    *fields,
) -> str:
    try:
        command = training_command(
            make_config(*fields),
            base_model,
            mode,
            push_to_hub,
            fetch=False,
        )
        display_command = []
        for argument in command:
            try:
                display_command.append(str(Path(argument).relative_to(ROOT)))
            except (TypeError, ValueError):
                display_command.append(argument)
        return shlex.join(display_command)
    except Exception as error:  # noqa: BLE001
        return f"Could not build command: {error}"


def local_dataset_preview(root: Path) -> list[tuple[str, str]]:
    metadata = root / "metadata.jsonl"
    rows = [
        json.loads(line)
        for line in metadata.read_text().splitlines()
        if line.strip()
    ]
    return [
        (str(root / row["file_name"]), row.get("text", ""))
        for row in rows[:12]
    ]


def preview_dataset(
    dataset_name: str,
    image_column: str = "image",
    caption_column: str = "text",
) -> tuple[str, list]:
    try:
        local = resolve_path(dataset_name)
        if local.exists():
            gallery = local_dataset_preview(local)
            return f"Loaded {len(gallery)} local training pairs.", gallery

        # Keep startup light; the Hub client is needed only when this button is used.
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="train", streaming=True)
        gallery = []
        for index, row in enumerate(dataset):
            if index >= 12:
                break
            gallery.append((row[image_column], str(row.get(caption_column, ""))))
        return f"Loaded {len(gallery)} streamed Hub samples.", gallery
    except Exception as error:  # noqa: BLE001
        return f"Dataset preview failed: {error}", []


def sample_images(output_dir: str) -> list[tuple[str, str]]:
    root = resolve_path(output_dir)
    if not root.exists():
        return []
    images = sorted(
        root.rglob("*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:12]
    return [(str(path), path.name) for path in images]


def claim_process(command: list[str], workflow: str) -> subprocess.Popen[str]:
    global ACTIVE_PROCESS, ACTIVE_WORKFLOW
    with PROCESS_LOCK:
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            raise RuntimeError(f"{ACTIVE_WORKFLOW} is already running")
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        ACTIVE_WORKFLOW = workflow
        return ACTIVE_PROCESS


def release_process(process: subprocess.Popen[str]) -> None:
    global ACTIVE_PROCESS, ACTIVE_WORKFLOW
    with PROCESS_LOCK:
        if ACTIVE_PROCESS is process:
            ACTIVE_PROCESS = None
            ACTIVE_WORKFLOW = ""


def training_updates(
    base_model: str,
    mode: str,
    push_to_hub: bool,
    *fields,
) -> Iterator[tuple[str, float, str, list]]:
    try:
        config = make_config(*fields)
        command = training_command(
            config,
            base_model,
            mode,
            push_to_hub,
            fetch=True,
        )
        process = claim_process(command, "LoRA training")
    except Exception as error:  # noqa: BLE001
        yield status_card("Could not start", str(error), "error"), 0, "", []
        return

    logs = ["$ " + shlex.join(command)]
    progress = 0.0
    yield (
        status_card("Training in progress", "The live log will update below.", "running"),
        progress,
        "\n".join(logs),
        sample_images(config["output_dir"]),
    )

    assert process.stdout is not None
    for line_number, line in enumerate(process.stdout, start=1):
        logs.append(line.rstrip())
        logs = logs[-500:]
        matches = STEP_PATTERN.findall(line)
        if matches:
            current, total = map(int, matches[-1])
            if total > 0:
                progress = min(max(current / total, 0), 1)
        images = sample_images(config["output_dir"]) if line_number % 20 == 0 else gr.skip()
        yield (
            status_card("Training in progress", f"{progress:.0%} complete", "running"),
            progress,
            "\n".join(logs),
            images,
        )

    return_code = process.wait()
    release_process(process)
    if return_code == 0:
        final_status = status_card(
            "Training complete",
            "The style weights and validation samples are ready.",
            "success",
        )
    else:
        final_status = status_card(
            "Training stopped",
            f"The trainer exited with code {return_code}. Review the log.",
            "error",
        )
    yield (
        final_status,
        1.0 if return_code == 0 else progress,
        "\n".join(logs),
        sample_images(config["output_dir"]),
    )


COREML_ACTIONS = (
    "1 · Export stateful U-Net",
    "2 · Compile for Xcode",
    "3 · Validate parity",
)


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


def coreml_command(
    action: str,
    model_dir: str,
    style_file: str,
    output_dir: str,
    minimum_psnr: float,
) -> list[str]:
    model, style, output = coreml_paths(model_dir, style_file, output_dir)
    if action == COREML_ACTIONS[0]:
        return [str(COREML_DIR / "export_stateful.sh"), str(model), str(output), str(style)]
    if action == COREML_ACTIONS[1]:
        return [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(output / "Unet.mlpackage"),
            str(output / "compiled"),
        ]
    if action == COREML_ACTIONS[2]:
        return [
            str(ROOT / ".venv-coreml" / "bin" / "python"),
            str(COREML_DIR / "validate_stateful_lora.py"),
            "--model-version",
            str(model),
            "--coreml-model",
            str(output / "Unet.mlpackage"),
            "--adapter-schema",
            str(output / "coreml-state-schema.json"),
            "--lora-weights",
            str(style),
            "--minimum-psnr",
            str(float(minimum_psnr)),
        ]
    raise ValueError(f"Unknown Core ML action: {action}")


def preview_coreml_command(
    action: str,
    model_dir: str,
    style_file: str,
    output_dir: str,
    minimum_psnr: float,
) -> str:
    try:
        model = model_dir.strip()
        style = style_file.strip()
        output = output_dir.strip().rstrip("/")
        if action == COREML_ACTIONS[0]:
            command = ["./coreml/export_stateful.sh", model, output, style]
        elif action == COREML_ACTIONS[1]:
            command = [
                "xcrun",
                "coremlcompiler",
                "compile",
                f"{output}/Unet.mlpackage",
                f"{output}/compiled",
            ]
        elif action == COREML_ACTIONS[2]:
            command = [
                ".venv-coreml/bin/python",
                "coreml/validate_stateful_lora.py",
                "--model-version",
                model,
                "--coreml-model",
                f"{output}/Unet.mlpackage",
                "--adapter-schema",
                f"{output}/coreml-state-schema.json",
                "--lora-weights",
                style,
                "--minimum-psnr",
                str(float(minimum_psnr)),
            ]
        else:
            raise ValueError(f"Unknown Core ML action: {action}")
        return shlex.join(command)
    except Exception as error:  # noqa: BLE001
        return f"Could not build command: {error}"


def coreml_readiness(
    action: str,
    model_dir: str,
    style_file: str,
    output_dir: str,
    minimum_psnr: float,
) -> str:
    del minimum_psnr
    model, style, output = coreml_paths(model_dir, style_file, output_dir)
    checks: list[tuple[str, str, str]] = []

    checks.append(
        ("ok" if platform.system() == "Darwin" else "bad", "macOS", platform.system())
    )
    python_name = os.environ.get("PYTHON_BIN", "python3.11")
    python_path = shutil.which(python_name)
    checks.append(
        ("ok" if python_path else "bad", "Python 3.11", python_path or "Not found")
    )
    xcrun = shutil.which("xcrun")
    checks.append(("ok" if xcrun else "bad", "Xcode tools", xcrun or "Not found"))

    if action in (COREML_ACTIONS[0], COREML_ACTIONS[2]):
        model_ok = model.is_dir() and (model / "model_index.json").is_file()
        checks.append(
            (
                "ok" if model_ok else "bad",
                "Clover model",
                str(model) if model_ok else "Choose a local Diffusers model folder",
            )
        )
        style_ok = style.is_file() and style.suffix.lower() == ".safetensors"
        checks.append(
            (
                "ok" if style_ok else "bad",
                "Style weights",
                f"{style.name} · {format_bytes(style.stat().st_size)}"
                if style_ok
                else "Choose a .safetensors file",
            )
        )

    package = output / "Unet.mlpackage"
    schema = output / "coreml-state-schema.json"
    if action == COREML_ACTIONS[1]:
        checks.append(
            ("ok" if package.is_dir() else "bad", "Stateful U-Net", str(package))
        )
    if action == COREML_ACTIONS[2]:
        checks.extend(
            [
                ("ok" if package.is_dir() else "bad", "Stateful U-Net", str(package)),
                ("ok" if schema.is_file() else "bad", "State schema", str(schema)),
                (
                    "ok" if (ROOT / ".venv-coreml/bin/python").is_file() else "bad",
                    "Converter environment",
                    "Created automatically during export",
                ),
            ]
        )

    free = shutil.disk_usage(ROOT).free
    disk_tone = "ok" if free >= 15 * 1024**3 else "warn" if free >= 8 * 1024**3 else "bad"
    checks.append((disk_tone, "Free disk space", format_bytes(free)))

    rows = "".join(
        '<div class="check {tone}"><span class="check-dot"></span>'
        '<span><b>{label}</b> · {detail}</span></div>'.format(
            tone=html.escape(tone),
            label=html.escape(label),
            detail=html.escape(detail),
        )
        for tone, label, detail in checks
    )
    ready = all(tone != "bad" for tone, _, _ in checks)
    title = "Ready for this step" if ready else "Setup needs attention"
    tone = "success" if ready else "error"
    return (
        f'<div class="status-card {tone}"><strong>{title}</strong>'
        '<span>Each requirement is checked locally.</span>'
        f'<div class="check-list">{rows}</div></div>'
    )


def coreml_artifacts(output_dir: str) -> str:
    output = resolve_path(output_dir)
    candidates = [
        output / "Unet.mlpackage",
        output / "coreml-state-schema.json",
        output / "compiled",
    ]
    found = [path for path in candidates if path.exists()]
    if not found:
        return '<div class="quiet-card">Artifacts will appear here after export.</div>'

    rows = []
    for path in found:
        if path.is_file():
            detail = format_bytes(path.stat().st_size)
        else:
            detail = "folder"
        rows.append(f"<li><code>{html.escape(path.name)}</code> · {detail}</li>")
    return '<div class="artifact-list"><strong>Output artifacts</strong><ul>' + "".join(rows) + "</ul></div>"


def coreml_progress(line: str, current: float) -> float:
    markers = (
        ("clone", 0.08),
        ("install", 0.18),
        ("loading", 0.32),
        ("tracing", 0.48),
        ("convert", 0.68),
        ("state schema", 0.9),
        ("psnr", 0.75),
        ("complete", 0.98),
    )
    lowered = line.lower()
    for marker, progress in markers:
        if marker in lowered:
            current = max(current, progress)
    return current


def coreml_updates(
    action: str,
    model_dir: str,
    style_file: str,
    output_dir: str,
    minimum_psnr: float,
) -> Iterator[tuple[str, float, str, str]]:
    try:
        command = coreml_command(
            action, model_dir, style_file, output_dir, minimum_psnr
        )
        process = claim_process(command, "Core ML conversion")
    except Exception as error:  # noqa: BLE001
        yield (
            status_card("Could not start", str(error), "error"),
            0,
            "",
            coreml_artifacts(output_dir),
        )
        return

    logs = ["$ " + shlex.join(command)]
    progress = 0.03
    yield (
        status_card("Core ML task running", action, "running"),
        progress,
        "\n".join(logs),
        coreml_artifacts(output_dir),
    )

    assert process.stdout is not None
    for line in process.stdout:
        logs.append(line.rstrip())
        logs = logs[-500:]
        progress = coreml_progress(line, progress)
        yield (
            status_card("Core ML task running", action, "running"),
            progress,
            "\n".join(logs),
            coreml_artifacts(output_dir),
        )

    return_code = process.wait()
    release_process(process)
    if return_code == 0:
        final_status = status_card(
            "Core ML step complete",
            "The output artifacts are ready for the next step.",
            "success",
        )
    else:
        final_status = status_card(
            "Core ML step failed",
            f"The process exited with code {return_code}. Review the log.",
            "error",
        )
    yield (
        final_status,
        1.0 if return_code == 0 else progress,
        "\n".join(logs),
        coreml_artifacts(output_dir),
    )


def stop_active_process() -> str:
    with PROCESS_LOCK:
        process = ACTIVE_PROCESS
        workflow = ACTIVE_WORKFLOW
        if process is None or process.poll() is not None:
            return status_card("Nothing is running", "There is no active local process.")
        process.terminate()
    return status_card("Stop requested", f"Waiting for {workflow} to exit.", "running")


def studio_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="slate",
    )


def build_gui() -> gr.Blocks:
    first_config = next(iter(CONFIGS))
    defaults = config_values(first_config)

    with gr.Blocks(title="Clover Studio") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <header class="app-header">
                  <div class="brand-lockup">
                    <div class="brand-mark">CL</div>
                    <div><div class="brand-title">Clover Studio</div>
                    <div class="brand-subtitle">LoRA training and Core ML export</div></div>
                  </div>
                  <div class="local-pill">Runs locally · your data stays on this machine</div>
                </header>
                <section class="hero">
                  <div class="eyebrow">Clover Image Tiny</div>
                  <h1>Train a style. Ship a tiny file.</h1>
                  <p>Build a focused visual LoRA, inspect every training input, then export one
                  stateful Core ML base that can load named style files at runtime on iPhone.</p>
                  <div class="workflow-strip">
                    <div class="workflow-step"><div class="step-number">01</div><div><strong>Prepare</strong><span>Choose and inspect a dataset</span></div></div>
                    <div class="workflow-step"><div class="step-number">02</div><div><strong>Train</strong><span>Run a repeatable LoRA recipe</span></div></div>
                    <div class="workflow-step"><div class="step-number">03</div><div><strong>Export</strong><span>Validate the Core ML runtime</span></div></div>
                  </div>
                </section>
                """
            )

            with gr.Tabs(elem_classes=["workspace-tabs"]):
                with gr.Tab("Train a style", id="train"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Step 1", "Style recipe", "Start from a published recipe, then make it yours."))
                            with gr.Row():
                                config_name = gr.Dropdown(
                                    choices=list(CONFIGS), value=first_config, label="Recipe"
                                )
                                style = gr.Textbox(value=defaults[0], label="Style name")
                            dataset = gr.Textbox(value=defaults[1], label="Dataset", info="Hugging Face dataset ID or local imagefolder")
                            trigger = gr.Textbox(value=defaults[2], label="Trigger phrase")
                            validation_prompt = gr.Textbox(
                                value=defaults[3], label="Validation prompt", lines=2
                            )
                        with gr.Column(scale=7, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Dataset check", "See what the model will learn", "Stream up to 12 Hub examples or inspect a local imagefolder."))
                            preview_status = gr.Markdown("Select **Load dataset preview** to inspect the training pairs.")
                            preview_gallery = gr.Gallery(
                                label="Training pairs", columns=3, rows=2, height=350, object_fit="cover"
                            )
                            preview_button = gr.Button("Load dataset preview", elem_classes=["secondary-action"])

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Step 2", "Training plan", "Use a smoke test first, then start the full run."))
                            with gr.Row():
                                base_model = gr.Textbox(value=train_lora.BASE_MODEL, label="Base model")
                                mode = gr.Radio(
                                    ["5-step smoke test", "Full training"],
                                    value="5-step smoke test",
                                    label="Run mode",
                                )
                            with gr.Row():
                                max_steps = gr.Slider(100, 3000, value=defaults[5], step=50, label="Training steps")
                                rank = gr.Radio([4, 8, 16, 32], value=defaults[6], label="LoRA rank")
                            output_dir = gr.Textbox(value=defaults[4], label="Output directory")
                            with gr.Accordion("Advanced tuning", open=False):
                                with gr.Row():
                                    learning_rate = gr.Number(value=defaults[7], label="Learning rate")
                                    batch_size = gr.Number(value=defaults[8], label="Batch size", precision=0)
                                    accumulation = gr.Number(value=defaults[9], label="Gradient accumulation", precision=0)
                                with gr.Row():
                                    mixed_precision = gr.Dropdown(["fp16", "bf16", "no"], value=defaults[10], label="Mixed precision")
                                    checkpoint_steps = gr.Number(value=defaults[11], label="Checkpoint interval", precision=0)
                                    seed = gr.Number(value=defaults[12], label="Seed", precision=0)
                                hub_model_id = gr.Textbox(value=defaults[13], label="Hugging Face repository")
                                push_to_hub = gr.Checkbox(label="Push the finished style to Hugging Face")
                        with gr.Column(scale=5, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Step 3", "Review and run", "Nothing starts until you select Start training."))
                            gr.HTML(
                                """<div class="metric-row">
                                <div class="metric"><strong>512 × 512</strong><span>training resolution</span></div>
                                <div class="metric"><strong>Rank 16</strong><span>about 6.9 MB</span></div>
                                <div class="metric"><strong>Local</strong><span>accelerate process</span></div>
                                </div>"""
                            )
                            training_status = gr.HTML(status_card("Ready to review", "Preview the exact command before starting."))
                            training_command_preview = gr.Textbox(
                                label="Exact command", lines=6, interactive=False, elem_classes=["command-box"]
                            )
                            preview_command_button = gr.Button("Preview command")
                            with gr.Row():
                                run_button = gr.Button("Start training", variant="primary", elem_classes=["primary-action"])
                                stop_training_button = gr.Button("Stop", variant="stop", elem_classes=["secondary-action"])

                    with gr.Column(elem_classes=["section-card"]):
                        gr.HTML(section_heading("Live run", "Progress and samples", "Follow the trainer without leaving the control room."))
                        training_progress = gr.Slider(0, 1, value=0, label="Progress", interactive=False)
                        with gr.Row(equal_height=False):
                            training_logs = gr.Textbox(label="Live log", lines=18, max_lines=30, interactive=False, elem_classes=["console"])
                            samples = gr.Gallery(label="Validation samples", columns=2, height=430, object_fit="cover")

                with gr.Tab("Core ML export", id="coreml"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Step 1", "Choose local inputs", "The base stays shared; the selected LoRA defines the runtime state shapes."))
                            model_dir = gr.Textbox(
                                value="/path/to/Clover-Image-Tiny",
                                label="Clover model folder",
                                info="Local Diffusers checkout containing model_index.json",
                            )
                            style_file = gr.Textbox(
                                value="outputs/monet-lora/pytorch_lora_weights.safetensors",
                                label="Style .safetensors",
                                info="A compatible Clover LoRA produced by the training workspace",
                            )
                            coreml_output_dir = gr.Textbox(
                                value="coreml-models/clover-stateful",
                                label="Core ML output folder",
                            )
                            with gr.Row():
                                coreml_action = gr.Radio(
                                    COREML_ACTIONS,
                                    value=COREML_ACTIONS[0],
                                    label="Workflow step",
                                )
                                minimum_psnr = gr.Number(value=35.0, label="Minimum PSNR", precision=1)
                        with gr.Column(scale=5, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Step 2", "Preflight", "Check tools, inputs, generated artifacts, and free disk space."))
                            readiness = gr.HTML(status_card("Not checked yet", "Choose your paths, then run the readiness check."))
                            readiness_button = gr.Button("Check readiness", elem_classes=["secondary-action"])
                            artifacts = gr.HTML(coreml_artifacts("coreml-models/clover-stateful"))

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Step 3", "Review the operation", "The GUI calls the pinned, reproducible conversion tools in this repository."))
                            coreml_command_preview = gr.Textbox(
                                label="Command preview", lines=7, interactive=False, elem_classes=["command-box"]
                            )
                            preview_coreml_button = gr.Button("Preview command")
                            with gr.Row():
                                run_coreml_button = gr.Button("Run selected step", variant="primary", elem_classes=["primary-action"])
                                stop_coreml_button = gr.Button("Stop", variant="stop", elem_classes=["secondary-action"])
                        with gr.Column(scale=5, elem_classes=["section-card"]):
                            gr.HTML(section_heading("Architecture", "One base, many styles", "Stateful export replaces the old 648 MB-per-style model copies."))
                            gr.HTML(
                                """<div class="metric-row">
                                <div class="metric"><strong>~1.5 GB</strong><span>shared Core ML base</span></div>
                                <div class="metric"><strong>144</strong><span>LoRA state tensors</span></div>
                                <div class="metric"><strong>~6.9 MB</strong><span>each rank-16 style</span></div>
                                </div>
                                <div class="quiet-card"><strong>Three deliberate steps</strong><br>
                                Export the stateful U-Net, compile it for Xcode, then validate base and style parity at 35 dB or better.</div>"""
                            )

                    with gr.Column(elem_classes=["section-card"]):
                        gr.HTML(section_heading("Live conversion", "Progress and diagnostics", "Long downloads and conversion stages stream here in real time."))
                        coreml_status = gr.HTML(status_card("Ready", "Run the readiness check before exporting."))
                        coreml_run_progress = gr.Slider(0, 1, value=0, label="Progress", interactive=False)
                        coreml_logs = gr.Textbox(label="Core ML log", lines=20, max_lines=32, interactive=False, elem_classes=["console"])

                with gr.Tab("Quick guide", id="guide"):
                    with gr.Row(equal_height=False):
                        with gr.Column(elem_classes=["section-card"]):
                            gr.HTML(section_heading("Training", "A safe first run", "Prove the full path with five steps before committing GPU time."))
                            gr.Markdown(
                                """
1. Pick a recipe and load its dataset preview.
2. Keep **5-step smoke test** selected.
3. Preview the command, then start training.
4. Inspect the log and four validation samples.
5. Switch to **Full training** only after the smoke test succeeds.
                                """
                            )
                        with gr.Column(elem_classes=["section-card"]):
                            gr.HTML(section_heading("Core ML", "From weights to iPhone", "Export one stateful base and keep every style as a small named file."))
                            gr.Markdown(
                                """
1. Select the local Clover Diffusers folder and one trained style.
2. Run **Export stateful U-Net**.
3. Run **Compile for Xcode** against the same output folder.
4. Run **Validate parity** and require at least 35 dB.
5. Copy the compiled base and state schema to the iOS catalog; distribute styles separately.
                                """
                            )

            gr.HTML('<div class="footer-note">Apache-2.0 tools · CreativeML Open RAIL-M model derivatives · localhost by default</div>')

        training_fields = [
            style,
            dataset,
            trigger,
            validation_prompt,
            output_dir,
            max_steps,
            rank,
            learning_rate,
            batch_size,
            accumulation,
            mixed_precision,
            checkpoint_steps,
            seed,
            hub_model_id,
        ]
        config_name.change(config_values, config_name, training_fields)
        preview_button.click(preview_dataset, dataset, [preview_status, preview_gallery])
        preview_command_button.click(
            preview_training_command,
            [base_model, mode, push_to_hub, *training_fields],
            training_command_preview,
        )
        run_button.click(
            training_updates,
            [base_model, mode, push_to_hub, *training_fields],
            [training_status, training_progress, training_logs, samples],
            concurrency_limit=1,
        )
        stop_training_button.click(stop_active_process, outputs=training_status, queue=False)

        coreml_fields = [
            coreml_action,
            model_dir,
            style_file,
            coreml_output_dir,
            minimum_psnr,
        ]
        readiness_button.click(coreml_readiness, coreml_fields, readiness)
        preview_coreml_button.click(
            preview_coreml_command, coreml_fields, coreml_command_preview
        )
        run_coreml_button.click(
            coreml_updates,
            coreml_fields,
            [coreml_status, coreml_run_progress, coreml_logs, artifacts],
            concurrency_limit=1,
        )
        stop_coreml_button.click(stop_active_process, outputs=coreml_status, queue=False)
        coreml_action.change(coreml_readiness, coreml_fields, readiness)
        coreml_output_dir.change(coreml_artifacts, coreml_output_dir, artifacts)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_gui().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=studio_theme(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
