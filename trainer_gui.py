#!/usr/bin/env python3
"""Local Gradio control room for Clover LoRA training."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Iterator

import gradio as gr
from datasets import load_dataset

import train_lora


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    path.stem: path for path in sorted((ROOT / "configs").glob("*.json"))
}
PROCESS_LOCK = threading.Lock()
ACTIVE_PROCESS: subprocess.Popen[str] | None = None
STEP_PATTERN = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")


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


def command_for(
    config: dict,
    base_model: str,
    mode: str,
    push_to_hub: bool,
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
        smoke=mode == "Smoke test",
        push_to_hub=push_to_hub,
    )


def preview_command(
    base_model: str,
    mode: str,
    push_to_hub: bool,
    *fields,
) -> str:
    try:
        config = make_config(*fields)
        return shlex.join(
            command_for(config, base_model, mode, push_to_hub, fetch=False)
        )
    except Exception as error:  # noqa: BLE001 - surface form errors in the UI
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
        local = Path(dataset_name).expanduser()
        if local.exists():
            gallery = local_dataset_preview(local.resolve())
            return f"Loaded {len(gallery)} local training pairs.", gallery

        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )
        gallery = []
        for index, row in enumerate(dataset):
            if index >= 12:
                break
            gallery.append((row[image_column], str(row.get(caption_column, ""))))
        return f"Loaded {len(gallery)} streamed Hub samples.", gallery
    except Exception as error:  # noqa: BLE001 - report dataset problems in UI
        return f"Dataset preview failed: {error}", []


def sample_images(output_dir: str) -> list[tuple[str, str]]:
    root = (ROOT / output_dir).resolve()
    if not root.exists():
        return []
    images = sorted(
        root.rglob("*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:12]
    return [(str(path), path.name) for path in images]


def training_updates(
    base_model: str,
    mode: str,
    push_to_hub: bool,
    *fields,
) -> Iterator[tuple[str, float, str, list]]:
    global ACTIVE_PROCESS

    try:
        config = make_config(*fields)
        command = command_for(
            config,
            base_model,
            mode,
            push_to_hub,
            fetch=True,
        )
    except Exception as error:  # noqa: BLE001
        yield f"Could not start: {error}", 0, "", []
        return

    with PROCESS_LOCK:
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            yield "Training is already running.", 0, "", []
            return
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        process = ACTIVE_PROCESS

    logs = ["$ " + shlex.join(command)]
    progress = 0.0
    yield "Running", progress, "\n".join(logs), sample_images(config["output_dir"])

    assert process.stdout is not None
    for line_number, line in enumerate(process.stdout, start=1):
        logs.append(line.rstrip())
        logs = logs[-500:]
        matches = STEP_PATTERN.findall(line)
        if matches:
            current, total = map(int, matches[-1])
            if total > 0:
                progress = min(max(current / total, 0), 1)
        images = (
            sample_images(config["output_dir"])
            if line_number % 20 == 0
            else gr.skip()
        )
        yield "Running", progress, "\n".join(logs), images

    return_code = process.wait()
    with PROCESS_LOCK:
        if ACTIVE_PROCESS is process:
            ACTIVE_PROCESS = None
    status = "Complete" if return_code == 0 else f"Stopped with exit code {return_code}"
    yield (
        status,
        1.0 if return_code == 0 else progress,
        "\n".join(logs),
        sample_images(config["output_dir"]),
    )


def stop_training() -> str:
    with PROCESS_LOCK:
        process = ACTIVE_PROCESS
        if process is None or process.poll() is not None:
            return "No training process is running."
        process.terminate()
    return "Stop requested. Waiting for the trainer to exit."


def build_gui() -> gr.Blocks:
    first_config = next(iter(CONFIGS))
    defaults = config_values(first_config)

    with gr.Blocks(title="Clover LoRA Trainer") as demo:
        gr.Markdown(
            "# Clover LoRA Trainer\n"
            "Configure a style, inspect training data, and follow a local run. "
            "The GUI launches the same pinned Diffusers trainer as the CLI."
        )

        with gr.Row():
            config_name = gr.Dropdown(
                choices=list(CONFIGS),
                value=first_config,
                label="Preset",
            )
            base_model = gr.Textbox(
                value=train_lora.BASE_MODEL,
                label="Base model",
            )
            mode = gr.Radio(
                ["Smoke test", "Full training"],
                value="Smoke test",
                label="Run mode",
            )
            push_to_hub = gr.Checkbox(label="Push result to Hugging Face")

        with gr.Tabs():
            with gr.Tab("Configuration"):
                with gr.Row():
                    style = gr.Textbox(value=defaults[0], label="Style slug")
                    dataset = gr.Textbox(value=defaults[1], label="Dataset")
                    trigger = gr.Textbox(value=defaults[2], label="Trigger phrase")
                validation_prompt = gr.Textbox(
                    value=defaults[3],
                    label="Validation prompt",
                    lines=2,
                )
                output_dir = gr.Textbox(value=defaults[4], label="Output directory")
                with gr.Row():
                    max_steps = gr.Number(value=defaults[5], label="Training steps", precision=0)
                    rank = gr.Number(value=defaults[6], label="LoRA rank", precision=0)
                    learning_rate = gr.Number(value=defaults[7], label="Learning rate")
                    batch_size = gr.Number(value=defaults[8], label="Batch size", precision=0)
                with gr.Row():
                    accumulation = gr.Number(value=defaults[9], label="Gradient accumulation", precision=0)
                    mixed_precision = gr.Dropdown(
                        ["fp16", "bf16", "no"],
                        value=defaults[10],
                        label="Mixed precision",
                    )
                    checkpoint_steps = gr.Number(value=defaults[11], label="Checkpoint interval", precision=0)
                    seed = gr.Number(value=defaults[12], label="Seed", precision=0)
                hub_model_id = gr.Textbox(value=defaults[13], label="Hub model ID")

            with gr.Tab("Dataset preview"):
                preview_status = gr.Markdown("Select **Load preview** to inspect up to 12 pairs.")
                preview_gallery = gr.Gallery(
                    label="Training pairs",
                    columns=4,
                    height="auto",
                )
                preview_button = gr.Button("Load preview")

            with gr.Tab("Run"):
                command_preview = gr.Textbox(
                    label="Exact command",
                    lines=5,
                    interactive=False,
                )
                preview_command_button = gr.Button("Preview command")
                with gr.Row():
                    run_button = gr.Button("Start training", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop")
                status = gr.Textbox(value="Idle", label="Status", interactive=False)
                progress = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    label="Progress",
                    interactive=False,
                )
                logs = gr.Textbox(
                    label="Live log",
                    lines=18,
                    max_lines=30,
                    interactive=False,
                )
                samples = gr.Gallery(
                    label="Generated validation samples",
                    columns=4,
                    height="auto",
                )

        fields = [
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
        config_name.change(config_values, config_name, fields)
        preview_button.click(
            preview_dataset,
            dataset,
            [preview_status, preview_gallery],
        )
        preview_command_button.click(
            preview_command,
            [base_model, mode, push_to_hub, *fields],
            command_preview,
        )
        run_button.click(
            training_updates,
            [base_model, mode, push_to_hub, *fields],
            [status, progress, logs, samples],
            concurrency_limit=1,
        )
        stop_button.click(stop_training, outputs=status, queue=False)

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
    )


if __name__ == "__main__":
    main()
