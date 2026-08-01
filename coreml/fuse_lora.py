#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "accelerate==1.14.0",
#   "diffusers==0.39.0",
#   "huggingface-hub==0.36.2",
#   "peft==0.17.1",
#   "pillow==12.3.0",
#   "safetensors==0.8.0",
#   "torch==2.7.0",
#   "transformers==4.57.6",
# ]
# ///
"""Fuse a Clover LoRA into a conversion-ready Diffusers pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, PNDMScheduler


PIPELINE_ITEMS = [
    "feature_extractor",
    "model_index.json",
    "safety_checker",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "unet",
    "vae",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardlink_pipeline(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in PIPELINE_ITEMS:
        item = source / name
        target = destination / name
        if item.is_dir():
            shutil.copytree(item, target, copy_function=os.link)
        elif item.is_file():
            os.link(item, target)
        else:
            raise FileNotFoundError(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--lora-revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--prompt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    base = args.base.resolve()
    output = args.output.resolve()
    base_unet = base / "unet" / "diffusion_pytorch_model.safetensors"
    base_digest = sha256(base_unet)

    pipe = DiffusionPipeline.from_pretrained(
        base,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.load_lora_weights(args.lora, revision=args.lora_revision)
    pipe.fuse_lora(lora_scale=args.scale)
    pipe.unload_lora_weights()

    hardlink_pipeline(base, output)
    shutil.rmtree(output / "unet")
    pipe.unet.save_pretrained(
        output / "unet",
        safe_serialization=True,
    )

    validation = None
    if args.prompt:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        image = pipe(
            prompt=args.prompt,
            negative_prompt="blurry, distorted, low detail",
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
        validation_path = output / "fusion-validation.png"
        image.save(validation_path)
        validation = {
            "prompt": args.prompt,
            "steps": args.steps,
            "seed": args.seed,
            "image": validation_path.name,
            "image_sha256": sha256(validation_path),
        }

    fused_unet = output / "unet" / "diffusion_pytorch_model.safetensors"
    fused_digest = sha256(fused_unet)
    if fused_digest == base_digest:
        raise RuntimeError("Fused U-Net is byte-identical to the base U-Net")

    report = {
        "base": str(base),
        "lora": args.lora,
        "lora_revision": args.lora_revision,
        "scale": args.scale,
        "base_unet_sha256": base_digest,
        "fused_unet_sha256": fused_digest,
        "validation": validation,
    }
    (output / "fusion-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
