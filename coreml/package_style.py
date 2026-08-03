#!/usr/bin/env python3
"""Package one trained Clover LoRA for the iOS stateful Core ML model."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


BASE_MODEL = "neonforestmist/Clover-Image-Tiny"
EXPECTED_STATES = 144
EXPECTED_TARGETS = 72


def normalized_key(key: str) -> str:
    if key.endswith(".lora_A.weight"):
        return key.removesuffix(".lora_A.weight") + ".lora.down.weight"
    if key.endswith(".lora_B.weight"):
        return key.removesuffix(".lora_B.weight") + ".lora.up.weight"
    return key


def state_name(source_key: str) -> str:
    return (
        source_key.removeprefix("unet.")
        .removesuffix(".weight")
        .replace(".", "_")
    )


def style_name(output: Path) -> str:
    parts = [
        part
        for part in re.split(r"[-_\s]+", output.name)
        if part and part.casefold() not in {"coreml", "lora"}
    ]
    if not parts:
        return "Clover-Style"
    return "-".join(
        part.upper() if part.casefold() in {"gpt"} else part.capitalize()
        for part in parts
    )


def package_style(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.expanduser()
    output = output.expanduser().resolve()
    if not source.is_file() or source.suffix.casefold() != ".safetensors":
        raise ValueError("Choose a trained Clover .safetensors file.")
    source = source.resolve()

    raw = load_file(str(source), device="cpu")
    tensors = {}
    for key, tensor in raw.items():
        key = normalized_key(key)
        if key in tensors:
            raise ValueError(f"Duplicate normalized tensor name: {key}")
        tensors[key] = tensor.detach().to(dtype=tensor.dtype, device="cpu").contiguous()

    if len(tensors) != EXPECTED_STATES:
        raise ValueError(
            f"Expected {EXPECTED_STATES} Clover LoRA tensors, found {len(tensors)}."
        )
    invalid = [
        key
        for key in tensors
        if not key.startswith("unet.")
        or not key.endswith((".lora.down.weight", ".lora.up.weight"))
    ]
    if invalid:
        raise ValueError(f"Unsupported Clover LoRA tensor: {invalid[0]}")

    targets = {
        key.rsplit(".lora.", 1)[0]
        for key in tensors
    }
    if len(targets) != EXPECTED_TARGETS:
        raise ValueError(
            f"Expected {EXPECTED_TARGETS} Clover LoRA targets, found {len(targets)}."
        )
    for target in targets:
        down = tensors.get(f"{target}.lora.down.weight")
        up = tensors.get(f"{target}.lora.up.weight")
        if down is None or up is None:
            raise ValueError(f"LoRA down/up pair is incomplete: {target}")
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
            raise ValueError(f"LoRA tensor shapes are incompatible: {target}")
        if down.shape[0] != 16:
            raise ValueError(
                f"The iOS Clover model expects rank 16; {target} uses rank {down.shape[0]}."
            )

    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{style_name(output)}.safetensors"
    save_file(tensors, str(destination), metadata={"format": "pt"})

    states = []
    for key, tensor in tensors.items():
        shape = list(tensor.shape) + [1, 1]
        states.append(
            {
                "source_key": key,
                "state_name": state_name(key),
                "shape": shape,
                "element_count": tensor.numel(),
            }
        )
    schema = {
        "schema_version": 1,
        "format": "safetensors",
        "tensor_dtype": "F32",
        "state_dtype": "F16",
        "lora_scale": 1.0,
        "state_count": len(states),
        "states": sorted(states, key=lambda item: item["state_name"]),
    }
    schema_path = output / "coreml-state-schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")

    license_source = Path(__file__).with_name("CREATIVEML-OPENRAIL-M.txt")
    shutil.copyfile(license_source, output / "LICENSE")
    (output / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
    )
    label = style_name(output).replace("-", " ")
    title = label if label.casefold().endswith(" style") else f"{label} Style"
    (output / "README.md").write_text(
        "---\n"
        "library_name: coreml\n"
        "pipeline_tag: text-to-image\n"
        "license: creativeml-openrail-m\n"
        f"base_model: {BASE_MODEL}\n"
        "tags:\n"
        "  - coreml\n"
        "  - stable-diffusion\n"
        "  - ios\n"
        "  - clover-image\n"
        "  - lora\n"
        "---\n\n"
        f"# Clover Image Tiny — {title}\n\n"
        "A compact rank-16 style for the stateful iOS 18 Clover Core ML pipeline.\n\n"
        f"- Style file: **{destination.name}** ({destination.stat().st_size:,} bytes)\n"
        "- Required base: [neonforestmist/Clover-Image-Tiny-CoreML]"
        "(https://huggingface.co/neonforestmist/Clover-Image-Tiny-CoreML)\n\n"
        f"The Clover Swift pipeline loads `{destination.name}` into the shared stateful "
        "U-Net. It is not a standalone 648 MB replacement U-Net. "
        "`coreml-state-schema.json` maps the LoRA tensors to the 144 Core ML state buffers.\n"
    )
    return destination, schema_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("style_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    destination, schema = package_style(args.style_file, args.output_dir)
    print(f"Core ML style: {destination}")
    print(f"State schema: {schema}")
    print("Complete.")


if __name__ == "__main__":
    main()
