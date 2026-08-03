#!/usr/bin/env python3
"""Validate base and runtime-loaded style outputs of a stateful Clover U-Net."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

from python_coreml_stable_diffusion import unet as coreml_unet

from convert_lora_unet import inject_lora


def psnr(reference: np.ndarray, actual: np.ndarray) -> float:
    error = np.asarray(reference, dtype=np.float64) - np.asarray(
        actual,
        dtype=np.float64,
    )
    rmse = math.sqrt(float(np.mean(error * error)))
    if rmse == 0:
        return math.inf
    dynamic_range = float(reference.max() - reference.min())
    return 20 * math.log10(dynamic_range / rmse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--coreml-model", type=Path, required=True)
    parser.add_argument("--adapter-schema", type=Path, required=True)
    parser.add_argument("--lora-weights", type=Path, required=True)
    parser.add_argument("--minimum-psnr", type=float, default=35.0)
    args = parser.parse_args()

    torch.manual_seed(2026)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_version,
        local_files_only=True,
    )
    reference = coreml_unet.UNet2DConditionModel(
        **pipeline.unet.config
    ).eval()
    reference.load_state_dict(pipeline.unet.state_dict())
    del pipeline

    model = ct.models.MLModel(
        str(args.coreml_model.resolve()),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    input_specs = {
        item.name: item for item in model.get_spec().description.input
    }
    sample_shape = tuple(
        input_specs["sample"].type.multiArrayType.shape
    )
    hidden_shape = tuple(
        input_specs["encoder_hidden_states"].type.multiArrayType.shape
    )
    batch_size = sample_shape[0]
    sample = torch.rand(*sample_shape)
    timestep = torch.full((batch_size,), 981.0, dtype=torch.float32)
    hidden_states = torch.rand(*hidden_shape)
    inputs = {
        "sample": sample.numpy().astype(np.float16),
        "timestep": timestep.numpy().astype(np.float16),
        "encoder_hidden_states": hidden_states.numpy().astype(np.float16),
    }

    with torch.no_grad():
        base_reference = reference(sample, timestep, hidden_states)[0].numpy()

    state = model.make_state()
    base_actual = model.predict(inputs, state=state)["noise_pred"]
    base_psnr = psnr(base_reference, base_actual)
    print(f"Base PSNR: {base_psnr:.2f} dB")

    schema = json.loads(args.adapter_schema.read_text())
    style = load_file(str(args.lora_weights.resolve()))
    for record in schema["states"]:
        value = style[record["source_key"]]
        value = value.reshape(record["shape"]).numpy().astype(np.float32)
        state.write_state(name=record["state_name"], value=value)

    inject_lora(reference, args.lora_weights.resolve(), scale=1.0)
    with torch.no_grad():
        style_reference = reference(sample, timestep, hidden_states)[0].numpy()
    style_actual = model.predict(inputs, state=state)["noise_pred"]
    style_psnr = psnr(style_reference, style_actual)
    print(f"LoRA PSNR: {style_psnr:.2f} dB")

    if min(base_psnr, style_psnr) < args.minimum_psnr:
        raise RuntimeError(
            "Stateful U-Net parity fell below "
            f"{args.minimum_psnr:.1f} dB"
        )

    print(
        json.dumps(
            {
                "base_psnr_db": base_psnr,
                "lora_psnr_db": style_psnr,
                "state_count": schema["state_count"],
                "style_bytes": args.lora_weights.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
