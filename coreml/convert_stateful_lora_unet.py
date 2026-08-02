#!/usr/bin/env python3
"""Export Clover's base U-Net with runtime-loadable LoRA state buffers.

The exported Core ML model stores the base weights once. Every LoRA down/up
matrix is represented as an iOS 18 ``MLState`` buffer initialized to zero, so
the unmodified model behaves exactly like base Clover. The iOS app fills those
buffers from a standard Diffusers ``safetensors`` style file.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from safetensors.torch import load_file

from python_coreml_stable_diffusion import torch2coreml
from python_coreml_stable_diffusion import unet as coreml_unet


STATE_SUFFIXES = ("lora_down", "lora_up")


class StatefulLoRAConv2d(torch.nn.Module):
    """A base 1x1 convolution with mutable, zero-initialized LoRA buffers."""

    def __init__(
        self,
        base: torch.nn.Conv2d,
        down_shape: torch.Size,
        up_shape: torch.Size,
        scale: float,
    ) -> None:
        super().__init__()
        if base.kernel_size != (1, 1):
            raise ValueError("Clover attention LoRAs require 1x1 convolutions")

        rank, input_channels = down_shape
        output_channels, up_rank = up_shape
        if rank != up_rank:
            raise ValueError("LoRA down/up ranks do not match")
        if input_channels != base.in_channels:
            raise ValueError("LoRA input width does not match base convolution")
        if output_channels != base.out_channels:
            raise ValueError("LoRA output width does not match base convolution")

        self.base = base
        self.register_buffer(
            "lora_down",
            torch.zeros(rank, input_channels, 1, 1),
        )
        self.register_buffer(
            "lora_up",
            torch.zeros(output_channels, rank, 1, 1),
        )
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels_last = hidden_states.permute(0, 2, 3, 1)
        down = self.lora_down[:, :, 0, 0]
        up = self.lora_up[:, :, 0, 0]
        residual = torch.matmul(channels_last, down.transpose(0, 1))
        residual = torch.matmul(residual, up.transpose(0, 1))
        residual = residual.permute(0, 3, 1, 2)
        return self.base(hidden_states) + residual * self.scale


def source_key(target_name: str, state_suffix: str) -> str:
    direction = state_suffix.removeprefix("lora_")
    return f"unet.{target_name}.lora.{direction}.weight"


def inject_stateful_lora(
    model: torch.nn.Module,
    template_path: Path,
    scale: float,
) -> None:
    template = load_file(str(template_path))
    target_names = sorted(
        {
            key.removeprefix("unet.").split(".lora.")[0]
            for key in template
        }
    )

    for target_name in target_names:
        parent_name, child_name = target_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base = model.get_submodule(target_name)
        if not isinstance(base, torch.nn.Conv2d):
            raise TypeError(
                f"Expected Conv2d for {target_name}, found {type(base)}"
            )

        down = template[source_key(target_name, "lora_down")]
        up = template[source_key(target_name, "lora_up")]
        setattr(
            parent,
            child_name,
            StatefulLoRAConv2d(base, down.shape, up.shape, scale),
        )

    if len(target_names) != 72:
        raise RuntimeError(
            f"Expected 72 Clover LoRA targets, found {len(target_names)}"
        )


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--lora-template", type=Path, required=True)
    wrapper.add_argument("--adapter-schema", type=Path, required=True)
    wrapper.add_argument("--lora-scale", type=float, default=1.0)
    wrapper_args, converter_args = wrapper.parse_known_args()

    template_path = wrapper_args.lora_template.resolve()
    if not template_path.is_file():
        raise FileNotFoundError(template_path)

    original_class = coreml_unet.UNet2DConditionModel

    class StatefulLoRAUNet2DConditionModel(original_class):
        def load_state_dict(self, state_dict, *args, **kwargs):
            result = super().load_state_dict(state_dict, *args, **kwargs)
            inject_stateful_lora(
                self,
                template_path=template_path,
                scale=wrapper_args.lora_scale,
            )
            return result

    coreml_unet.UNet2DConditionModel = StatefulLoRAUNet2DConditionModel

    state_records: list[dict] = []
    original_convert = torch2coreml._convert_to_coreml

    def convert_with_lora_states(
        submodule_name,
        torchscript_module,
        sample_inputs,
        output_names,
        args,
        out_path=None,
        precision=None,
        compute_unit=None,
    ):
        if submodule_name != "unet":
            return original_convert(
                submodule_name,
                torchscript_module,
                sample_inputs,
                output_names,
                args,
                out_path,
                precision,
                compute_unit,
            )

        out_path = out_path or torch2coreml._get_out_path(
            args,
            submodule_name,
        )
        compute_unit = compute_unit or ct.ComputeUnit[args.compute_unit]
        if os.path.exists(out_path):
            raise FileExistsError(
                f"Refusing to reuse an existing non-stateful export: {out_path}"
            )

        buffers = [
            (name, value)
            for name, value in torchscript_module.named_buffers()
            if name.endswith(STATE_SUFFIXES)
        ]
        if len(buffers) != 144:
            raise RuntimeError(
                f"Expected 144 LoRA state buffers, found {len(buffers)}"
            )

        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=tuple(value.shape),
                    dtype=np.float16,
                ),
                name=name,
            )
            for name, value in buffers
        ]

        start = time.time()
        model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            inputs=torch2coreml._get_coreml_inputs(sample_inputs, args),
            outputs=[
                ct.TensorType(name=name, dtype=np.float32)
                for name in output_names
            ],
            states=states,
            compute_units=compute_unit,
            compute_precision=precision,
            skip_model_load=True,
        )
        print(f"Converted stateful LoRA U-Net in {time.time() - start:.1f}s")

        exported_states = list(model.get_spec().description.state)
        if len(exported_states) != len(buffers):
            raise RuntimeError(
                "Core ML state count changed during conversion: "
                f"{len(buffers)} -> {len(exported_states)}"
            )

        for (torch_name, value), exported in zip(buffers, exported_states):
            target_name, state_suffix = torch_name.rsplit(".", 1)
            state_records.append(
                {
                    "source_key": source_key(target_name, state_suffix),
                    "state_name": exported.name,
                    "shape": list(value.shape),
                    "element_count": value.numel(),
                }
            )

        del torchscript_module
        gc.collect()
        return model, out_path

    torch2coreml._convert_to_coreml = convert_with_lora_states
    converter_parser = torch2coreml.parser_spec()
    converter_namespace = converter_parser.parse_args(converter_args)
    if converter_namespace.min_deployment_target != "iOS18":
        raise ValueError(
            "Stateful LoRA export requires --min-deployment-target iOS18"
        )
    if converter_namespace.chunk_unet:
        raise ValueError(
            "Export the stateful full U-Net first; state-aware chunking is not supported"
        )
    torch2coreml.main(converter_namespace)

    schema = {
        "schema_version": 1,
        "format": "safetensors",
        "tensor_dtype": "F32",
        "state_dtype": "F16",
        "lora_scale": wrapper_args.lora_scale,
        "state_count": len(state_records),
        "states": sorted(state_records, key=lambda item: item["state_name"]),
    }
    schema_path = wrapper_args.adapter_schema.resolve()
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote state schema: {schema_path}")


if __name__ == "__main__":
    main()
