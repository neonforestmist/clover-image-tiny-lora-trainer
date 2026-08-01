#!/usr/bin/env python3
"""Convert an unfused Clover LoRA U-Net for Core ML multifunction merging.

This wrapper keeps the base convolution weights byte-identical between style
exports. Each LoRA remains as its pair of rank-reduction/rank-expansion 1x1
convolutions so Core ML can deduplicate the shared base weights later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from python_coreml_stable_diffusion import torch2coreml
from python_coreml_stable_diffusion import unet as coreml_unet


class LoRAConv2d(torch.nn.Module):
    """A frozen base 1x1 convolution plus an unfused LoRA residual."""

    def __init__(
        self,
        base: torch.nn.Conv2d,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        scale: float,
    ) -> None:
        super().__init__()
        if base.kernel_size != (1, 1):
            raise ValueError("Clover attention LoRAs require 1x1 convolutions")

        rank, input_channels = down_weight.shape
        output_channels, up_rank = up_weight.shape
        if rank != up_rank:
            raise ValueError("LoRA down/up ranks do not match")
        if input_channels != base.in_channels:
            raise ValueError("LoRA input width does not match base convolution")
        if output_channels != base.out_channels:
            raise ValueError("LoRA output width does not match base convolution")

        self.base = base
        self.down = torch.nn.Conv2d(
            input_channels,
            rank,
            kernel_size=1,
            bias=False,
        )
        self.up = torch.nn.Conv2d(
            rank,
            output_channels,
            kernel_size=1,
            bias=False,
        )
        self.scale = scale

        with torch.no_grad():
            self.down.weight.copy_(
                down_weight.reshape(rank, input_channels, 1, 1)
            )
            self.up.weight.copy_(
                up_weight.reshape(output_channels, rank, 1, 1)
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.down(hidden_states))
        return self.base(hidden_states) + residual * self.scale


def inject_lora(
    model: torch.nn.Module,
    weights_path: Path,
    scale: float,
) -> None:
    state = load_file(str(weights_path))
    target_names = sorted(
        {
            key.removeprefix("unet.").split(".lora.")[0]
            for key in state
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

        prefix = f"unet.{target_name}.lora"
        down = state[f"{prefix}.down.weight"]
        up = state[f"{prefix}.up.weight"]
        setattr(parent, child_name, LoRAConv2d(base, down, up, scale))

    if len(target_names) != 72:
        raise RuntimeError(
            f"Expected 72 Clover LoRA targets, found {len(target_names)}"
        )


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--lora-weights", type=Path, required=True)
    wrapper.add_argument("--lora-scale", type=float, default=1.0)
    wrapper_args, converter_args = wrapper.parse_known_args()

    weights_path = wrapper_args.lora_weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    original_class = coreml_unet.UNet2DConditionModel

    class LoRAUNet2DConditionModel(original_class):
        def load_state_dict(self, state_dict, *args, **kwargs):
            result = super().load_state_dict(state_dict, *args, **kwargs)
            inject_lora(
                self,
                weights_path=weights_path,
                scale=wrapper_args.lora_scale,
            )
            return result

    coreml_unet.UNet2DConditionModel = LoRAUNet2DConditionModel
    converter_parser = torch2coreml.parser_spec()
    converter_namespace = converter_parser.parse_args(converter_args)
    torch2coreml.main(converter_namespace)


if __name__ == "__main__":
    main()
