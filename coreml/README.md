# Core ML style packaging

Clover installs one shared stateful Core ML pipeline on iPhone. A trained
rank-16 LoRA remains a small independent style file; it does not need another
copy of the roughly 618 MB U-Net.

```text
Shared Clover Core ML model (~1.5 GB, installed once)
                         +
Named style.safetensors (~6.9 MB)
                         ↓
            144 Core ML MLState tensors
```

## Use the desktop app

Open `trainer_gui.py`, select **Core ML**, and choose only:

1. the trained `pytorch_lora_weights.safetensors` file;
2. the output folder.

Select **Create Core ML style**. The output matches Clover's published Monet,
Pointillism, and Watercolor Anime Core ML repositories:

```text
storybook-anime-coreml/
├── Storybook-Anime.safetensors
├── coreml-state-schema.json
├── README.md
├── LICENSE
└── .gitattributes
```

The packager validates all 72 LoRA target pairs and the rank-16 shape expected
by the iOS model. Diffusers `lora_A` and `lora_B` tensors are renamed to the
`lora.down` and `lora.up` names consumed by Clover. The tensor data is not fused
into the base model and is not expanded into a large per-style U-Net.

## Command line

The GUI calls the same small script directly:

```bash
python coreml/package_style.py \
  outputs/my-style-lora/pytorch_lora_weights.safetensors \
  coreml-models/my-style-coreml
```

This packaging step works on macOS and Windows and does not require Xcode.

## Shared-base development tools

The other scripts in this folder build and validate Clover's shared stateful
U-Net. They are base-model development tools, not part of the normal per-style
workflow exposed by the desktop app.

## Licensing

Packaging code is Apache-2.0. The generated style weights and included model
license use CreativeML Open RAIL-M.
