# Core ML export for Clover LoRA styles

The current iPhone architecture stores Clover once and applies small named
style files at runtime:

```text
Shared Core ML pipeline (~1.5 GB)
             +
Monet.safetensors (~6.9 MB)
             ↓
iOS writes 144 LoRA tensors into MLState
```

This requires iOS 18 or newer. The exported U-Net behaves like base Clover
when its state buffers are zero and changes style when the app loads Monet,
Pointillism, or Watercolor Anime weights.

## Files

```text
coreml/
├── export_stateful.sh              # current iOS 18 export entry point
├── convert_stateful_lora_unet.py   # add zero-initialized LoRA MLState buffers
├── validate_stateful_lora.py       # compare Core ML base/style outputs
├── convert.sh                      # legacy fused/chunked converter
├── fuse_lora.py                    # legacy fused style preparation
├── convert_lora_unet.py            # validation helper and legacy unfused path
├── apple-no-mid-block.patch        # Clover converter compatibility patch
└── constraints.txt                 # pinned Apple converter environment
```

## Requirements

- macOS and Xcode command-line tools (`xcrun` on `PATH`)
- Python 3.11
- a local checkout of `Clover-Image-Tiny`
- one compatible Clover LoRA `safetensors` file to define the state shapes

The template file determines names and shapes only. All exported state values
start at zero, so the base model is not permanently styled.

## Export the stateful U-Net

Run from this repository:

```bash
./coreml/export_stateful.sh \
  /path/to/Clover-Image-Tiny \
  /tmp/clover-stateful \
  /path/to/Monet.safetensors
```

The script checks out Apple's converter at the repository's pinned revision,
applies Clover's no-middle-block compatibility patch, creates an isolated
conversion environment, and exports:

- `Unet.mlpackage`, the stateful Core ML U-Net;
- `coreml-state-schema.json`, the mapping from Diffusers tensor keys to Core
  ML state names.

Compile the package for distribution:

```bash
xcrun coremlcompiler compile \
  /tmp/clover-stateful/Unet.mlpackage \
  /tmp/clover-stateful/compiled
```

The text encoder, VAE decoder, safety checker, tokenizer, and merges are shared
with the normal base conversion and only need to be distributed once.

## Validate base and style parity

The validation tool runs identical deterministic U-Net inputs through PyTorch
and Core ML twice: with zero state and with a named style loaded.

```bash
python coreml/validate_stateful_lora.py \
  --model-version /path/to/Clover-Image-Tiny \
  --coreml-model /tmp/clover-stateful/Unet.mlpackage \
  --adapter-schema /tmp/clover-stateful/coreml-state-schema.json \
  --lora-weights /path/to/Monet.safetensors
```

Both PSNR values must meet the default 35 dB threshold. Validate at least one
style from every training configuration before publishing a new base model.

## Legacy fused conversion

`convert.sh` and `fuse_lora.py` bake a style into a separate Core ML U-Net.
That path explains the old roughly 648 MB-per-style downloads and remains for
reproducibility. It is not the architecture used by the current iOS catalog.

## Licensing

The conversion code is Apache-2.0. Converted model weights and style files are
derivatives of Clover Image Tiny and remain under CreativeML Open RAIL-M.
