# Core ML export

Clover's native Python desktop app provides a complete GUI for creating and
validating the shared stateful Core ML U-Net used by the iPhone app.

```text
Shared Core ML pipeline (~1.5 GB)
             +
Named style.safetensors (~6.9 MB at rank 16)
             ↓
iOS writes 144 LoRA tensors into MLState
```

The large base is installed once. Monet, Pointillism, Watercolor Anime, and
user-imported compatible styles stay as small independent files.

## Use the GUI

From the repository root:

```bash
source .venv/bin/activate
python trainer_gui.py
```

Select the **Core ML** tab in the desktop window. It uses standard native
controls and file pickers; no browser or local web server is involved.

<p align="center">
  <img src="../assets/gui-coreml.png" alt="Core ML export workspace" width="1200">
</p>

### 1. Choose local inputs

- **Clover model folder** — a local Diffusers checkpoint containing
  `model_index.json`.
- **Style .safetensors** — a compatible Clover LoRA. It defines tensor names
  and shapes; exported state values still start at zero.
- **Core ML output folder** — where the package and schema should be written.

### 2. Check readiness

The preflight checks macOS, Python 3.11, Xcode command-line tools, the chosen
model and style, previously generated artifacts, and available disk space.
Resolve every red requirement before running a step. Treat the disk warning as
actionable because conversion environments and intermediate packages are
large.

### 3. Run all three steps

1. **Export stateful U-Net** creates `Unet.mlpackage` and
   `coreml-state-schema.json`.
2. **Compile for Xcode** invokes `xcrun coremlcompiler` and writes compiled
   resources under `compiled/`.
3. **Validate parity** runs deterministic base and styled inputs through
   PyTorch and Core ML. Both PSNR results must meet the 35 dB default.

Every action has a command preview, live progress, streamed logs, cancellation,
and an artifact summary. The GUI calls the same scripts documented below.

## Equivalent command line

### Export

```bash
./coreml/export_stateful.sh \
  /path/to/Clover-Image-Tiny \
  coreml-models/clover-stateful \
  /path/to/Monet.safetensors
```

The script checks out Apple’s converter at the repository’s pinned revision,
applies Clover’s no-middle-block patch, creates an isolated Python environment,
and exports:

- `Unet.mlpackage` — the iOS 18 stateful Core ML U-Net;
- `coreml-state-schema.json` — the Diffusers-key to Core ML-state mapping.

### Compile

```bash
xcrun coremlcompiler compile \
  coreml-models/clover-stateful/Unet.mlpackage \
  coreml-models/clover-stateful/compiled
```

### Validate

```bash
.venv-coreml/bin/python coreml/validate_stateful_lora.py \
  --model-version /path/to/Clover-Image-Tiny \
  --coreml-model coreml-models/clover-stateful/Unet.mlpackage \
  --adapter-schema coreml-models/clover-stateful/coreml-state-schema.json \
  --lora-weights /path/to/Monet.safetensors \
  --minimum-psnr 35
```

Validate at least one style from every training configuration before
publishing a new shared base.

## Tooling map

```text
coreml/
├── export_stateful.sh              # current iOS 18 entry point
├── convert_stateful_lora_unet.py   # inject and export MLState buffers
├── validate_stateful_lora.py       # base/style PyTorch ↔ Core ML parity
├── constraints.txt                 # pinned converter environment
├── apple-no-mid-block.patch        # Clover converter compatibility
├── convert.sh                      # legacy fused/chunked conversion
├── fuse_lora.py                    # legacy style fusion
└── convert_lora_unet.py            # legacy and validation helper
```

## Legacy fused conversion

`convert.sh` and `fuse_lora.py` bake a style into its own Core ML U-Net. That
path produced the old roughly 648 MB-per-style downloads and remains only for
reproducibility. The current iOS catalog uses one shared stateful base and
small named `.safetensors` styles.

## Licensing

The conversion code is Apache-2.0. Converted model weights and style files are
derivatives of Clover Image Tiny and remain under CreativeML Open RAIL-M.
