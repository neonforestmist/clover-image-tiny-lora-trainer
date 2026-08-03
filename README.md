# Clover Image Tiny LoRA Trainer

[![Quality checks](https://github.com/neonforestmist/clover-image-tiny-lora-trainer/actions/workflows/quality.yml/badge.svg)](https://github.com/neonforestmist/clover-image-tiny-lora-trainer/actions/workflows/quality.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11–3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Clover model](https://img.shields.io/badge/Model-Clover_Image_Tiny-FFD21E?logo=huggingface&logoColor=111)](https://huggingface.co/neonforestmist/Clover-Image-Tiny)
[![iPhone app](https://img.shields.io/badge/iPhone-Core_ML-111111?logo=apple&logoColor=white)](https://github.com/neonforestmist/Clover-Image-Tiny-iOS)
[![License](https://img.shields.io/badge/Code-Apache_2.0-D22128?logo=apache&logoColor=white)](LICENSE)

A visual, local workspace for training compact style LoRAs for
[`neonforestmist/Clover-Image-Tiny`](https://huggingface.co/neonforestmist/Clover-Image-Tiny)
and exporting its stateful Core ML U-Net for iPhone.

The base model stays shared. A rank-16 style is roughly 6.9 MB and remains a
normal `.safetensors` file; the iOS app loads it into 144 Core ML state tensors
instead of downloading another roughly 648 MB U-Net.

<p align="center">
  <img src="assets/gui-training.png" alt="Clover Studio training workspace" width="1200">
</p>

## Start the studio

Use Python 3.11 or 3.12. Install the PyTorch build appropriate for your CUDA,
ROCm, Apple Silicon, or CPU environment first, then install this repository.

```bash
git clone https://github.com/neonforestmist/clover-image-tiny-lora-trainer.git
cd clover-image-tiny-lora-trainer

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python trainer_gui.py
```

Open `http://127.0.0.1:7860`. The server listens only on localhost unless you
explicitly pass `--share`.

## Training workspace

The interface keeps the work in three visible stages:

1. **Style recipe** — load Monet, Pointillism, or Watercolor Anime defaults;
   choose a Hub dataset or local imagefolder; and set the trigger phrase.
2. **Training plan** — choose a five-step smoke test or a full run, then tune
   steps, rank, learning rate, batch behavior, checkpoints, and seed.
3. **Review and run** — inspect the exact `accelerate` command before starting,
   follow progress and logs, stop safely, and review generated validation
   samples.

Always run the five-step smoke test first. It uses four samples, writes only
local artifacts, and catches most environment or dataset problems before a
long GPU run.

### Published recipes

| Style | Dataset | Trigger | Steps | Runtime style file |
|---|---|---|---:|---|
| Monet | [`GPT_Monet_Style_Images`](https://huggingface.co/datasets/neonforestmist/GPT_Monet_Style_Images) | `Monet Style` | 1,000 | [`Monet.safetensors`](https://huggingface.co/neonforestmist/clover-image-tiny-monet-lora-coreml) |
| Pointillism | [`GPT_Pointillism_Style_Images`](https://huggingface.co/datasets/neonforestmist/GPT_Pointillism_Style_Images) | `pointillism painting` | 700 | [`Pointillism.safetensors`](https://huggingface.co/neonforestmist/clover-image-tiny-pointillism-lora-coreml) |
| Watercolor Anime | [`GPT_Watercolor_Anime_Style_Images`](https://huggingface.co/datasets/neonforestmist/GPT_Watercolor_Anime_Style_Images) | `watercolor anime` | 1,200 | [`Watercolor-Anime.safetensors`](https://huggingface.co/neonforestmist/clover-image-tiny-watercolor-anime-lora-coreml) |

All recipes use the official Diffusers 0.39.0 text-to-image LoRA trainer. The
U-Net attention layers train at rank 16; the text encoder and VAE remain
frozen.

## Core ML workspace

Core ML conversion is part of the same studio—no shell-only handoff is
required.

<p align="center">
  <img src="assets/gui-coreml.png" alt="Clover Studio Core ML export workspace" width="1200">
</p>

The **Core ML export** tab provides:

- local path fields for the Clover Diffusers checkpoint, a compatible
  `.safetensors` style, and the output directory;
- a preflight panel that checks macOS, Python 3.11, Xcode tools, model files,
  style weights, generated artifacts, and free disk space;
- separate **Export stateful U-Net**, **Compile for Xcode**, and **Validate
  parity** actions;
- a copyable command preview, streamed conversion logs, progress, stop control,
  and output-artifact discovery.

The stateful export requires macOS, Xcode command-line tools, Python 3.11, and
iOS 18 or newer at runtime. Keep at least 15 GB free while building the
converter environment and model package. The studio warns below 15 GB and
blocks on missing required inputs.

See [`coreml/README.md`](coreml/README.md) for the artifact contract, the
equivalent CLI commands, and the legacy fused-conversion explanation.

## Dataset format

A local dataset uses the Diffusers imagefolder layout:

```text
my-style/
├── images/
│   ├── 0001.png
│   └── 0002.png
└── metadata.jsonl
```

Each JSONL row pairs an image with a caption:

```json
{"file_name": "images/0001.png", "text": "My Style, a quiet garden at sunrise"}
```

Aim for 50–150 visually consistent images, use a stable trigger at the start
of every caption, and validate the set before training:

```bash
python prepare_dataset.py /path/to/my-style --trigger "My Style"
```

The bundled [`data/example-monet`](data/example-monet) folder is a complete
three-image example. More detail is in [`data/README.md`](data/README.md).

## Command line

The GUI and CLI call the same pinned wrapper and JSON recipes.

```bash
# Inspect the command without downloading or training.
python train_lora.py configs/monet.json --dry-run

# Five steps on four samples; never pushes.
python train_lora.py configs/monet.json --smoke

# Full local run.
python train_lora.py configs/monet.json

# Full run, then publish to the configured Hugging Face repository.
python train_lora.py configs/monet.json --push-to-hub
```

Training writes `pytorch_lora_weights.safetensors` under `outputs/`. Rename
the published copy for its style, such as `Monet.safetensors`; users should see
the style name, not the implementation term “adapter.”

## Use a trained style with Diffusers

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "neonforestmist/Clover-Image-Tiny",
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_lora_weights("outputs/monet-lora")

image = pipe(
    "Monet Style, a small blue cat beside a lily pond",
    num_inference_steps=20,
    guidance_scale=7.5,
).images[0]
image.save("sample.png")
```

## Repository map

```text
clover-image-tiny-lora-trainer/
├── trainer_gui.py          # Clover Studio web-local GUI
├── train_lora.py           # pinned Diffusers training wrapper
├── prepare_dataset.py      # local dataset validator
├── configs/                # reproducible published recipes
├── data/                   # format guide and sample imagefolder
├── assets/                 # style examples and real GUI screenshots
├── tests/                  # command and workflow checks
└── coreml/                 # stateful export, compilation, and validation
```

## Ecosystem

- [Clover Image Tiny model and model card](https://huggingface.co/neonforestmist/Clover-Image-Tiny)
- [Live Hugging Face demo](https://huggingface.co/spaces/neonforestmist/Clover-Image-Tiny-Demo)
- [Native SwiftUI/Core ML iPhone app](https://github.com/neonforestmist/Clover-Image-Tiny-iOS)
- [Clover Image Tiny runnable examples](https://github.com/neonforestmist/Clover-Image-Tiny)

## Licensing

- Code in this repository: Apache-2.0; see [`LICENSE`](LICENSE).
- Clover Image Tiny and trained style weights: CreativeML Open RAIL-M.
- The linked generated style datasets: Apache-2.0.

Generated content can inherit limitations and biases from the base checkpoint
and training data. Review outputs before use or publication.
