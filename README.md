# Clover LoRA Training

Train style **LoRAs** for
[`neonforestmist/Clover-Image-Tiny`](https://huggingface.co/neonforestmist/Clover-Image-Tiny)
locally. The adapters are standard Diffusers/PEFT LoRAs — use them anywhere
Diffusers runs (CUDA, Apple MPS, CPU, a Gradio Space, your own app). Exporting
to **Core ML** is an *optional* extra path for shipping a style on Apple
platforms, not a requirement.

This repo is the reproducible source for the three published Clover style
adapters:

| Style | Dataset | Trigger phrase | Steps | Adapter |
|---|---|---|---:|---|
| Monet | [`GPT_Monet_Style_Images`](https://huggingface.co/datasets/neonforestmist/GPT_Monet_Style_Images) | `Monet Style` | 1,000 | [`clover-image-tiny-monet-lora`](https://huggingface.co/neonforestmist/clover-image-tiny-monet-lora) |
| Pointillism | [`GPT_Pointillism_Style_Images`](https://huggingface.co/datasets/neonforestmist/GPT_Pointillism_Style_Images) | `pointillism painting` | 700 | [`clover-image-tiny-pointillism-lora`](https://huggingface.co/neonforestmist/clover-image-tiny-pointillism-lora) |
| Watercolor Anime | [`GPT_Watercolor_Anime_Style_Images`](https://huggingface.co/datasets/neonforestmist/GPT_Watercolor_Anime_Style_Images) | `watercolor anime` | 1,200 | [`clover-image-tiny-watercolor-anime-lora`](https://huggingface.co/neonforestmist/clover-image-tiny-watercolor-anime-lora) |

All three are rank-16 U-Net LoRAs trained with the official Diffusers 0.39.0
text-to-image LoRA trainer. The text encoder and VAE stay frozen.

```text
clover-lora-training/
├── train_lora.py        # local training (wraps the pinned Diffusers trainer)
├── prepare_dataset.py   # validate an image/caption dataset before training
├── configs/             # one JSON per published style
├── data/
│   ├── README.md        # exactly what an image/caption pair should look like
│   └── example-monet/   # three real pairs you can inspect and train on
└── coreml/              # fuse an adapter and convert it to Core ML (see coreml/README.md)
```

---

## 1. What a training pair looks like

A dataset is a Diffusers **imagefolder**: an `images/` folder plus a
`metadata.jsonl` file. Each line pairs one image with one caption.

```json
{"file_name": "images/gpt_monet_0001.png", "text": "Monet Style, a bakery interior with loaves of bread, pastries, and warm light, charming impressionist street shop atmosphere"}
```

- **Every caption starts with the same trigger phrase** (`Monet Style`). That is
  the phrase you type at inference to switch the style on.
- After the trigger, describe the **subject** and a few **style/light/mood**
  cues in natural language.
- Images are square, clean, and consistent in style — 512×512 or larger (they
  are resized to 512 during training).
- Aim for **50–150 pairs** per style; style consistency matters more than count.

Three real pairs are included under [`data/example-monet/`](data/example-monet)
so you can see the exact format. Full details and more caption examples are in
[`data/README.md`](data/README.md).

Validate any dataset before you spend GPU time on it:

```bash
python prepare_dataset.py data/example-monet --trigger "Monet Style"
```

It checks that every image exists, opens, and is large enough, that every
caption is non-empty, and (optionally) that every caption starts with the
trigger — then exits non-zero if anything is wrong.

---

## 2. Set up

```bash
python -m venv .venv && source .venv/bin/activate

# Install the PyTorch build for YOUR platform first (CUDA / ROCm / Apple MPS /
# CPU) from https://pytorch.org/get-started/locally/ — then:
pip install -r requirements.txt

# Log in only if you will train against a private dataset or push to the Hub.
hf auth login
```

A CUDA GPU with ≥8 GB is recommended. Apple Silicon (MPS) and CPU work but are
slow; use `--smoke` there to verify the pipeline end to end.

---

## 3. Train locally

`train_lora.py` reads a style config, downloads the pinned official trainer
once into `.trainers/`, and launches it with `accelerate`.

```bash
# See the exact command without running anything.
python train_lora.py configs/monet.json --dry-run

# Smoke test: 5 steps on 4 images, no Hub push. Proves your setup works.
python train_lora.py configs/monet.json --smoke

# Full local run. Adapter weights land in outputs/monet-lora/.
python train_lora.py configs/monet.json
```

Swap the config for `configs/pointillism.json` or
`configs/watercolor-anime.json` to train the other styles. Each config pins the
dataset, trigger, step count, rank, learning rate, and seed used for the
published adapter.

### Train on your own local dataset

Point a config's `"dataset"` at a local imagefolder path instead of a Hub id
and `train_lora.py` switches to `--train_data_dir` automatically:

```jsonc
{
  "style": "my-style",
  "dataset": "data/example-monet",
  "trigger": "Monet Style",
  "validation_prompt": "Monet Style, a quiet lily pond",
  "max_train_steps": 800,
  "rank": 16
}
```

```bash
python train_lora.py configs/my-style.json
```

### The trained adapter

Output is a single `pytorch_lora_weights.safetensors` (plus checkpoints and
TensorBoard logs). Try it in a few lines:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "neonforestmist/Clover-Image-Tiny", torch_dtype=torch.float16
).to("cuda")  # or "mps"
pipe.load_lora_weights("outputs/monet-lora")

image = pipe(
    "Monet Style, a small blue cat resting beside a lily pond",
    num_inference_steps=20, guidance_scale=7.5,
).images[0]
image.save("sample.png")
```

### Push to the Hub (optional)

```bash
python train_lora.py configs/monet.json --push-to-hub
```

This requires `hub_model_id` in the config and pushes the finished adapter and
validation images to that repo.

---

## 4. Convert to Core ML (optional)

You do **not** need this step to use a trained adapter — it runs directly in
Diffusers (see the snippet above) on CUDA, Apple MPS, or CPU, and in the
[Gradio demo Space](https://huggingface.co/spaces/neonforestmist/Clover-Image-Tiny-Demo).
Core ML is only for shipping a style **on Apple platforms** (iOS/macOS apps).

Apple's Stable Diffusion Core ML runtime cannot load PEFT/Diffusers LoRA
adapters at runtime, so a Core ML style is a copy of the pipeline with **one
adapter fused in**, converted to its own Core ML bundle.

```bash
# 1. Fuse the adapter into a conversion-ready pipeline (proves the U-Net changed).
python coreml/fuse_lora.py \
  --base /path/to/Clover-Image-Tiny \
  --lora outputs/monet-lora \
  --output /tmp/clover-monet-fused \
  --prompt "Monet Style, a blue cat beside a lily pond"

# 2. Convert the fused U-Net to Core ML (macOS + Xcode CLT + Python 3.11).
CONVERT_UNET_ONLY=1 ./coreml/convert.sh /tmp/coreml-out/monet /tmp/clover-monet-fused
```

The full walkthrough, requirements, and the advanced unfused-multifunction path
are in [`coreml/README.md`](coreml/README.md).

---

## Reproducibility

Every config pins the dataset, hyperparameters, and a fixed seed (`20260730`),
and `train_lora.py` pins the trainer to Diffusers `v0.39.0`. Given the same
dataset revision and comparable hardware, a local run reproduces the published
adapter. The base checkpoint revision each adapter was trained against is
recorded on that adapter's Hub model card.

## Licensing

- **Code** in this repo: Apache-2.0 (see [`LICENSE`](LICENSE)).
- **Trained adapters** are derivatives of Clover Image Tiny and inherit its
  **CreativeML Open RAIL-M** model license.
- The **GPT-generated style datasets** are Apache-2.0.

Generated content can inherit limitations and biases from the base checkpoint
and training data. Review outputs before use.
