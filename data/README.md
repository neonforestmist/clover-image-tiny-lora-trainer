# Dataset format

A Clover style dataset is a Diffusers **imagefolder**: a directory with an
`images/` folder and a `metadata.jsonl` file that pairs each image with a
caption.

```text
data/example-monet/
├── images/
│   ├── gpt_monet_0001.png
│   ├── gpt_monet_0002.png
│   └── gpt_monet_0003.png
└── metadata.jsonl
```

Each line of `metadata.jsonl` is one JSON object with two fields:

```json
{"file_name": "images/gpt_monet_0001.png", "text": "Monet Style, a bakery interior with loaves of bread, pastries, and warm light, charming impressionist street shop atmosphere"}
```

- **`file_name`** — path to the image, relative to the dataset directory.
- **`text`** — the caption. Start every caption with the same **trigger phrase**
  (here `Monet Style`) so the adapter learns to attach the style to that
  phrase. At inference you then write `Monet Style, ...` to invoke it.

## What a good pair looks like

| | |
|---|---|
| **Image** | 512×512 or larger, square, clean, consistent in style. The three examples here are 1024×1024 and get resized to 512 during training. |
| **Caption** | `trigger, subject, then a few style/lighting/mood cues`. Describe what is in the image, not how to paint it. Keep it natural. |

Good captions for a Monet adapter:

```text
Monet Style, a quiet lily pond with floating blossoms, soft morning light, loose brushwork
Monet Style, a village street after rain, reflections on wet stones, hazy atmosphere
Monet Style, a garden table with tea cups and fresh flowers, sunlight through leaves
```

Aim for **50–150 pairs** for a single style. Consistency of style across the
set matters more than raw count.

## The example

`example-monet/` holds three real pairs pulled from
[`neonforestmist/GPT_Monet_Style_Images`](https://huggingface.co/datasets/neonforestmist/GPT_Monet_Style_Images),
which has the full 100-pair set. Validate any dataset before training:

```bash
python prepare_dataset.py data/example-monet --trigger "Monet Style"
```

You can train directly against a local folder or against a Hub dataset id — see
the top-level [README](../README.md).
