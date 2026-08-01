# Core ML conversion (optional)

This step is **optional**. A trained adapter already runs directly in Diffusers
on CUDA, Apple MPS, or CPU, and in the Gradio demo Space — no conversion
needed. Core ML is only for shipping a style inside a native **Apple-platform**
app (iOS/macOS).

Apple's Stable Diffusion Core ML runtime does **not** load Diffusers/PEFT LoRA
adapters at runtime. To ship a style that way you bake one adapter into a copy
of the pipeline and convert that fused U-Net to its own Core ML bundle. The
picker in an app then chooses the base bundle or one of the fused style
bundles.

This directory has two conversion paths.

```text
coreml/
├── fuse_lora.py           # fuse one adapter into a Diffusers pipeline
├── convert.sh             # drive Apple's ml-stable-diffusion converter
├── convert_lora_unet.py   # advanced: keep the LoRA unfused for multifunction merging
├── apple-no-mid-block.patch  # Clover has no U-Net mid block; patch the converter
└── constraints.txt        # pinned converter dependencies
```

## Requirements

- macOS with Xcode command-line tools (`xcrun` on `PATH`)
- Python 3.11 (Apple's converter pins this); set `PYTHON_BIN=python3.11`
- The base Diffusers checkpoint checked out locally (the `Clover-Image-Tiny`
  repo). `convert.sh` treats the directory *above* the model as the working
  root, matching Apple's `--model-version` convention.

## Path A — fuse then convert (what the shipped iOS styles use)

1. Fuse a trained adapter into a conversion-ready pipeline. This hard-links the
   unchanged components, writes a new U-Net, and proves the U-Net checksum
   changed:

   ```bash
   python fuse_lora.py \
     --base /path/to/Clover-Image-Tiny \
     --lora neonforestmist/clover-image-tiny-monet-lora \
     --output /tmp/clover-monet-fused \
     --prompt "Monet Style, a blue cat beside a lily pond"
   ```

   The optional `--prompt` renders a local validation image (uses Apple MPS
   when available) so you can eyeball the fused weights before the long
   conversion.

2. Convert just the fused U-Net (the other components are style-independent and
   only need converting once from the base):

   ```bash
   CONVERT_UNET_ONLY=1 ./convert.sh \
     /tmp/coreml-out/monet \
     /tmp/clover-monet-fused
   ```

   Drop `CONVERT_UNET_ONLY=1` on the first run to also convert the VAE decoder,
   text encoder, and safety checker, and to bundle Swift CLI resources.

`convert.sh` clones Apple's `ml-stable-diffusion` at a pinned revision, applies
`apple-no-mid-block.patch` (Clover's U-Net omits the mid block), builds an
isolated venv from `constraints.txt`, and emits SPLIT_EINSUM `.mlpackage`
bundles plus a chunked U-Net for on-device use.

## Path B — unfused multifunction merge (advanced)

`convert_lora_unet.py` keeps each adapter as its rank-down / rank-up 1×1
convolution pair instead of fusing it, so the shared base convolutions stay
byte-identical across styles and Core ML can deduplicate them when several
styles are merged into one multifunction model. Use this only if you are
building a single bundle that carries multiple styles; for one style per
bundle, Path A is simpler.

## Licensing

Fused and converted U-Nets are derivatives of Clover Image Tiny and stay under
the **CreativeML Open RAIL-M** model license. The conversion code here is
Apache-2.0.
