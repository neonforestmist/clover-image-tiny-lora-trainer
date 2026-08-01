#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
model_dir="${2:-${repo_root}}"
model_dir="$(cd "${model_dir}" && pwd)"
model_parent="$(dirname "${model_dir}")"
model_name="$(basename "${model_dir}")"
output_dir="${1:-${model_dir}/coreml-models}"
venv_dir="${repo_root}/.venv-coreml"
converter_dir="${repo_root}/.coreml-converter"
python_bin="${PYTHON_BIN:-python3.11}"
converter_revision="e12202c1f6405b83918b58a5d097cd61e3e1f702"
unet_only="${CONVERT_UNET_ONLY:-0}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Set PYTHON_BIN to its executable." >&2
  exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
  echo "Xcode command-line tools are required." >&2
  exit 1
fi

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"

if [[ ! -d "${converter_dir}/.git" ]]; then
  git clone https://github.com/apple/ml-stable-diffusion.git "${converter_dir}"
fi

git -C "${converter_dir}" checkout --detach "${converter_revision}"
if ! git -C "${converter_dir}" apply --check "${script_dir}/apple-no-mid-block.patch" >/dev/null 2>&1; then
  if git -C "${converter_dir}" apply --reverse --check "${script_dir}/apple-no-mid-block.patch" >/dev/null 2>&1; then
    echo "Clover middle-block compatibility patch is already applied."
  else
    echo "Apple converter does not match the pinned patch." >&2
    exit 1
  fi
else
  git -C "${converter_dir}" apply "${script_dir}/apple-no-mid-block.patch"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install \
  --constraint "${script_dir}/constraints.txt" \
  --editable "${converter_dir}"

cd "${model_parent}"

converter=(
  "${venv_dir}/bin/python"
  -m python_coreml_stable_diffusion.torch2coreml
  --model-version "${model_name}"
  --min-deployment-target iOS17
  -o "${output_dir}"
)

if [[ "${unet_only}" != "1" ]]; then
  "${converter[@]}" --convert-vae-decoder --check-output-correctness
  "${converter[@]}" --convert-text-encoder --check-output-correctness

  # The converted checker is usable, but its observed random-input PSNR is
  # 34.0 dB, just below Apple's 35 dB assertion. Validate it in the smoke test.
  "${converter[@]}" --convert-safety-checker
fi

"${converter[@]}" \
  --convert-unet \
  --attention-implementation SPLIT_EINSUM \
  --check-output-correctness

"${converter[@]}" \
  --convert-unet \
  --attention-implementation SPLIT_EINSUM \
  --chunk-unet

"${converter[@]}" --bundle-resources-for-swift-cli

if [[ "${unet_only}" != "1" ]]; then
  # Use the checkpoint's exact tokenizer resources rather than equivalent
  # upstream CLIP serializations downloaded by Apple's bundler.
  cp "${model_dir}/tokenizer/vocab.json" "${output_dir}/Resources/vocab.json"
  cp "${model_dir}/tokenizer/merges.txt" "${output_dir}/Resources/merges.txt"
fi

echo "Core ML export complete: ${output_dir}"
