#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 MODEL_DIR OUTPUT_DIR STYLE_SAFETENSORS" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
model_dir="$(cd "$1" && pwd)"
model_parent="$(dirname "${model_dir}")"
model_name="$(basename "${model_dir}")"
output_dir="$2"
style_file="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
venv_dir="${repo_root}/.venv-coreml"
converter_dir="${repo_root}/.coreml-converter"
python_bin="${PYTHON_BIN:-python3.11}"
converter_revision="e12202c1f6405b83918b58a5d097cd61e3e1f702"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Set PYTHON_BIN to its executable." >&2
  exit 1
fi
if ! command -v xcrun >/dev/null 2>&1; then
  echo "Xcode command-line tools are required." >&2
  exit 1
fi
if [[ ! -f "${style_file}" ]]; then
  echo "Style file does not exist: ${style_file}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"

if [[ ! -d "${converter_dir}/.git" ]]; then
  git clone https://github.com/apple/ml-stable-diffusion.git "${converter_dir}"
fi

git -C "${converter_dir}" checkout --detach "${converter_revision}"
if git -C "${converter_dir}" apply --check \
  "${script_dir}/apple-no-mid-block.patch" >/dev/null 2>&1; then
  git -C "${converter_dir}" apply "${script_dir}/apple-no-mid-block.patch"
elif ! git -C "${converter_dir}" apply --reverse --check \
  "${script_dir}/apple-no-mid-block.patch" >/dev/null 2>&1; then
  echo "Apple converter does not match Clover's pinned patch." >&2
  exit 1
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install \
  --constraint "${script_dir}/constraints.txt" \
  --editable "${converter_dir}" \
  safetensors

cd "${model_parent}"
"${venv_dir}/bin/python" "${script_dir}/convert_stateful_lora_unet.py" \
  --lora-template "${style_file}" \
  --adapter-schema "${output_dir}/coreml-state-schema.json" \
  --model-version "${model_name}" \
  --min-deployment-target iOS18 \
  --convert-unet \
  --attention-implementation ORIGINAL \
  --check-output-correctness \
  -o "${output_dir}"

echo "Stateful Core ML export complete: ${output_dir}"
