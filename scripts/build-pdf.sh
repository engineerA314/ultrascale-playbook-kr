#!/usr/bin/env bash
set -euo pipefail

# Move to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Ensure TeX binaries are available (macOS)
export PATH="${PATH}:/Library/TeX/texbin"

# Input / Output
INPUT_MD="The UltraScale Playbook.md"
OUTPUT_PDF="ultrascale-playbook-ko.pdf"

if [[ ! -f "${INPUT_MD}" ]]; then
  echo "Error: ${INPUT_MD} not found in ${REPO_ROOT}"
  exit 1
fi

echo "Building PDF → ${OUTPUT_PDF}"
pandoc metadata.yaml "${INPUT_MD}" \
  -o "${OUTPUT_PDF}" \
  --pdf-engine=xelatex \
  --top-level-division=chapter \
  --highlight-style=tango \
  -V colorlinks=true \
  -V linkcolor=blue \
  --resource-path=".:images" \
  --wrap=auto

echo "Done: ${OUTPUT_PDF}"

