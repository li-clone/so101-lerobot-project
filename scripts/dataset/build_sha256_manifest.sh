#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATASET_DIR OUTPUT_FILE" >&2
  exit 2
fi

dataset_dir="$(realpath "$1")"
output_file="$2"
parent="$(dirname "$dataset_dir")"
name="$(basename "$dataset_dir")"
mkdir -p "$(dirname "$output_file")"
(
  cd "$parent"
  find "$name" -type f -print0 | sort -z | xargs -0 sha256sum
) > "$output_file"
