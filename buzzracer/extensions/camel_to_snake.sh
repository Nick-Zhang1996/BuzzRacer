#!/usr/bin/env bash
# Recursively rename UpperCamelCase filenames to snake_case equivalents.
# Example: LapCounter.py -> lap_counter.py

set -euo pipefail

# Root directory (default: current)
ROOT_DIR="${1:-.}"

# Find all files under the directory (excluding .git and already snake_case files)
find "$ROOT_DIR" -type f \
  ! -path "*/.git/*" |
while read -r file; do
  dir=$(dirname "$file")
  base=$(basename "$file")

  # Split name and extension
  name="${base%.*}"
  ext="${base##*.}"
  [[ "$name" == "$ext" ]] && ext=""  # no extension case

  # Convert UpperCamelCase -> snake_case
  snake=$(echo "$name" | sed -E 's/([a-z0-9])([A-Z])/\1_\2/g' | tr 'A-Z' 'a-z')

  # Reattach extension if present
  [[ -n "$ext" ]] && snake="${snake}.${ext}"

  # Skip if no change
  if [[ "$base" != "$snake" ]]; then
    newpath="${dir}/${snake}"

    # Avoid overwriting existing files
    if [[ -e "$newpath" ]]; then
      echo "⚠️  Skipping '$file' → '$newpath' (already exists)"
    else
      echo "Renaming: $file → $newpath"
      git mv "$file" "$newpath" 2>/dev/null || mv "$file" "$newpath"
    fi
  fi
done
