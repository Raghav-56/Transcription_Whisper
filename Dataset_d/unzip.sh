#!/usr/bin/env bash

# Usage: unzip.sh [directory]

DIR="${1:-.}"

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: 'unzip' is required. Install it and retry." >&2
  exit 1
fi

if [ ! -d "$DIR" ]; then
  echo "Error: '$DIR' is not a directory." >&2
  exit 2
fi

find "$DIR" -maxdepth 1 -type f -iname "*.zip" -print0 |
while IFS= read -r -d '' z; do
  echo "Unzipping: $z"
  if unzip -o "$z" -d "$(dirname "$z")" >/dev/null; then
    rm -f -- "$z"
    echo "Removed: $z"
  else
    echo "Failed to unzip: $z" >&2
  fi
done

echo "Done."
