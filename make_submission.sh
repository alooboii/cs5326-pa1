#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "PA1.pdf" || ! -d "src" || ! -f "tests/adapters.py" ]]; then
  echo "Run this script from the assignment repository root." >&2
  exit 1
fi

required_files=(
  "REPORT.md"
  "final_model.pt"
  "logs/final_training.csv"
  "logs/final_metrics.json"
  "logs/generations.json"
)
for required in "${required_files[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required submission file: $required" >&2
    exit 1
  fi
done

if ! find src -type f -name '*.py' -print -quit | grep -q .; then
  echo "The src/ directory does not contain a Python implementation." >&2
  exit 1
fi
if ! find report_assets -type f -print -quit | grep -q .; then
  echo "report_assets/ must contain the figures referenced by REPORT.md." >&2
  exit 1
fi

test_output="public_tests.txt"
set +e
uv run pytest -q 2>&1 | tee "$test_output"
test_status=${PIPESTATUS[0]}
set -e
if [[ "$test_status" -ne 0 ]]; then
  echo "Public tests failed; the failure output will still be packaged." >&2
fi

uv run python - <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

model_path = Path("final_model.pt")
if model_path.stat().st_size > 50 * 1024 * 1024:
    raise SystemExit("final_model.pt exceeds the 50 MiB model-artifact limit")

state = torch.load(model_path, map_location="cpu", weights_only=True)
if not isinstance(state, dict) or not state:
    raise SystemExit("final_model.pt must contain a non-empty state dictionary")
if not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()):
    raise SystemExit("final_model.pt must map parameter names directly to tensors")
if sum(tensor.numel() for tensor in state.values()) != 19_272_192:
    raise SystemExit("final_model.pt does not contain exactly 19,272,192 values")
for name, tensor in state.items():
    if tensor.device.type != "cpu":
        raise SystemExit(f"{name} is not stored on CPU")
    if tensor.is_floating_point():
        if tensor.dtype != torch.float16:
            raise SystemExit(f"{name} is not stored as float16")
        if not torch.isfinite(tensor).all():
            raise SystemExit(f"{name} contains a non-finite value")

with Path("logs/final_training.csv").open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    required_columns = {
        "completed_steps",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "grad_norm",
    }
    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise SystemExit("logs/final_training.csv is missing required columns")
    rows = list(reader)
    if not rows or not any(int(row["completed_steps"]) == 10_000 for row in rows):
        raise SystemExit("logs/final_training.csv must include completed_steps=10000")

metrics = json.loads(Path("logs/final_metrics.json").read_text(encoding="utf-8"))
required_metrics = {
    "completed_steps",
    "validation_seed",
    "validation_batches",
    "validation_batch_size",
    "sequence_length",
    "validation_loss",
    "perplexity",
}
if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
    raise SystemExit("logs/final_metrics.json is missing required fields")
expected = {
    "completed_steps": 10_000,
    "validation_seed": 42,
    "validation_batches": 100,
    "validation_batch_size": 16,
    "sequence_length": 256,
}
for key, value in expected.items():
    if metrics[key] != value:
        raise SystemExit(f"logs/final_metrics.json requires {key}={value}")

generations = json.loads(Path("logs/generations.json").read_text(encoding="utf-8"))
required_generation_fields = {
    "prompt_text",
    "prompt_ids",
    "output_text",
    "output_ids",
    "temperature",
    "top_p",
    "seed",
    "max_new_tokens",
}
if not isinstance(generations, list) or len(generations) < 6:
    raise SystemExit("logs/generations.json must contain at least six generation records")
for index, record in enumerate(generations):
    if not isinstance(record, dict) or not required_generation_fields.issubset(record):
        raise SystemExit(f"generation record {index} is missing required fields")
PY

staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/cs5236-pa1-submission.XXXXXX")"
trap 'rm -rf "$staging_dir"' EXIT

mkdir -p "$staging_dir/tests"
cp -R src "$staging_dir/src"
cp tests/adapters.py "$staging_dir/tests/adapters.py"
cp REPORT.md final_model.pt public_tests.txt "$staging_dir/"
cp -R logs report_assets "$staging_dir/"

TEST_STATUS="$test_status" uv run python - "$staging_dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
staging = Path(sys.argv[1])
hashed = [
    "final_model.pt",
    "REPORT.md",
    "logs/final_training.csv",
    "logs/final_metrics.json",
    "logs/generations.json",
]

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    )
except (FileNotFoundError, subprocess.CalledProcessError):
    commit = None
    dirty = None

manifest = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": commit,
    "git_dirty": dirty,
    "public_test_exit_status": int(os.environ["TEST_STATUS"]),
    "sha256": {name: digest(root / name) for name in hashed},
}
(staging / "submission_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

find "$staging_dir" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$staging_dir" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

archive="submission.zip"
rm -f "$archive"
(cd "$staging_dir" && zip -qr "$OLDPWD/$archive" .)

archive_size=$(wc -c < "$archive" | tr -d ' ')
if (( archive_size > 60 * 1024 * 1024 )); then
  rm -f "$archive"
  echo "submission.zip exceeds the 60 MiB LMS limit." >&2
  exit 1
fi

echo "Created $archive ($(du -h "$archive" | awk '{print $1}'))."
unzip -l "$archive"
