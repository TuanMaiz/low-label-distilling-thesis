#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m rationales.validate_rationales \
  --config "${1:-configs/phase02_rationales.json}"
