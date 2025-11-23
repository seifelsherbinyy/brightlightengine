#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 ]];
then
  CONFIG_PATH="$1"
  shift
else
  CONFIG_PATH="${PROJECT_ROOT}/config.yaml"
fi

python -m brightstar.brightstar_pipeline --config "${CONFIG_PATH}" "$@"
