#!/usr/bin/env bash
set -euo pipefail
client_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$client_dir/irodori/.venv/bin/python" "$client_dir/app.py" "$@"
