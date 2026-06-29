#!/usr/bin/env bash
# PostToolUse hook: format edited Python files with black + isort using the 3mldev venv.
# Receives the tool-call payload as JSON on stdin; extracts the edited file path.
set -euo pipefail

VENV=/Users/jburgess/.environs/3mldev/bin

file_path=$(jq -r '.tool_input.file_path // empty')

# Only act on Python source files
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

[ -f "$file_path" ] || exit 0

"$VENV/black" -q "$file_path" 2>/dev/null || true
"$VENV/isort" -q "$file_path" 2>/dev/null || true

exit 0
