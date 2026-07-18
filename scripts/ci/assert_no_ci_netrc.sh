#!/usr/bin/bash
set -euo pipefail

while IFS= read -r variable_name; do
  if [[ -n "${!variable_name}" ]]; then
    printf 'ordinary CI step received non-empty %s\n' "${variable_name}" >&2
    exit 1
  fi
done < <(compgen -A variable CI_NETRC_)
