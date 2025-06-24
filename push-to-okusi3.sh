#!/bin/bash
set -euo pipefail

[[ "$HOSTNAME" == okusi ]] || exit 1

cd /ai/scripts/customkb

cln -Nq

declare dryrun='n'
[[ "${1:-}" != '-N' ]] || dryrun=''

declare -a excludes=(--exclude .venv --exclude __pycache__/ --exclude .git --exclude .cache --exclude .pytest_cache/)

rsync -avl$dryrun . okusi3:"$PWD"/ "${excludes[@]}"
