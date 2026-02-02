#!/bin/bash

declare -a cmds=(
  'customkb help'
  'customkb database --help'
  'customkb embed --help'
  'customkb query --help'
  'customkb edit --help'
  'customkb optimize --help'
  'customkb verify-indexes --help'
  'customkb bm25 --help'
)

{ echo "# CustomKB -- Custom Knowledgebases"
  echo
  for cmd in "${cmds[@]}"; do
    echo "## $cmd"
    echo
    eval "$cmd"
    echo
    echo '---'
    echo
  done
} >README.md

#fin
