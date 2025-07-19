#!/bin/bash
# Bash completion for customkb
#
# Installation:
# 1. Copy this file to /etc/bash_completion.d/customkb
#    sudo cp customkb_completion.bash /etc/bash_completion.d/customkb
# 2. Or source it in your .bashrc:
#    source /path/to/customkb_completion.bash

# Get knowledge base names from VECTORDBS directory
_get_kb_names() {
  local vectordbs="${VECTORDBS:-/var/lib/vectordbs}"
  if [[ -d "$vectordbs" ]]; then
    # List directories, excluding hidden ones and special directories
    local kbs=""
    for dir in "$vectordbs"/*; do
      if [[ -d "$dir" ]] \
          && [[ ! "$(basename "$dir")" =~ ^\. ]] \
          && [[ ! "$(basename "$dir")" =~ ^logs ]]; then
        kbs="${kbs} $(basename "$dir")"
      fi
    done
    echo "$kbs"
  fi
}

_customkb() {
  local cur prev words cword
  _init_completion || return

  local commands="query database embed edit optimize verify-indexes bm25 help version"
  
  # Common options that appear in multiple commands
  local common_opts="-v --verbose -q --quiet -d --debug"
  
  # First argument - command selection
  if [[ $cword -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )
    return 0
  fi

  # Get the command (first argument)
  local cmd="${words[1]}"

  # Handle options and arguments based on the command
  case "${cmd}" in
    query)
      case "${prev}" in
        -Q|--query_file)
          # Complete with files
          _filedir
          return 0
          ;;
        -f|--format)
          COMPREPLY=( $(compgen -W "xml json markdown plain" -- "${cur}") )
          return 0
          ;;
        -p|--prompt-template)
          COMPREPLY=( $(compgen -W "default instructive scholarly concise analytical conversational technical" -- "${cur}") )
          return 0
          ;;
        -R|--role|-m|--model|-k|--top-k|-s|--context-scope|-t|--temperature|-M|--max-tokens)
          # These options take free-form values
          return 0
          ;;
        *)
          if [[ ${cur} == -* ]]; then
            local query_opts="-Q --query_file -c --context --context-only -R --role -m --model -k --top-k -s --context-scope -t --temperature -M --max-tokens -f --format -p --prompt-template ${common_opts}"
            COMPREPLY=( $(compgen -W "${query_opts}" -- "${cur}") )
          elif [[ $cword -eq 2 ]]; then
            # Knowledge base name or config file
            local kb_names=$(_get_kb_names)
            COMPREPLY=( $(compgen -W "${kb_names}" -- "${cur}") )
            # Also complete with .cfg files
            _filedir 'cfg'
          fi
          ;;
      esac
      ;;

    database)
      case "${prev}" in
        -l|--language)
          # Complete with language codes and names
          local languages="zh chinese da danish nl dutch en english fi finnish fr french de german id indonesian it italian pt portuguese es spanish sv swedish"
          COMPREPLY=( $(compgen -W "${languages}" -- "${cur}") )
          return 0
          ;;
        *)
          if [[ ${cur} == -* ]]; then
            local database_opts="-l --language --detect-language -f --force ${common_opts}"
            COMPREPLY=( $(compgen -W "${database_opts}" -- "${cur}") )
          elif [[ $cword -eq 2 ]]; then
            # Knowledge base name or config file
            local kb_names=$(_get_kb_names)
            COMPREPLY=( $(compgen -W "${kb_names}" -- "${cur}") )
            # Also complete with .cfg files
            _filedir 'cfg'
          else
            # File patterns for database command - common text formats
            _filedir '@(txt|md|markdown|html|htm|xml|rst|tex|log|json|yaml|yml|py|js|java|c|cpp|go|rs|php|rb|ts|swift)'
          fi
          ;;
      esac
      ;;

    embed)
      if [[ ${cur} == -* ]]; then
        local embed_opts="-r --reset-database ${common_opts}"
        COMPREPLY=( $(compgen -W "${embed_opts}" -- "${cur}") )
      elif [[ $cword -eq 2 ]]; then
        # Knowledge base name or config file
        local kb_names=$(_get_kb_names)
        COMPREPLY=( $(compgen -W "${kb_names}" -- "${cur}") )
        # Also complete with .cfg files
        _filedir 'cfg'
      fi
      ;;

    edit)
      if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${common_opts}" -- "${cur}") )
      elif [[ $cword -eq 2 ]]; then
        # Knowledge base name or config file
        local kb_names=$(_get_kb_names)
        COMPREPLY=( $(compgen -W "${kb_names}" -- "${cur}") )
        # Also complete with .cfg files
        _filedir 'cfg'
      fi
      ;;

    optimize)
      case "${prev}" in
        --memory-gb)
          # Numeric value expected
          return 0
          ;;
        *)
          if [[ ${cur} == -* ]]; then
            local optimize_opts="--dry-run --analyze --show-tiers --memory-gb ${common_opts}"
            COMPREPLY=( $(compgen -W "${optimize_opts}" -- "${cur}") )
          elif [[ $cword -eq 2 ]]; then
            # Knowledge base name or config file
            local kb_names=$(_get_kb_names)
            COMPREPLY=( $(compgen -W "${kb_names}" -- "${cur}") )
            # Also complete with .cfg files
            _filedir 'cfg'
          fi
          ;;
      esac
      ;;

    verify-indexes)
      if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${common_opts}" -- "${cur}") )
      elif [[ $cword -eq 2 ]]; then
        # Knowledge base name or config file
        local kb_names=$(_get_kb_names)
        COMPREPLY=( $(compgen -W "${kb_names}" -- "${cur}") )
        # Also complete with .cfg files
        _filedir 'cfg'
      fi
      ;;

    bm25)
      if [[ ${cur} == -* ]]; then
        local bm25_opts="--force"
        COMPREPLY=( $(compgen -W "${bm25_opts}" -- "${cur}") )
      elif [[ $cword -eq 2 ]]; then
        # Knowledge base name or config file
        local kb_names=$(_get_kb_names)
        COMPREPLY=( $(compgen -W "${kb_names}" -- "${cur}") )
        # Also complete with .cfg files
        _filedir 'cfg'
      fi
      ;;

    version)
      if [[ ${cur} == -* ]]; then
        local version_opts="--build"
        COMPREPLY=( $(compgen -W "${version_opts}" -- "${cur}") )
      fi
      ;;

    help)
      # No additional options for help
      ;;

    *)
      # Unknown command
      ;;
  esac

  return 0
}

complete -F _customkb customkb

#fin