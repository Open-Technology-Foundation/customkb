#!/bin/bash
#shellcheck disable=SC2155
set -euo pipefail

# CustomKB version management script
# Usage: ./version.sh [major|minor|patch]

readonly VERSION_FILE="version.py"

bump_version() {
  local type=$1
  
  # Extract current version components
  local major=$(grep "VERSION_MAJOR =" "$VERSION_FILE" | sed 's/VERSION_MAJOR = //')
  local minor=$(grep "VERSION_MINOR =" "$VERSION_FILE" | sed 's/VERSION_MINOR = //')
  local patch=$(grep "VERSION_PATCH =" "$VERSION_FILE" | sed 's/VERSION_PATCH = //')
  
  # Bump requested component
  case "$type" in
    major)
      major=$((major + 1))
      minor=0
      patch=0
      ;;
    minor)
      minor=$((minor + 1))
      patch=0
      ;;
    patch)
      patch=$((patch + 1))
      ;;
    *)
      >&2 echo "Invalid version type. Use major, minor, or patch."
      exit 1
      ;;
  esac
  
  # Update version file
  sed -i "s/VERSION_MAJOR = .*/VERSION_MAJOR = $major/" "$VERSION_FILE"
  sed -i "s/VERSION_MINOR = .*/VERSION_MINOR = $minor/" "$VERSION_FILE"
  sed -i "s/VERSION_PATCH = .*/VERSION_PATCH = $patch/" "$VERSION_FILE"
  sed -i "s/VERSION_BUILD = .*/VERSION_BUILD = 1/" "$VERSION_FILE"
  sed -i "s/RELEASE_DATE = .*/RELEASE_DATE = \"$(date +%Y-%m-%d)\"/" "$VERSION_FILE"
  
  echo "Version bumped to $major.$minor.$patch (build 1)"
}

bump_build() {
  # Extract current build number
  local build=$(grep "VERSION_BUILD =" "$VERSION_FILE" | sed 's/VERSION_BUILD = //')
  
  # Increment build number
  build=$((build + 1))
  
  # Update version file
  sed -i "s/VERSION_BUILD = .*/VERSION_BUILD = $build/" "$VERSION_FILE"
  
  # Extract current version
  local major=$(grep "VERSION_MAJOR =" "$VERSION_FILE" | sed 's/VERSION_MAJOR = //')
  local minor=$(grep "VERSION_MINOR =" "$VERSION_FILE" | sed 's/VERSION_MINOR = //')
  local patch=$(grep "VERSION_PATCH =" "$VERSION_FILE" | sed 's/VERSION_PATCH = //')
  
  echo "Build number bumped to $major.$minor.$patch.$build"
}

# Main logic
if (($#==0)) || [[ $* == *"-h"*  ]]; then
  echo "Usage: $0 [major|minor|patch|build]"
  echo "  major  - Bump major version (X.0.0), resets minor and patch"
  echo "  minor  - Bump minor version (x.X.0), resets patch"
  echo "  patch  - Bump patch version (x.x.X)"
  echo "  build  - Bump build number only (x.x.x.X)"
  exit 0
fi

# Ensure version.py exists
if [[ ! -f "$VERSION_FILE" ]]; then
  >&2 echo "Error: $VERSION_FILE not found"
  exit 1
fi

# Process command
case "$1" in
  build)
    bump_build
    ;;
  major|minor|patch)
    bump_version "$1"
    ;;
  version|-V|--version)
    grep '^VERSION_[MMPB]?*' "$VERSION_FILE" 
    ;;
  *)
    >&2 echo "Invalid argument: $1"
    >&2 echo "Usage: $0 [major|minor|patch|build]"
    exit 1
    ;;
esac

#fin