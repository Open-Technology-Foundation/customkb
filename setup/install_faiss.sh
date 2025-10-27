#!/usr/bin/env bash
#
# install_faiss.sh - Intelligent FAISS package installer for CustomKB
#
# This script automatically detects your system configuration and installs
# the appropriate FAISS package (CPU-only, GPU with CUDA 12, or GPU with CUDA 11).
#
# Usage:
#   ./setup/install_faiss.sh [--force] [--dry-run]
#
# Options:
#   --force     Force reinstallation even if FAISS is already installed
#   --dry-run   Show what would be installed without actually installing
#
# Environment Variables:
#   FAISS_VARIANT   Override auto-detection. Values: cpu, gpu-cu12, gpu-cu11, auto
#
# Examples:
#   ./setup/install_faiss.sh                    # Auto-detect and install
#   FAISS_VARIANT=cpu ./setup/install_faiss.sh  # Force CPU-only installation
#   ./setup/install_faiss.sh --dry-run          # Show what would be installed
#

set -euo pipefail

# Script directory
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Icons
readonly ICON_INFO="◉"
readonly ICON_SUCCESS="✓"
readonly ICON_ERROR="✗"
readonly ICON_WARNING="▲"

# Colors
if [[ -t 1 ]]; then
  readonly COLOR_RESET='\033[0m'
  readonly COLOR_GREEN='\033[0;32m'
  readonly COLOR_YELLOW='\033[0;33m'
  readonly COLOR_RED='\033[0;31m'
  readonly COLOR_BLUE='\033[0;34m'
else
  readonly COLOR_RESET=''
  readonly COLOR_GREEN=''
  readonly COLOR_YELLOW=''
  readonly COLOR_RED=''
  readonly COLOR_BLUE=''
fi

# Parse command-line arguments
FORCE_INSTALL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --force)
      FORCE_INSTALL=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      sed -n '2,/^$/p' "$0" | sed 's/^# //; s/^#//'
      exit 0
      ;;
    *)
      echo -e "${COLOR_RED}${ICON_ERROR} Unknown option: $1${COLOR_RESET}" >&2
      exit 1
      ;;
  esac
done

# Functions
print_info() {
  echo -e "${COLOR_BLUE}${ICON_INFO} $*${COLOR_RESET}"
}

print_success() {
  echo -e "${COLOR_GREEN}${ICON_SUCCESS} $*${COLOR_RESET}"
}

print_warning() {
  echo -e "${COLOR_YELLOW}${ICON_WARNING} $*${COLOR_RESET}"
}

print_error() {
  echo -e "${COLOR_RED}${ICON_ERROR} $*${COLOR_RESET}" >&2
}

detect_gpu() {
  # Check if NVIDIA GPU is present
  if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
      local gpu_name
      gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
      if [[ -n "$gpu_name" ]]; then
        print_info "Detected NVIDIA GPU: $gpu_name"
        return 0
      fi
    fi
  fi

  print_info "No NVIDIA GPU detected"
  return 1
}

detect_cuda_version() {
  local cuda_major=""

  # Try nvcc first
  if command -v nvcc &>/dev/null; then
    local nvcc_output
    nvcc_output="$(nvcc --version 2>/dev/null)" || true
    if [[ -n "$nvcc_output" ]]; then
      cuda_major="$(echo "$nvcc_output" | grep -oP 'release \K[0-9]+' || true)"
      if [[ -n "$cuda_major" ]]; then
        print_info "Detected CUDA version from nvcc: $cuda_major.x"
        echo "$cuda_major"
        return 0
      fi
    fi
  fi

  # Try nvidia-smi as fallback
  if command -v nvidia-smi &>/dev/null; then
    local driver_cuda
    driver_cuda="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" || true
    if [[ -n "$driver_cuda" ]]; then
      # Estimate CUDA version from driver
      # Driver >= 530: CUDA 12
      # Driver >= 520 and < 530: CUDA 11
      local driver_major="${driver_cuda%%.*}"
      if [[ $driver_major -ge 530 ]]; then
        print_info "NVIDIA Driver $driver_cuda supports CUDA 12"
        echo "12"
        return 0
      elif [[ $driver_major -ge 520 ]]; then
        print_info "NVIDIA Driver $driver_cuda supports CUDA 11"
        echo "11"
        return 0
      else
        print_warning "NVIDIA Driver $driver_cuda may not support recent CUDA versions"
      fi
    fi
  fi

  print_warning "Could not detect CUDA version"
  return 1
}

determine_faiss_variant() {
  # Check for manual override
  local variant="${FAISS_VARIANT:-auto}"
  variant="${variant,,}" # Convert to lowercase

  if [[ "$variant" != "auto" ]]; then
    case "$variant" in
      cpu|gpu-cu12|gpu-cu11)
        print_info "Using manually specified FAISS variant: $variant"
        echo "$variant"
        return 0
        ;;
      *)
        print_error "Invalid FAISS_VARIANT: $variant (must be cpu, gpu-cu12, gpu-cu11, or auto)"
        return 1
        ;;
    esac
  fi

  # Auto-detection
  print_info "Auto-detecting optimal FAISS variant..."

  if ! detect_gpu; then
    echo "cpu"
    return 0
  fi

  local cuda_version
  if cuda_version="$(detect_cuda_version)"; then
    if [[ "$cuda_version" == "12" ]]; then
      echo "gpu-cu12"
    elif [[ "$cuda_version" == "11" ]]; then
      echo "gpu-cu11"
    else
      print_warning "Unsupported CUDA version: $cuda_version, falling back to CPU"
      echo "cpu"
    fi
  else
    print_warning "Could not determine CUDA version, falling back to CPU"
    echo "cpu"
  fi
}

check_existing_faiss() {
  if python -c "import faiss" 2>/dev/null; then
    local faiss_version
    faiss_version="$(python -c "import faiss; print(faiss.__version__)" 2>/dev/null)" || faiss_version="unknown"
    print_info "FAISS is already installed (version: $faiss_version)"

    # Check which variant is installed
    local installed_variant=""
    if pip list 2>/dev/null | grep -q "faiss-gpu-cu12"; then
      installed_variant="gpu-cu12"
    elif pip list 2>/dev/null | grep -q "faiss-gpu-cu11"; then
      installed_variant="gpu-cu11"
    elif pip list 2>/dev/null | grep -q "faiss-cpu"; then
      installed_variant="cpu"
    else
      installed_variant="unknown"
    fi

    print_info "Installed FAISS variant: $installed_variant"
    echo "$installed_variant"
    return 0
  fi

  return 1
}

install_faiss() {
  local variant="$1"
  local requirements_file=""

  case "$variant" in
    cpu)
      requirements_file="$PROJECT_DIR/requirements-faiss-cpu.txt"
      ;;
    gpu-cu12)
      requirements_file="$PROJECT_DIR/requirements-faiss-gpu-cu12.txt"
      ;;
    gpu-cu11)
      requirements_file="$PROJECT_DIR/requirements-faiss-gpu-cu11.txt"
      ;;
    *)
      print_error "Invalid FAISS variant: $variant"
      return 1
      ;;
  esac

  if [[ ! -f "$requirements_file" ]]; then
    print_error "Requirements file not found: $requirements_file"
    return 1
  fi

  if [[ "$DRY_RUN" == true ]]; then
    print_info "[DRY RUN] Would install FAISS variant: $variant"
    print_info "[DRY RUN] Would run: pip install -r $requirements_file"
    return 0
  fi

  print_info "Installing FAISS variant: $variant"
  print_info "Using requirements file: $requirements_file"

  if pip install -r "$requirements_file"; then
    print_success "FAISS installed successfully"
    return 0
  else
    print_error "Failed to install FAISS"
    return 1
  fi
}

main() {
  print_info "CustomKB FAISS Installation Script"
  echo

  # Check if we're in a virtual environment
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    print_warning "No virtual environment detected"
    print_warning "It's recommended to run this script inside a Python virtual environment"
    echo
  fi

  # Check for existing installation
  local existing_variant=""
  if existing_variant="$(check_existing_faiss)"; then
    if [[ "$FORCE_INSTALL" != true ]]; then
      print_success "FAISS is already installed"
      print_info "Use --force to reinstall"
      return 0
    else
      print_warning "Forcing reinstallation (--force specified)"

      # Uninstall existing FAISS
      if [[ "$DRY_RUN" != true ]]; then
        print_info "Uninstalling existing FAISS..."
        pip uninstall -y faiss-cpu faiss-gpu-cu12 faiss-gpu-cu11 2>/dev/null || true
      fi
    fi
  fi

  # Determine which variant to install
  local variant
  if ! variant="$(determine_faiss_variant)"; then
    print_error "Failed to determine FAISS variant"
    return 1
  fi

  echo
  print_info "Selected FAISS variant: $variant"

  # Show variant details
  case "$variant" in
    cpu)
      print_info "CPU-only installation (no GPU acceleration)"
      ;;
    gpu-cu12)
      print_info "GPU installation for CUDA 12.x (requires NVIDIA Driver >= R530)"
      ;;
    gpu-cu11)
      print_info "GPU installation for CUDA 11.8 (requires NVIDIA Driver >= R520)"
      ;;
  esac

  echo

  # Install FAISS
  if install_faiss "$variant"; then
    echo
    print_success "Installation complete!"

    # Verify installation
    if [[ "$DRY_RUN" != true ]]; then
      if python -c "import faiss; print(f'FAISS version: {faiss.__version__}')" 2>/dev/null; then
        print_success "FAISS import successful"
      else
        print_error "FAISS import failed"
        return 1
      fi
    fi

    return 0
  else
    echo
    print_error "Installation failed"
    return 1
  fi
}

# Run main function
main "$@"

#fin
