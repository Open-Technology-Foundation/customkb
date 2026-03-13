# CustomKB Installation Guide

This comprehensive guide covers installation for all deployment scenarios including development machines, production servers, CPU-only systems, and Docker containers.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [FAISS Installation](#faiss-installation)
- [Deployment Scenarios](#deployment-scenarios)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## Prerequisites

### Required

- **Python**: 3.12 or higher
- **SQLite**: 3.45 or higher
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 2GB+ for base installation, additional space for knowledgebases

### Optional (for GPU acceleration)

- **NVIDIA GPU**: CUDA-compatible (Compute Capability 7.0 or higher)
- **CUDA Toolkit**: 11.8 or 12.x
- **NVIDIA Driver**: R520+ (for CUDA 11) or R530+ (for CUDA 12)

### API Keys

At least one of the following:
- **OpenAI**: For GPT models and text-embedding models
- **Anthropic**: For Claude models
- **Google**: For Gemini models and gemini-embedding
- **xAI**: For Grok models

## Quick Start

For users who want to get started quickly:

```bash
# 1. Clone repository
git clone https://github.com/Open-Technology-Foundation/customkb.git
cd customkb

# 2. Install all dependencies (creates .venv automatically)
uv sync --extra faiss-gpu-cu12 --extra mcp --extra test
# For CPU-only: uv sync --extra faiss-cpu --extra mcp --extra test

# 3. Install NLTK data
sudo ./setup/nltk_setup.py download cleanup

# 4. Configure API keys
export OPENAI_API_KEY="your-key-here"
export VECTORDBS="/var/lib/vectordbs"

# 5. Verify installation
source .venv/bin/activate
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
customkb version
```

## Detailed Installation

### 1. Clone Repository

```bash
git clone https://github.com/Open-Technology-Foundation/customkb.git
cd customkb
```

Or download a specific release:

```bash
wget https://github.com/Open-Technology-Foundation/customkb/archive/refs/tags/v1.1.0.tar.gz
tar -xzf v1.1.0.tar.gz
cd customkb-1.1.0
```

### 2. Install Dependencies

Dependencies are managed via `pyproject.toml` and `uv`. Install everything in one command:

```bash
# GPU systems (CUDA 12):
uv sync --extra faiss-gpu-cu12 --extra mcp --extra test

# GPU systems (CUDA 11):
uv sync --extra faiss-gpu-cu11 --extra mcp --extra test

# CPU-only systems:
uv sync --extra faiss-cpu --extra mcp --extra test
```

This creates a `.venv`, installs all dependencies including FAISS, and sets up the editable install.

### 3. Install NLTK Data

CustomKB requires NLTK stopwords and tokenizers for text processing:

```bash
# Full installation (recommended)
sudo ./setup/nltk_setup.py download cleanup

# Or check status first
./setup/nltk_setup.py status

# Or manual installation
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 4. Configure Environment Variables

Create a `.env` file or add to your shell profile:

```bash
# Required: At least one API key
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
export XAI_API_KEY="xai-..."

# Required: Knowledgebase storage location
export VECTORDBS="/var/lib/vectordbs"

# Optional: NLTK data location
export NLTK_DATA="/usr/share/nltk_data"

# Optional: Override default models
export VECTOR_MODEL="text-embedding-3-small"
export QUERY_MODEL="claude-sonnet-4-6"

# Optional: FAISS configuration
export FAISS_VARIANT="auto"  # or "cpu", "gpu-cu12", "gpu-cu11"
export FAISS_NO_GPU="0"      # Set to 1 to disable GPU
```

### 5. Create Knowledgebase Storage

```bash
# Create the base directory
sudo mkdir -p /var/lib/vectordbs

# Set ownership (adjust username as needed)
sudo chown -R $USER:$USER /var/lib/vectordbs

# Set permissions
chmod 755 /var/lib/vectordbs
```

## FAISS Installation

FAISS (Facebook AI Similarity Search) is required for vector operations. The installation method depends on your hardware.

### Automatic Installation (Recommended)

The install script auto-detects your GPU and CUDA version:

```bash
./setup/install_faiss.sh
```

**Features:**
- Detects NVIDIA GPU presence using nvidia-smi
- Determines CUDA version from nvcc or driver version
- Installs appropriate FAISS package automatically
- Verifies installation after completion

**Options:**
```bash
./setup/install_faiss.sh --dry-run   # Show what would be installed
./setup/install_faiss.sh --force     # Reinstall even if already present
./setup/install_faiss.sh --help      # Show help message
```

### Manual Installation

If you prefer manual control or the automatic installer fails:

**CPU-only systems:**
```bash
uv sync --extra faiss-cpu
```

**GPU with CUDA 12.x (NVIDIA Driver >= R530):**
```bash
uv sync --extra faiss-gpu-cu12
```

**GPU with CUDA 11.8 (NVIDIA Driver >= R520):**
```bash
uv sync --extra faiss-gpu-cu11
```

### Force Specific Variant

Override auto-detection:

```bash
# Force CPU-only (useful for testing)
FAISS_VARIANT=cpu ./setup/install_faiss.sh

# Force CUDA 12 (if detection fails)
FAISS_VARIANT=gpu-cu12 ./setup/install_faiss.sh

# Force CUDA 11
FAISS_VARIANT=gpu-cu11 ./setup/install_faiss.sh
```

### Verify FAISS Installation

```bash
# Check FAISS version
python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"

# Check which variant is installed
uv pip list | grep faiss

# Test GPU support (if applicable)
python -c "import faiss; print(f'GPU available: {hasattr(faiss, \"StandardGpuResources\")}')"
```

## Deployment Scenarios

### Development Machine (GPU)

Typical setup for development with NVIDIA GPU:

```bash
# Standard installation
uv sync --extra faiss-gpu-cu12 --extra mcp

# Enable GPU acceleration in config
cat > /var/lib/vectordbs/myproject/myproject.cfg << 'EOF'
[DEFAULT]
vector_model = text-embedding-3-small
query_model = claude-sonnet-4-6

[ALGORITHMS]
reranking_device = cuda
reranking_batch_size = 64
EOF
```

### Production Server (GPU)

Optimized for production with GPU:

```bash
# Install with specific CUDA version
FAISS_VARIANT=gpu-cu12 ./setup/install_faiss.sh

# Copy and customize the example configuration
cp example.cfg /var/lib/vectordbs/myproject/myproject.cfg

# Auto-optimize for production
customkb optimize myproject
```

### CPU-Only Server

For systems without GPU:

```bash
# Force CPU-only FAISS
FAISS_VARIANT=cpu ./setup/install_faiss.sh

# Configure for CPU
cat > /var/lib/vectordbs/myproject/myproject.cfg << 'EOF'
[DEFAULT]
vector_model = text-embedding-3-small
query_model = claude-sonnet-4-6

[ALGORITHMS]
reranking_device = cpu
reranking_batch_size = 32
EOF
```

### Docker Container

#### CPU-Only Container

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app/

# Install uv and dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --extra faiss-cpu --extra mcp

# Install NLTK data
RUN . /app/.venv/bin/activate && \
    python -m nltk.downloader stopwords punkt

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV VECTORDBS="/var/lib/vectordbs"

VOLUME ["/var/lib/vectordbs"]

CMD ["bash"]
```

#### GPU-Enabled Container

```dockerfile
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app/

# Install uv and dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --extra faiss-gpu-cu12 --extra mcp

# Install NLTK data
RUN . /app/.venv/bin/activate && \
    python -m nltk.downloader stopwords punkt

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV VECTORDBS="/var/lib/vectordbs"

VOLUME ["/var/lib/vectordbs"]

CMD ["bash"]
```

Build and run:

```bash
# Build
docker build -t customkb:latest .

# Run (CPU)
docker run -it --rm \
  -v /var/lib/vectordbs:/var/lib/vectordbs \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  customkb:latest

# Run (GPU)
docker run -it --rm --gpus all \
  -v /var/lib/vectordbs:/var/lib/vectordbs \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  customkb:latest
```

## Troubleshooting

### FAISS Installation Issues

**Problem**: `faiss-gpu-cu12` fails to install

**Solutions**:
```bash
# 1. Verify CUDA version
nvcc --version
nvidia-smi

# 2. Try CUDA 11 version instead
uv sync --extra faiss-gpu-cu11

# 3. Fall back to CPU version
uv sync --extra faiss-cpu
```

**Problem**: ImportError: libcudart.so.12 not found

**Solution**:
```bash
# Install CUDA runtime
sudo apt-get install cuda-runtime-12-0

# Or set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### NLTK Data Issues

**Problem**: NLTK data not found

**Solution**:
```bash
# Reinstall NLTK data
./setup/nltk_setup.py download cleanup

# Or set NLTK_DATA path
export NLTK_DATA=/usr/share/nltk_data
```

### Permission Issues

**Problem**: Cannot create knowledgebase

**Solution**:
```bash
# Fix ownership
sudo chown -R $USER:$USER /var/lib/vectordbs

# Fix permissions
sudo chmod -R 755 /var/lib/vectordbs
```

### Memory Issues

**Problem**: Out of memory during embedding

**Solution**:
```bash
# Optimize for your system memory
customkb optimize myproject --memory-gb 8

# Or manually reduce batch sizes in config
[API]
api_max_concurrency = 4

[PERFORMANCE]
embedding_batch_size = 50
```

## Verification

After installation, verify everything is working:

### 1. Check Python Environment

```bash
python --version  # Should be 3.12+
uv pip list | grep -E "(anthropic|openai|faiss|langchain)"
```

### 2. Verify FAISS

```bash
python -c "import faiss; print(f'FAISS {faiss.__version__} imported successfully')"
```

### 3. Test CustomKB

```bash
# Check version
customkb version

# Create test knowledgebase
mkdir -p /var/lib/vectordbs/test
cat > /var/lib/vectordbs/test/test.cfg << 'EOF'
[DEFAULT]
vector_model = text-embedding-3-small
query_model = claude-sonnet-4-6
EOF

# Create test document
echo "This is a test document for CustomKB." > /tmp/test.txt

# Import document
customkb database test /tmp/test.txt

# Generate embeddings
customkb embed test

# Query
customkb query test "What is this about?"
```

### 4. Check GPU Support (if applicable)

```bash
# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi

# Test GPU in CustomKB
customkb query test "test" --debug | grep -i cuda
```

## Next Steps

After successful installation:

1. Read the [Quick Start](../README.md#quick-start) guide
2. Review [GPU Acceleration](GPU_ACCELERATION.md) for optimization
3. Check [Performance Optimization Guide](performance_optimization_guide.md)
4. Review `example.cfg` for all available configuration options

## Getting Help

- **Documentation**: Check docs/ directory for detailed guides
- **Issues**: Report problems at https://github.com/Open-Technology-Foundation/customkb/issues
- **Examples**: See `example.cfg` for sample configuration

#fin
