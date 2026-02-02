#!/bin/bash
# CUDA Verification Test Script
# Run this after rebooting to verify CUDA works with PyTorch

echo "=== CUDA Verification Test ==="
echo ""

echo "1. NVIDIA Driver Info:"
nvidia-smi --query-gpu=driver_version,name,compute_mode --format=csv,noheader
echo ""

echo "2. CUDA Version:"
nvidia-smi | grep "CUDA Version" | awk '{print $9}'
echo ""

echo "3. Kernel Module:"
modinfo nvidia | grep -E "^filename:|^version:"
echo ""

echo "4. PyTorch CUDA Test:"
source .venv/bin/activate
python3 << 'EOF'
import torch
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA compiled version: {torch.version.cuda}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✓ CUDA device count: {torch.cuda.device_count()}")
    print(f"   ✓ CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA capability: {torch.cuda.get_device_capability(0)}")
    # Test tensor operation
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x @ y
    print(f"   ✓ GPU tensor operation successful: {z.device}")
else:
    print("   ✗ CUDA NOT available")
    import ctypes
    try:
        libcuda = ctypes.CDLL('libcuda.so.1')
        result = libcuda.cuInit(0)
        print(f"   cuInit result: {result}")
    except Exception as e:
        print(f"   Error: {e}")
EOF

echo ""
echo "=== Test Complete ==="

#fin
