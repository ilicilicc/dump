#!/bin/bash
# Test each model individually in separate processes to avoid memory accumulation

cd /home/user/webapp

echo "======================================="
echo "HST MODEL TPS BENCHMARK"
echo "Testing each model individually..."
echo "======================================="
echo ""

# Create Python script for single model test
cat > test_one.py << 'PYTHON_SCRIPT'
import torch
import torch.nn as nn
import sys
import importlib.util
import time
import gc

model_file = sys.argv[1]

def find_hst_class(module):
    for name in ["HSTv3Ultra", "HSTv4Unified", "HSTv5_2Unified", "HSTv6Giga",
                 "HSTv61", "HSTv7Agile", "HSTv7_1Ultimate", "HSTv7Ultimate", 
                 "HST", "HSTModel"]:
        if hasattr(module, name):
            return getattr(module, name), name
    for attr_name in dir(module):
        if attr_name.startswith('HST'):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                return attr, attr_name
    return None, None

try:
    # Load module
    spec = importlib.util.spec_from_file_location("m", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    model_class, class_name = find_hst_class(module)
    if not model_class:
        print(f"FAIL: No HST class found")
        sys.exit(1)
    
    # Try configs
    configs = [
        {'vocab_size': 5000, 'd_model': 128, 'n_layers': 6, 'n_heads': 4, 
         'max_seq_len': 512, 'horizon': 8, 'mode': 'token', 'chunk_size': 64, 'num_experts': 4},
        {'vocab_size': 5000, 'd_model': 128, 'n_layers': 6, 'n_heads': 4, 
         'max_seq_len': 512, 'horizon': 8, 'mode': 'token', 'chunk_size': 64},
        {'vocab_size': 5000, 'd_model': 128, 'n_layers': 6, 'n_heads': 4, 
         'max_seq_len': 512, 'horizon': 8},
        {'vocab_size': 5000, 'd_model': 128, 'n_layers': 6, 'n_heads': 4},
    ]
    
    model = None
    for config in configs:
        try:
            model = model_class(**config)
            break
        except:
            continue
    
    if not model:
        print(f"FAIL: Could not instantiate")
        sys.exit(1)
    
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    
    # Test forward pass
    seq_len = 128
    input_ids = torch.randint(0, 5000, (1, seq_len))
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(5):
            start = time.perf_counter()
            _ = model(input_ids)
            times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    tps = seq_len / avg_time
    
    print(f"SUCCESS|{class_name}|{param_count/1e6:.2f}|{tps:.2f}|{avg_time*1000:.2f}")
    
except Exception as e:
    print(f"FAIL: {str(e)[:100]}")
    sys.exit(1)
PYTHON_SCRIPT

# Test each model
for model in hst_v3_ultra.py hst_v4_unified.py hst_v5_2_unified.py hst_v6_giga.py hst.v6.1.py hst.v7.0_agile.py hst.v7.1_ultimate.py HST-chaos_logic_ai.py; do
    if [ -f "$model" ]; then
        echo "Testing: $model"
        timeout 60 python3 test_one.py "$model" 2>&1 | head -5
        echo ""
    fi
done

echo "======================================="
echo "Benchmark Complete"
echo "======================================="
