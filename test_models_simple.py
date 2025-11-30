#!/usr/bin/env python3
"""
Simple model instantiation test to see which models can be loaded
"""

import torch
import torch.nn as nn
import sys
import importlib.util
from pathlib import Path

MODEL_FILES = [
    "hst_v3_ultra.py",
    "hst_v4_unified.py",
    "hst_v5_2_unified.py",
    "hst_v6_giga.py",
    "hst.v6.1.py",
    "hst.v7.0_agile.py",
    "hst.v7.1_ultimate.py",
    "HST-chaos_logic_ai.py",
]

def find_hst_class(module):
    """Find the main HST model class"""
    possible_names = [
        "HSTv3Ultra", "HSTv4Unified", "HSTv5_2Unified", "HSTv6Giga",
        "HSTv61", "HSTv7Agile", "HSTv7_1Ultimate", "HSTv7Ultimate",
        "HST", "HSTModel"
    ]
    
    for name in possible_names:
        if hasattr(module, name):
            return getattr(module, name), name
    
    # Find any class starting with HST
    for attr_name in dir(module):
        if attr_name.startswith('HST'):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                return attr, attr_name
    
    return None, None

def test_model(model_file):
    """Test if model can be instantiated"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_file}")
    print(f"{'='*80}")
    
    model_path = Path("/home/user/webapp") / model_file
    if not model_path.exists():
        print("❌ File not found")
        return False
    
    try:
        # Load module
        spec = importlib.util.spec_from_file_location("test_module", str(model_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_module"] = module
        spec.loader.exec_module(module)
        
        model_class, class_name = find_hst_class(module)
        if model_class is None:
            print("❌ No HST model class found")
            return False
        
        print(f"✓ Found class: {class_name}")
        
        # Try to instantiate with different configs
        configs = [
            # Config 1: Full params
            {
                'vocab_size': 10000,
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 4,
                'max_seq_len': 1024,
                'horizon': 8,
                'mode': 'token',
                'chunk_size': 64,
                'num_experts': 4,
            },
            # Config 2: No MoE
            {
                'vocab_size': 10000,
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 4,
                'max_seq_len': 1024,
                'horizon': 8,
                'mode': 'token',
                'chunk_size': 64,
            },
            # Config 3: No mode
            {
                'vocab_size': 10000,
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 4,
                'max_seq_len': 1024,
                'horizon': 8,
            },
            # Config 4: Minimal
            {
                'vocab_size': 10000,
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 4,
            },
        ]
        
        model = None
        for i, config in enumerate(configs):
            try:
                model = model_class(**config)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✓ Instantiated with config {i+1}")
                print(f"  Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
                
                # Try a forward pass
                input_ids = torch.randint(0, config['vocab_size'], (1, 128))
                with torch.no_grad():
                    output = model(input_ids)
                print(f"✓ Forward pass successful")
                print(f"  Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
                
                del model
                return True
                
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"  Config {i+1} failed: {error_msg}")
                continue
        
        print("❌ All configs failed")
        return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}")
        return False

def main():
    print("="*80)
    print("HST MODEL INSTANTIATION TEST")
    print("="*80)
    
    results = {}
    for model_file in MODEL_FILES:
        success = test_model(model_file)
        results[model_file] = success
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    working = [k for k, v in results.items() if v]
    failing = [k for k, v in results.items() if not v]
    
    print(f"\n✓ Working models ({len(working)}):")
    for model in working:
        print(f"  - {model}")
    
    print(f"\n❌ Failing models ({len(failing)}):")
    for model in failing:
        print(f"  - {model}")

if __name__ == "__main__":
    main()
