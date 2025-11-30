#!/usr/bin/env python3
"""
Test the remaining models that failed with "index out of range" error
These models need at least 8 layers
"""

import torch
import torch.nn as nn
import sys
import importlib.util
import time
import gc
import json

REMAINING_MODELS = [
    "hst.v6.1.py",
    "hst.v7.0_agile.py",
    "HST-chaos_logic_ai.py",
]

def find_hst_class(module):
    """Find the main HST model class in module"""
    for name in ["HSTv6Giga", "HSTv61", "HSTv7Agile", "HSTv7_1Ultimate", 
                 "HST", "HSTModel"]:
        if hasattr(module, name):
            return getattr(module, name), name
    
    for attr_name in dir(module):
        if attr_name.startswith('HST'):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                return attr, attr_name
    return None, None

def test_model(model_file, warmup=2, test_iters=5):
    """Test a single model for TPS"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_file}")
    print(f"{'='*80}")
    
    try:
        # Load module
        spec = importlib.util.spec_from_file_location("test_mod", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        model_class, class_name = find_hst_class(module)
        if not model_class:
            print("❌ No HST class found")
            return None
        
        print(f"✓ Found class: {class_name}")
        
        # Try with 8, 10, and 12 layers
        for n_layers in [8, 10, 12]:
            print(f"\nTrying with {n_layers} layers...")
            
            configs = [
                {'vocab_size': 5000, 'd_model': 128, 'n_layers': n_layers, 'n_heads': 4, 
                 'max_seq_len': 512, 'horizon': 8, 'mode': 'token', 'chunk_size': 64, 'num_experts': 4},
                {'vocab_size': 5000, 'd_model': 128, 'n_layers': n_layers, 'n_heads': 4, 
                 'max_seq_len': 512, 'horizon': 8, 'mode': 'token', 'chunk_size': 64},
                {'vocab_size': 5000, 'd_model': 128, 'n_layers': n_layers, 'n_heads': 4, 
                 'max_seq_len': 512, 'horizon': 8},
                {'vocab_size': 5000, 'd_model': 128, 'n_layers': n_layers, 'n_heads': 4},
            ]
            
            model = None
            used_config = None
            
            for config in configs:
                try:
                    model = model_class(**config)
                    used_config = config
                    break
                except Exception as e:
                    continue
            
            if not model:
                print(f"  ❌ Could not instantiate with {n_layers} layers")
                continue
            
            model.eval()
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Model instantiated")
            print(f"    Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
            
            # Try a forward pass
            try:
                seq_len = 128
                input_ids = torch.randint(0, used_config['vocab_size'], (1, seq_len))
                
                with torch.no_grad():
                    _ = model(input_ids)
                
                print(f"  ✓ Forward pass successful!")
                
                # Now do full benchmark
                results = {}
                for seq_len in [128, 256, 512]:
                    print(f"  Testing seq_len={seq_len}:", end=" ")
                    try:
                        input_ids = torch.randint(0, used_config['vocab_size'], (1, seq_len))
                        
                        # Warmup
                        with torch.no_grad():
                            for _ in range(warmup):
                                _ = model(input_ids)
                        
                        # Benchmark
                        times = []
                        with torch.no_grad():
                            for _ in range(test_iters):
                                start = time.perf_counter()
                                _ = model(input_ids)
                                times.append(time.perf_counter() - start)
                        
                        avg_time = sum(times) / len(times)
                        tps = seq_len / avg_time
                        
                        results[seq_len] = {
                            'avg_time_ms': avg_time * 1000,
                            'tps': tps,
                            'min_time_ms': min(times) * 1000,
                            'max_time_ms': max(times) * 1000,
                        }
                        
                        print(f"TPS={tps:.2f}, Time={avg_time*1000:.2f}ms")
                        
                    except Exception as e:
                        print(f"Error: {str(e)[:50]}")
                        break
                
                # Clean up and return results
                del model
                gc.collect()
                
                if results:
                    return {
                        'class_name': class_name,
                        'params_M': param_count / 1e6,
                        'config': used_config,
                        'n_layers': n_layers,
                        'results': results
                    }
                
            except Exception as e:
                print(f"  ❌ Forward pass failed: {str(e)[:80]}")
                del model
                gc.collect()
                continue
        
        print(f"❌ All layer configurations failed")
        return None
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None

def main():
    from pathlib import Path
    
    print("="*80)
    print("TESTING REMAINING HST MODELS")
    print("="*80)
    
    all_results = {}
    
    for model_file in REMAINING_MODELS:
        if not Path(model_file).exists():
            print(f"\n⚠️  Skipping {model_file} - not found")
            continue
        
        result = test_model(model_file)
        if result:
            all_results[model_file] = result
        
        # Cleanup between models
        gc.collect()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - ADDITIONAL RESULTS")
    print("="*80)
    
    if not all_results:
        print("No successful tests")
        return
    
    for seq_len in [128, 256, 512]:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 80)
        print(f"{'Model':<25} {'Layers':>7} {'Params(M)':>10} {'TPS':>12} {'Time(ms)':>12}")
        print("-" * 80)
        
        for model_name, result in all_results.items():
            if seq_len in result['results']:
                print(f"{model_name.replace('.py', ''):<25} {result['n_layers']:>7} "
                      f"{result['params_M']:>10.2f} "
                      f"{result['results'][seq_len]['tps']:>12.2f} "
                      f"{result['results'][seq_len]['avg_time_ms']:>12.2f}")
    
    # Save results
    output_file = "tps_remaining_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
