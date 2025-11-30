#!/usr/bin/env python3
"""
Efficient TPS (Tokens Per Second) Benchmark for HST Models
Memory-efficient version that tests smaller configurations
"""

import torch
import torch.nn as nn
import time
import traceback
from pathlib import Path
import sys
import json
import gc

# Model file names in the repo
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

class TPS_Tester:
    def __init__(self, warmup_iterations=2, test_iterations=5):
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()

    def load_model_from_file(self, model_file):
        """Dynamically load model class from Python file"""
        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_module", model_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["model_module"] = module
            spec.loader.exec_module(module)
            
            # Try to find the main model class
            possible_names = [
                "HSTv3Ultra", "HSTv4Unified", "HSTv5_2Unified", "HSTv6Giga",
                "HSTv61", "HSTv7Agile", "HSTv7_1Ultimate", "HSTv7Ultimate",
                "HST", "HSTModel"
            ]
            
            model_class = None
            for name in possible_names:
                if hasattr(module, name):
                    model_class = getattr(module, name)
                    print(f"  Found model class: {name}")
                    break
            
            if model_class is None:
                # Find any class that starts with HST
                for attr_name in dir(module):
                    if attr_name.startswith('HST'):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, nn.Module):
                            model_class = attr
                            print(f"  Found model class: {attr_name}")
                            break
            
            return model_class
        except Exception as e:
            print(f"  Error loading: {str(e)}")
            return None

    def test_model_tps(self, model_class, model_name):
        """Test a single model for TPS with small config"""
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Use very small config to avoid memory issues
            small_config = {
                'vocab_size': 10000,
                'd_model': 256,
                'n_layers': 6,  # Use 6 layers minimum to avoid index errors
                'n_heads': 4,
                'chunk_size': 64,
                'max_seq_len': 1024,
                'horizon': 8,
                'mode': 'token',
                'num_experts': 4,
            }
            
            # Try different parameter combinations
            model = None
            configs_to_try = [
                # Try with all params
                small_config,
                # Without num_experts
                {k: v for k, v in small_config.items() if k != 'num_experts'},
                # Without mode and num_experts
                {k: v for k, v in small_config.items() if k not in ['num_experts', 'mode']},
                # Minimal
                {
                    'vocab_size': small_config['vocab_size'],
                    'd_model': small_config['d_model'],
                    'n_heads': small_config['n_heads'],
                    'n_layers': small_config['n_layers']
                }
            ]
            
            last_error = None
            for i, config in enumerate(configs_to_try):
                try:
                    model = model_class(**config)
                    print(f"✓ Model instantiated (config attempt {i+1})")
                    break
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if model is None:
                print(f"❌ Failed to instantiate model")
                print(f"   Last error: {last_error}")
                return None
            
            model = model.to(self.device)
            model.eval()
            
            # Get model size
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
            
            # Test with small sequence length
            results = {}
            seq_len = 256  # Start with small sequence
            
            print(f"\nTesting sequence length: {seq_len}")
            
            try:
                # Create test data
                input_ids = torch.randint(0, small_config['vocab_size'], 
                                         (1, seq_len), device=self.device)
                
                # Warmup
                print(f"  Warming up...")
                with torch.no_grad():
                    for _ in range(self.warmup_iterations):
                        try:
                            output = model(input_ids)
                        except Exception as e:
                            print(f"  Warmup error: {type(e).__name__}: {str(e)[:100]}")
                            return None
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                print(f"  Running benchmark...")
                times = []
                
                with torch.no_grad():
                    for _ in range(self.test_iterations):
                        start_time = time.perf_counter()
                        _ = model(input_ids)
                        
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                
                # Statistics
                avg_time = sum(times) / len(times)
                tps = seq_len / avg_time
                
                results[seq_len] = {
                    'avg_time_ms': avg_time * 1000,
                    'tps': tps,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'params_M': param_count / 1e6,
                }
                
                print(f"  ✓ Average time: {avg_time*1000:.2f} ms")
                print(f"  ✓ TPS: {tps:.2f} tokens/second")
                print(f"  ✓ Range: {min(times)*1000:.2f} - {max(times)*1000:.2f} ms")
            
            except Exception as e:
                print(f"  ❌ Benchmark error: {str(e)}")
                traceback.print_exc()
            
            # Clean up
            del model
            if 'input_ids' in locals():
                del input_ids
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return results
        
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            traceback.print_exc()
            return None

    def run_all_tests(self):
        """Test all models"""
        print("\n" + "="*80)
        print("HST MODEL TPS BENCHMARK (Efficient Mode)")
        print("="*80)
        
        all_results = {}
        
        for model_file in MODEL_FILES:
            model_path = Path("/home/user/webapp") / model_file
            
            if not model_path.exists():
                print(f"\n⚠️  Skipping {model_file} - file not found")
                continue
            
            model_class = self.load_model_from_file(str(model_path))
            
            if model_class is None:
                print(f"❌ Could not load model from {model_file}")
                continue
            
            results = self.test_model_tps(model_class, model_file)
            
            if results:
                all_results[model_file] = results
            
            # Force cleanup between models
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results

    def print_summary(self, all_results):
        """Print summary"""
        print("\n" + "="*80)
        print("SUMMARY - TPS COMPARISON")
        print("="*80)
        
        if not all_results:
            print("No successful tests to summarize.")
            return
        
        print(f"\n{'Model':<40} {'Params(M)':>12} {'TPS':>15} {'Time (ms)':>15}")
        print("-" * 80)
        
        model_stats = []
        for model_name, results in all_results.items():
            for seq_len, metrics in results.items():
                model_stats.append({
                    'model': model_name,
                    'params': metrics['params_M'],
                    'tps': metrics['tps'],
                    'time': metrics['avg_time_ms']
                })
        
        # Sort by TPS
        model_stats.sort(key=lambda x: x['tps'], reverse=True)
        
        for stat in model_stats:
            print(f"{stat['model']:<40} {stat['params']:>12.2f} {stat['tps']:>15.2f} {stat['time']:>15.2f}")
        
        print("\n" + "="*80)


def main():
    tester = TPS_Tester(warmup_iterations=2, test_iterations=5)
    results = tester.run_all_tests()
    
    # Save results
    output_file = "/home/user/webapp/tps_results.json"
    
    serializable_results = {}
    for model, seq_results in results.items():
        serializable_results[model] = {
            str(seq_len): metrics for seq_len, metrics in seq_results.items()
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
