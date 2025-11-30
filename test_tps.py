#!/usr/bin/env python3
"""
TPS (Tokens Per Second) Benchmark for HST Models
Tests all model variants in the repository for inference speed
"""

import torch
import torch.nn as nn
import time
import traceback
from pathlib import Path
import sys

# Model file names in the repo
MODEL_FILES = [
    "hst_v3_ultra.py",
    "hst_v4_unified.py",
    "hst_v5_2_unified.py",
    "hst_v6_giga.py",
    "hst.v6.1.py",
    "hst.v7.0_agile.py",
    "hst.v7.1_ultimate.py",
    "hst_v7_1_ultimate.py",
    "HST-chaos_logic_ai.py",
    "HST-error_networks.py",
]

class TPS_Tester:
    def __init__(self, warmup_iterations=3, test_iterations=10):
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
            
            # Try to find the main model class (usually named HST or similar)
            possible_names = [
                "HSTv3Ultra", "HSTv4Unified", "HSTv5_2Unified", "HSTv6Giga",
                "HSTv61", "HSTv7Agile", "HSTv7_1Ultimate", "HSTv7Ultimate",
                "HST", "HSTModel", "HierarchicalStateTransformer",
                "ChaoticHST", "ErrorNetworkHST", "AgileHST", "UltimateHST",
                "UnifiedHST", "GigaHST", "UltraHST"
            ]
            
            model_class = None
            for name in possible_names:
                if hasattr(module, name):
                    model_class = getattr(module, name)
                    break
            
            if model_class is None:
                # Find any class that starts with HST and inherits from nn.Module
                for attr_name in dir(module):
                    if attr_name.startswith('HST'):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                            model_class = attr
                            break
            
            return model_class
        except Exception as e:
            print(f"Error loading {model_file}: {str(e)}")
            traceback.print_exc()
            return None

    def create_test_data(self, batch_size=1, seq_len=512):
        """Create test input data"""
        return {
            'input_ids': torch.randint(0, 50000, (batch_size, seq_len), device=self.device)
        }

    def test_model_tps(self, model_class, model_name, config=None):
        """Test a single model for TPS"""
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Default config if none provided
            if config is None:
                config = {
                    'vocab_size': 50000,
                    'd_model': 512,
                    'n_layers': 6,
                    'n_heads': 8,
                    'chunk_size': 128,
                    'max_seq_len': 2048,
                    'horizon': 16,
                    'mode': 'token',
                    'num_experts': 8,
                }
            
            # Try to instantiate the model with various parameter combinations
            model = None
            last_error = None
            attempts = [
                # Full config with all parameters
                lambda: model_class(**config),
                # Without num_experts (for non-MoE models)
                lambda: model_class(
                    vocab_size=config['vocab_size'],
                    d_model=config['d_model'],
                    n_layers=config['n_layers'],
                    n_heads=config['n_heads'],
                    max_seq_len=config['max_seq_len'],
                    horizon=config['horizon'],
                    mode=config['mode'],
                    chunk_size=config['chunk_size']
                ),
                # Without mode (for older models)
                lambda: model_class(
                    vocab_size=config['vocab_size'],
                    d_model=config['d_model'],
                    n_layers=config['n_layers'],
                    n_heads=config['n_heads'],
                    max_seq_len=config['max_seq_len'],
                    horizon=config['horizon']
                ),
                # Minimal config
                lambda: model_class(
                    vocab_size=config['vocab_size'],
                    d_model=config['d_model'],
                    n_heads=config['n_heads'],
                    n_layers=config['n_layers']
                ),
            ]
            
            for i, attempt in enumerate(attempts):
                try:
                    model = attempt()
                    print(f"✓ Model instantiated successfully (attempt {i+1})")
                    break
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if model is None:
                print(f"❌ Failed to instantiate model with available configs")
                if last_error:
                    print(f"   Last error: {last_error}")
                return None
            
            model = model.to(self.device)
            model.eval()
            
            # Get model size
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
            
            # Test different sequence lengths
            results = {}
            for seq_len in [128, 256, 512, 1024]:
                print(f"\nTesting sequence length: {seq_len}")
                
                try:
                    test_data = self.create_test_data(batch_size=1, seq_len=seq_len)
                    
                    # Warmup
                    print(f"  Warming up ({self.warmup_iterations} iterations)...")
                    with torch.no_grad():
                        for _ in range(self.warmup_iterations):
                            try:
                                _ = model(test_data['input_ids'])
                            except TypeError:
                                # Try without dict wrapper
                                _ = model(test_data['input_ids'])
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Actual benchmark
                    print(f"  Running benchmark ({self.test_iterations} iterations)...")
                    times = []
                    
                    with torch.no_grad():
                        for _ in range(self.test_iterations):
                            start_time = time.perf_counter()
                            
                            try:
                                _ = model(test_data['input_ids'])
                            except:
                                _ = model(test_data['input_ids'])
                            
                            if self.device.type == 'cuda':
                                torch.cuda.synchronize()
                            
                            end_time = time.perf_counter()
                            times.append(end_time - start_time)
                    
                    # Calculate statistics
                    avg_time = sum(times) / len(times)
                    tps = seq_len / avg_time
                    
                    results[seq_len] = {
                        'avg_time_ms': avg_time * 1000,
                        'tps': tps,
                        'min_time_ms': min(times) * 1000,
                        'max_time_ms': max(times) * 1000,
                    }
                    
                    print(f"  ✓ Average time: {avg_time*1000:.2f} ms")
                    print(f"  ✓ TPS: {tps:.2f} tokens/second")
                    print(f"  ✓ Range: {min(times)*1000:.2f} - {max(times)*1000:.2f} ms")
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  ⚠️  Out of memory for sequence length {seq_len}")
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        break
                    else:
                        print(f"  ❌ Error: {str(e)}")
                        break
                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    break
            
            # Clean up
            del model
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return results
        
        except Exception as e:
            print(f"❌ Failed to test model: {str(e)}")
            traceback.print_exc()
            return None

    def run_all_tests(self):
        """Test all models in the repository"""
        print("\n" + "="*80)
        print("HST MODEL TPS BENCHMARK")
        print("="*80)
        
        all_results = {}
        
        for model_file in MODEL_FILES:
            model_path = Path("/home/user/webapp") / model_file
            
            if not model_path.exists():
                print(f"\n⚠️  Skipping {model_file} - file not found")
                continue
            
            model_class = self.load_model_from_file(str(model_path))
            
            if model_class is None:
                print(f"\n❌ Could not load model from {model_file}")
                continue
            
            results = self.test_model_tps(model_class, model_file)
            
            if results:
                all_results[model_file] = results
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results

    def print_summary(self, all_results):
        """Print a summary comparison of all models"""
        print("\n" + "="*80)
        print("SUMMARY - TPS COMPARISON")
        print("="*80)
        
        if not all_results:
            print("No successful tests to summarize.")
            return
        
        # For each sequence length, compare all models
        seq_lengths = set()
        for results in all_results.values():
            seq_lengths.update(results.keys())
        
        for seq_len in sorted(seq_lengths):
            print(f"\nSequence Length: {seq_len}")
            print("-" * 80)
            print(f"{'Model':<40} {'TPS':>15} {'Time (ms)':>15}")
            print("-" * 80)
            
            model_tps = []
            for model_name, results in all_results.items():
                if seq_len in results:
                    tps = results[seq_len]['tps']
                    time_ms = results[seq_len]['avg_time_ms']
                    model_tps.append((model_name, tps, time_ms))
            
            # Sort by TPS (descending)
            model_tps.sort(key=lambda x: x[1], reverse=True)
            
            for model_name, tps, time_ms in model_tps:
                print(f"{model_name:<40} {tps:>14.2f} {time_ms:>14.2f}")
        
        print("\n" + "="*80)


def main():
    tester = TPS_Tester(warmup_iterations=3, test_iterations=10)
    results = tester.run_all_tests()
    
    # Save results to file
    import json
    output_file = "/home/user/webapp/tps_results.json"
    
    # Convert results to serializable format
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
