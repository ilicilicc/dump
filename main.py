import os
import glob
import importlib
import time
import torch
import re

# Define the maximum parameters to avoid OOM errors
MAX_D_MODEL = 128
MAX_N_LAYERS = 4

def find_model_class(module):
    """
    Finds the main model class in a given module.
    The model class is assumed to start with 'HSTv'.
    """
    for attr_name in dir(module):
        if attr_name.startswith('HSTv'):
            return getattr(module, attr_name)
    return None

def parse_params_from_file(filepath):
    """
    Parses the model parameters from the self-test block of a file.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    params = {}

    # Use regex to find the parameters in the self-test block
    vocab_size_match = re.search(r'vocab_size\s*=\s*(\d+)', content)
    d_model_match = re.search(r'd_model\s*=\s*(\d+)', content)
    n_heads_match = re.search(r'n_heads\s*=\s*(\d+)', content)
    n_layers_match = re.search(r'n_layers\s*=\s*(\d+)', content)

    if vocab_size_match:
        params['vocab_size'] = int(vocab_size_match.group(1))
    if d_model_match:
        params['d_model'] = int(d_model_match.group(1))
    if n_heads_match:
        params['n_heads'] = int(n_heads_match.group(1))
    if n_layers_match:
        params['n_layers'] = int(n_layers_match.group(1))

    return params

def benchmark_model(model_class, model_file):
    """
    Instantiates and benchmarks a given model class.
    """
    print(f"--- Benchmarking {model_class.__name__} ---")

    try:
        # Parse parameters from the model's file
        params = parse_params_from_file(model_file)

        if not all(k in params for k in ['vocab_size', 'd_model', 'n_heads', 'n_layers']):
            print(f"❌ Could not parse all required parameters from {model_file}")
            return 0

        # Scale down parameters if they exceed the maximum allowed values
        if params['d_model'] > MAX_D_MODEL:
            print(f"   Scaling d_model from {params['d_model']} to {MAX_D_MODEL}")
            params['d_model'] = MAX_D_MODEL
        if params['n_layers'] > MAX_N_LAYERS:
            print(f"   Scaling n_layers from {params['n_layers']} to {MAX_N_LAYERS}")
            params['n_layers'] = MAX_N_LAYERS

        # Instantiate the model with the (potentially scaled) parameters
        if 'mode' in model_class.__init__.__code__.co_varnames:
             model = model_class(
                **params,
                mode='token'
            )
        else:
            model = model_class(**params)

        # Create a dummy input tensor
        batch_size = 1
        seq_len = 512
        x = torch.randint(0, params['vocab_size'], (batch_size, seq_len))

        # Warm-up run
        _ = model(x)

        # Timed run
        start_time = time.time()
        _ = model(x)
        end_time = time.time()

        duration = end_time - start_time
        tokens_processed = batch_size * seq_len
        tokens_per_second = tokens_processed / duration

        print(f"✅ Success!")
        print(f"   Tokens per second: {tokens_per_second:.2f}")
        return tokens_per_second

    except Exception as e:
        print(f"❌ Failed to benchmark {model_class.__name__}: {e}")
        return 0

def main():
    """
    Main function to discover and benchmark all HST models.
    """
    print("=" * 70)
    print("Discovering and benchmarking all HST models...")
    print("=" * 70)

    model_files = glob.glob('hst_v*.py')
    results = {}

    for model_file in sorted(model_files):
        module_name = model_file.replace('.py', '')
        try:
            module = importlib.import_module(module_name)
            model_class = find_model_class(module)

            if model_class:
                tps = benchmark_model(model_class, model_file)
                results[model_class.__name__] = tps
            else:
                print(f"Could not find a model class in {model_file}")

        except Exception as e:
            print(f"Could not import or benchmark {module_name}: {e}")

    print("\n" + "=" * 70)
    print("Benchmarking Complete!")
    print("=" * 70)

    if results:
        # Sort results by tokens per second (descending)
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

        print("Results (fastest to slowest):")
        for model_name, tps in sorted_results:
            print(f"  - {model_name}: {tps:.2f} tokens/sec")

        print(f"\n🏆 Fastest model: {sorted_results[0][0]}")
    else:
        print("No models were successfully benchmarked.")

if __name__ == '__main__':
    main()
