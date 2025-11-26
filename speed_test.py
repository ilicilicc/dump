import torch
import time
import importlib

def get_model_and_input(module_name, model_class_name, batch_size, seq_len, vocab_size, d_model, n_heads, n_layers, mode='token'):
    """Dynamically imports a model and creates a dummy input tensor."""
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_class_name)

        # Determine the correct class name to instantiate
        if model_class_name == "HSTv7_1Ultimate":
             model = model_class(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                mode=mode
            )
        elif model_class_name == "HSTv7_2FullLattice":
            model = model_class(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                mode=mode
            )
        else:
            # Add other model instantiations here if needed
            raise ValueError(f"Unknown model class name: {model_class_name}")

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        return model, dummy_input
    except (ImportError, AttributeError) as e:
        print(f"Could not load model {model_class_name} from {module_name}: {e}")
        return None, None


def benchmark_model(model_name, module_name, model_class_name, batch_size, seq_len, vocab_size, d_model, n_heads, n_layers, num_batches=10, mode='token'):
    """Benchmarks a given model and prints the tokens per second."""
    print(f"--- Benchmarking {model_name} ---")

    model, dummy_input = get_model_and_input(module_name, model_class_name, batch_size, seq_len, vocab_size, d_model, n_heads, n_layers, mode)
    if model is None:
        return

    # Warm-up
    for _ in range(3):
        _ = model(dummy_input)

    # Benchmark
    start_time = time.time()
    for _ in range(num_batches):
        _ = model(dummy_input)
    end_time = time.time()

    total_tokens = num_batches * batch_size * seq_len
    total_time = end_time - start_time
    tokens_per_second = total_tokens / total_time

    print(f"Tokens per second: {tokens_per_second:.2f}")
    print("-" * 30)


if __name__ == '__main__':
    # --- Benchmark Parameters ---
    BATCH_SIZE = 1
    SEQ_LEN = 256
    VOCAB_SIZE = 50257
    D_MODEL = 64
    N_HEADS = 2
    N_LAYERS = 2
    NUM_BATCHES = 10

    # --- Models to Benchmark ---
    models_to_benchmark = [
        {
            "model_name": "HST v7.1 Ultimate",
            "module_name": "hst_v7_1_ultimate",
            "model_class_name": "HSTv7_1Ultimate",
        },
        {
            "model_name": "HST v7.2 Full Lattice",
            "module_name": "hst_v7_2_full_lattice",
            "model_class_name": "HSTv7_2FullLattice",
        },
    ]

    for model_info in models_to_benchmark:
        benchmark_model(
            model_name=model_info["model_name"],
            module_name=model_info["module_name"],
            model_class_name=model_info["model_class_name"],
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            num_batches=NUM_BATCHES
        )
