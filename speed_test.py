import os
import subprocess
import time
import re

def get_test_params(filepath):
    """Parses the script to find the total number of tokens used in the self-test."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Regex to find the input tensor creation line
        # It looks for torch.randint(..., (batch_size, num_tokens))
        match = re.search(r'torch\.randint\s*\(.*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)', content)
        if match:
            batch_size = int(match.group(1))
            num_tokens = int(match.group(2))
            return batch_size * num_tokens
        else:
            # Fallback for different patterns if needed
            return None
    except Exception as e:
        print(f"Could not parse {filepath}: {e}")
        return None

def run_speed_test():
    # Improved file discovery
    files_to_test = [f for f in os.listdir('.') if f.startswith('hst') and f.endswith('.py') and f != 'speed_test.py']
    results = {}

    for file in sorted(files_to_test):
        print(f"--- Testing {file} ---")
        start_time = time.time()
        try:
            # Run the script
            subprocess.run(['python3', file], check=True, capture_output=True, text=True)
            end_time = time.time()
            execution_time = end_time - start_time

            # Get token count
            total_tokens = get_test_params(file)

            if total_tokens and execution_time > 0:
                tokens_per_second = total_tokens / execution_time
                results[file] = tokens_per_second
                print(f"✅ {file} | Tokens: {total_tokens} | Time: {execution_time:.2f}s | Tokens/sec: {tokens_per_second:.2f}")
            else:
                results[file] = 0 # Mark as 0 if parsing fails
                print(f"✅ {file} executed in {execution_time:.2f}s, but could not calculate token speed.")

        except subprocess.CalledProcessError as e:
            results[file] = 'Error'
            print(f"❌ {file} failed with error:\n{e.stderr}")
        print("\n" + "=" * 70)

    # Sort results by tokens/sec (descending)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if isinstance(v, float)],
        key=lambda item: item[1],
        reverse=True
    )

    error_results = {k: v for k, v in results.items() if not isinstance(v, float)}

    print("\n--- Token Speed Test Results (Tokens/Second) ---")
    for file, tps in sorted_results:
        print(f"{file}: {tps:.2f} tokens/sec")

    if error_results:
        print("\n--- Errors ---")
        for file, err in error_results.items():
            print(f"{file}: {err}")


if __name__ == '__main__':
    run_speed_test()
