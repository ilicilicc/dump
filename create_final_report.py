#!/usr/bin/env python3
"""
Create final comprehensive TPS benchmark report
"""

import json
from pathlib import Path

def load_results():
    """Load all benchmark results"""
    results_main = {}
    results_remaining = {}
    
    if Path("tps_benchmark_results.json").exists():
        with open("tps_benchmark_results.json", 'r') as f:
            results_main = json.load(f)
    
    if Path("tps_remaining_results.json").exists():
        with open("tps_remaining_results.json", 'r') as f:
            results_remaining = json.load(f)
    
    # Combine results
    all_results = {**results_main, **results_remaining}
    return all_results

def print_report(all_results):
    """Print comprehensive TPS benchmark report"""
    
    print("="*90)
    print(" " * 25 + "HST MODEL TPS BENCHMARK REPORT")
    print("="*90)
    print()
    
    # Overall summary table
    print("OVERALL SUMMARY")
    print("-"*90)
    print(f"{'Model':<30} {'Class':<20} {'Params(M)':>10} {'Avg TPS':>12} {'Best TPS':>12}")
    print("-"*90)
    
    model_stats = []
    for model_name, result in all_results.items():
        class_name = result['class_name']
        params = result['params_M']
        
        # Calculate average TPS across all sequence lengths
        tps_values = [r['tps'] for r in result['results'].values()]
        avg_tps = sum(tps_values) / len(tps_values)
        best_tps = max(tps_values)
        
        model_stats.append({
            'model': model_name.replace('.py', ''),
            'class': class_name,
            'params': params,
            'avg_tps': avg_tps,
            'best_tps': best_tps,
        })
    
    # Sort by average TPS
    model_stats.sort(key=lambda x: x['avg_tps'], reverse=True)
    
    for stat in model_stats:
        print(f"{stat['model']:<30} {stat['class']:<20} {stat['params']:>10.2f} "
              f"{stat['avg_tps']:>12.2f} {stat['best_tps']:>12.2f}")
    
    print()
    print()
    
    # Detailed results by sequence length
    for seq_len in [128, 256, 512]:
        print(f"DETAILED RESULTS - SEQUENCE LENGTH {seq_len}")
        print("-"*90)
        print(f"{'Model':<30} {'TPS':>12} {'Time(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12}")
        print("-"*90)
        
        data = []
        for model_name, result in all_results.items():
            seq_key = str(seq_len)
            if seq_key in result['results']:
                r = result['results'][seq_key]
                data.append({
                    'model': model_name.replace('.py', ''),
                    'tps': r['tps'],
                    'time': r['avg_time_ms'],
                    'min': r['min_time_ms'],
                    'max': r['max_time_ms'],
                })
        
        # Sort by TPS
        data.sort(key=lambda x: x['tps'], reverse=True)
        
        for d in data:
            print(f"{d['model']:<30} {d['tps']:>12.2f} {d['time']:>12.2f} "
                  f"{d['min']:>12.2f} {d['max']:>12.2f}")
        
        print()
    
    print()
    
    # Performance insights
    print("PERFORMANCE INSIGHTS")
    print("-"*90)
    
    # Best performer
    best = max(model_stats, key=lambda x: x['avg_tps'])
    print(f"🏆 Best Average TPS: {best['model']} ({best['avg_tps']:.2f} tokens/sec)")
    
    # Most efficient (TPS per parameter)
    for stat in model_stats:
        stat['efficiency'] = stat['avg_tps'] / stat['params']
    best_eff = max(model_stats, key=lambda x: x['efficiency'])
    print(f"⚡ Most Efficient: {best_eff['model']} ({best_eff['efficiency']:.2f} TPS/M params)")
    
    # Largest model
    largest = max(model_stats, key=lambda x: x['params'])
    print(f"📊 Largest Model: {largest['model']} ({largest['params']:.2f}M parameters)")
    
    # Fastest inference
    fastest_time = None
    for model_name, result in all_results.items():
        for seq_len, r in result['results'].items():
            if fastest_time is None or r['min_time_ms'] < fastest_time['time']:
                fastest_time = {
                    'model': model_name.replace('.py', ''),
                    'time': r['min_time_ms'],
                    'seq_len': seq_len
                }
    print(f"⚡ Fastest Single Inference: {fastest_time['model']} "
          f"({fastest_time['time']:.2f}ms for seq_len={fastest_time['seq_len']})")
    
    print()
    print("="*90)
    print()
    
    # Model-specific details
    print("MODEL DETAILS")
    print("-"*90)
    for model_name, result in sorted(all_results.items()):
        print(f"\n{model_name}:")
        print(f"  Class: {result['class_name']}")
        print(f"  Parameters: {result['params_M']:.2f}M")
        if 'n_layers' in result:
            print(f"  Layers: {result['n_layers']}")
        print(f"  Config: d_model={result['config']['d_model']}, "
              f"n_layers={result['config']['n_layers']}, "
              f"n_heads={result['config']['n_heads']}")
        print(f"  Performance:")
        for seq_len, r in sorted(result['results'].items(), key=lambda x: int(x[0])):
            print(f"    seq_len={seq_len}: {r['tps']:.2f} TPS, {r['avg_time_ms']:.2f}ms")
    
    print()
    print("="*90)

def save_markdown_report(all_results):
    """Save a markdown version of the report"""
    
    with open("TPS_BENCHMARK_REPORT.md", 'w') as f:
        f.write("# HST Model TPS Benchmark Report\n\n")
        f.write("## Overall Summary\n\n")
        f.write("| Model | Class | Params (M) | Avg TPS | Best TPS |\n")
        f.write("|-------|-------|------------|---------|----------|\n")
        
        model_stats = []
        for model_name, result in all_results.items():
            tps_values = [r['tps'] for r in result['results'].values()]
            avg_tps = sum(tps_values) / len(tps_values)
            best_tps = max(tps_values)
            
            model_stats.append({
                'model': model_name.replace('.py', ''),
                'class': result['class_name'],
                'params': result['params_M'],
                'avg_tps': avg_tps,
                'best_tps': best_tps,
            })
        
        model_stats.sort(key=lambda x: x['avg_tps'], reverse=True)
        
        for stat in model_stats:
            f.write(f"| {stat['model']} | {stat['class']} | {stat['params']:.2f} | "
                   f"{stat['avg_tps']:.2f} | {stat['best_tps']:.2f} |\n")
        
        f.write("\n## Detailed Results by Sequence Length\n\n")
        
        for seq_len in [128, 256, 512]:
            f.write(f"### Sequence Length: {seq_len}\n\n")
            f.write("| Model | TPS | Time (ms) | Min (ms) | Max (ms) |\n")
            f.write("|-------|-----|-----------|----------|----------|\n")
            
            data = []
            for model_name, result in all_results.items():
                seq_key = str(seq_len)
                if seq_key in result['results']:
                    r = result['results'][seq_key]
                    data.append({
                        'model': model_name.replace('.py', ''),
                        'tps': r['tps'],
                        'time': r['avg_time_ms'],
                        'min': r['min_time_ms'],
                        'max': r['max_time_ms'],
                    })
            
            data.sort(key=lambda x: x['tps'], reverse=True)
            
            for d in data:
                f.write(f"| {d['model']} | {d['tps']:.2f} | {d['time']:.2f} | "
                       f"{d['min']:.2f} | {d['max']:.2f} |\n")
            
            f.write("\n")
        
        # Add insights
        f.write("## Performance Insights\n\n")
        best = max(model_stats, key=lambda x: x['avg_tps'])
        f.write(f"- **Best Average TPS**: {best['model']} ({best['avg_tps']:.2f} tokens/sec)\n")
        
        for stat in model_stats:
            stat['efficiency'] = stat['avg_tps'] / stat['params']
        best_eff = max(model_stats, key=lambda x: x['efficiency'])
        f.write(f"- **Most Efficient**: {best_eff['model']} ({best_eff['efficiency']:.2f} TPS per M params)\n")
        
        largest = max(model_stats, key=lambda x: x['params'])
        f.write(f"- **Largest Model**: {largest['model']} ({largest['params']:.2f}M parameters)\n")
        
        f.write("\n## Model Details\n\n")
        for model_name, result in sorted(all_results.items()):
            f.write(f"### {model_name}\n\n")
            f.write(f"- **Class**: {result['class_name']}\n")
            f.write(f"- **Parameters**: {result['params_M']:.2f}M\n")
            if 'n_layers' in result:
                f.write(f"- **Layers**: {result['n_layers']}\n")
            f.write(f"- **Config**: d_model={result['config']['d_model']}, "
                   f"n_layers={result['config']['n_layers']}, "
                   f"n_heads={result['config']['n_heads']}\n")
            f.write("- **Performance**:\n")
            for seq_len, r in sorted(result['results'].items(), key=lambda x: int(x[0])):
                f.write(f"  - seq_len={seq_len}: {r['tps']:.2f} TPS, {r['avg_time_ms']:.2f}ms\n")
            f.write("\n")

def main():
    all_results = load_results()
    
    if not all_results:
        print("No results found!")
        return
    
    # Print console report
    print_report(all_results)
    
    # Save markdown report
    save_markdown_report(all_results)
    print("\n✓ Markdown report saved to TPS_BENCHMARK_REPORT.md\n")

if __name__ == "__main__":
    main()
