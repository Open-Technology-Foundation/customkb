#!/usr/bin/env python
"""
Benchmark GPU vs CPU performance for reranking in CustomKB.
"""

import time
import sys
import os
import argparse
import subprocess
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_query(config_file, query, device='cuda'):
    """Run a query and measure time."""
    # Temporarily modify config for device
    if device == 'cpu':
        subprocess.run(['sed', '-i.bak', 's/reranking_device = cuda/reranking_device = cpu/', config_file], 
                      capture_output=True)
    
    start_time = time.time()
    result = subprocess.run(
        ['customkb', 'query', config_file, query],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start_time
    
    # Restore config if changed
    if device == 'cpu':
        subprocess.run(['mv', f'{config_file}.bak', config_file], capture_output=True)
    
    # Extract reranking time from logs
    rerank_time = None
    for line in result.stderr.split('\n'):
        if 'Reranking complete' in line:
            # Extract time from log if available
            import re
            match = re.search(r'Reranking (\d+) documents', line)
            if match:
                print(f"  Reranked {match.group(1)} documents")
    
    return elapsed, result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Benchmark GPU vs CPU reranking')
    parser.add_argument('config_file', help='Knowledge base configuration')
    parser.add_argument('--query', default='What is dharma in the secular context?', 
                       help='Query to benchmark')
    parser.add_argument('--iterations', type=int, default=3, 
                       help='Number of iterations per device')
    
    args = parser.parse_args()
    
    print("CustomKB GPU Benchmark")
    print("=" * 50)
    print(f"Config: {args.config_file}")
    print(f"Query: {args.query}")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Warm-up run
    print("Warming up...")
    run_query(args.config_file, args.query)
    
    # Benchmark each device
    devices = ['cuda', 'cpu']
    results = {}
    
    for device in devices:
        print(f"\nBenchmarking {device.upper()}...")
        times = []
        
        for i in range(args.iterations):
            print(f"  Run {i+1}/{args.iterations}...", end='', flush=True)
            elapsed, success = run_query(args.config_file, args.query, device)
            
            if success:
                times.append(elapsed)
                print(f" {elapsed:.2f}s")
            else:
                print(" FAILED")
        
        if times:
            results[device] = {
                'mean': statistics.mean(times),
                'min': min(times),
                'max': max(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0
            }
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    for device, stats in results.items():
        print(f"\n{device.upper()}:")
        print(f"  Mean:  {stats['mean']:.3f}s")
        print(f"  Min:   {stats['min']:.3f}s")
        print(f"  Max:   {stats['max']:.3f}s")
        print(f"  Stdev: {stats['stdev']:.3f}s")
    
    # Calculate speedup
    if 'cuda' in results and 'cpu' in results:
        speedup = results['cpu']['mean'] / results['cuda']['mean']
        print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")

if __name__ == '__main__':
    main()

#fin