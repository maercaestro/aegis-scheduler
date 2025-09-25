#!/usr/bin/env python3
"""
CPLEX vs OR-Tools Performance Benchmark

This script runs the same optimization problem using both CPLEX and OR-Tools
to compare performance and solution quality.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from cplex_refinery_optimizer import main as cplex_main


def run_cplex_benchmark():
    """Run CPLEX optimization with throughput objective."""
    print("🔵 Running CPLEX Optimization...")
    print("=" * 60)
    
    # Set command line arguments for CPLEX
    sys.argv = [
        'cplex_refinery_optimizer.py',
        '--config', 'config.json',
        '--vessels', '6',
        '--optimization', 'throughput',
        '--demurrage-limit', '10',
        '--max-transitions', '11',
        '--time-limit', '1800',  # 30 minutes
        '--data-path', './test_data',
        '--output-dir', './results'
    ]
    
    start_time = time.time()
    
    try:
        result = cplex_main()
        cplex_time = time.time() - start_time
        
        if result == 0:
            print(f"✅ CPLEX completed successfully in {cplex_time:.2f} seconds")
            return {
                'status': 'success',
                'solve_time': cplex_time,
                'solver': 'CPLEX'
            }
        else:
            print(f"❌ CPLEX failed with exit code {result}")
            return {
                'status': 'failed',
                'solve_time': cplex_time,
                'solver': 'CPLEX',
                'exit_code': result
            }
            
    except Exception as e:
        cplex_time = time.time() - start_time
        print(f"❌ CPLEX error: {e}")
        return {
            'status': 'error',
            'solve_time': cplex_time,
            'solver': 'CPLEX',
            'error': str(e)
        }


def run_ortools_benchmark():
    """Run OR-Tools optimization for comparison."""
    print("🔴 Running OR-Tools Optimization...")
    print("=" * 60)
    
    # Change to parent directory to run OR-Tools
    original_dir = os.getcwd()
    
    try:
        os.chdir('..')
        
        # Run OR-Tools optimizer
        command = """python ortools_refinery_optimizer.py \\
            --config config.json \\
            --vessels 6 \\
            --optimization throughput \\
            --demurrage-limit 10 \\
            --max-transitions 11 \\
            --time-limit 1800 \\
            --data-path ./test_data \\
            --output-dir ./results"""
        
        start_time = time.time()
        result = os.system(command)
        ortools_time = time.time() - start_time
        
        if result == 0:
            print(f"✅ OR-Tools completed successfully in {ortools_time:.2f} seconds")
            return {
                'status': 'success',
                'solve_time': ortools_time,
                'solver': 'OR-Tools'
            }
        else:
            print(f"❌ OR-Tools failed with exit code {result}")
            return {
                'status': 'failed',
                'solve_time': ortools_time,
                'solver': 'OR-Tools',
                'exit_code': result
            }
            
    except Exception as e:
        ortools_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ OR-Tools error: {e}")
        return {
            'status': 'error',
            'solve_time': ortools_time,
            'solver': 'OR-Tools',
            'error': str(e)
        }
    finally:
        os.chdir(original_dir)


def analyze_results():
    """Analyze and compare results from both solvers."""
    print("📊 Analyzing Results...")
    print("=" * 60)
    
    results_dir = Path('./results')
    
    # Find latest CPLEX results
    cplex_files = list(results_dir.glob('cplex_summary_throughput_6_vessels_10_demurrage_*.json'))
    if cplex_files:
        cplex_file = max(cplex_files, key=lambda x: x.stat().st_mtime)
        with open(cplex_file, 'r') as f:
            cplex_results = json.load(f)
    else:
        cplex_results = None
    
    # Find latest OR-Tools results  
    ortools_files = list(results_dir.glob('ortools_summary_throughput_*_vessels_*_demurrage_*.json'))
    if ortools_files:
        ortools_file = max(ortools_files, key=lambda x: x.stat().st_mtime)
        with open(ortools_file, 'r') as f:
            ortools_results = json.load(f)
    else:
        ortools_results = None
    
    print("Performance Comparison:")
    print("-" * 40)
    
    if cplex_results:
        print(f"🔵 CPLEX:")
        print(f"   Status: {cplex_results.get('status', 'unknown')}")
        print(f"   Solve time: {cplex_results.get('solve_time_seconds', 0):.2f}s")
        print(f"   Objective: {cplex_results.get('objective_value', 'N/A')}")
        print(f"   Production: {cplex_results.get('total_production', 0):.1f}")
    else:
        print("🔵 CPLEX: No results found")
    
    if ortools_results:
        print(f"🔴 OR-Tools:")
        print(f"   Status: {ortools_results.get('status', 'unknown')}")
        print(f"   Solve time: {ortools_results.get('solve_time_seconds', 0):.2f}s")
        print(f"   Objective: {ortools_results.get('objective_value', 'N/A')}")
        print(f"   Production: {ortools_results.get('total_production', 0):.1f}")
    else:
        print("🔴 OR-Tools: No results found")
    
    # Performance comparison
    if cplex_results and ortools_results:
        cplex_time = cplex_results.get('solve_time_seconds', float('inf'))
        ortools_time = ortools_results.get('solve_time_seconds', float('inf'))
        
        print(f"\\n📈 Speed Comparison:")
        if cplex_time < ortools_time:
            speedup = ortools_time / cplex_time
            print(f"   CPLEX is {speedup:.2f}x faster than OR-Tools")
        elif ortools_time < cplex_time:
            speedup = cplex_time / ortools_time
            print(f"   OR-Tools is {speedup:.2f}x faster than CPLEX")
        else:
            print("   Both solvers have similar performance")
        
        # Solution quality comparison
        cplex_obj = cplex_results.get('objective_value', 0)
        ortools_obj = ortools_results.get('objective_value', 0)
        
        print(f"\\n🎯 Solution Quality:")
        if cplex_obj > ortools_obj:
            improvement = ((cplex_obj - ortools_obj) / ortools_obj) * 100
            print(f"   CPLEX found {improvement:.2f}% better solution")
        elif ortools_obj > cplex_obj:
            improvement = ((ortools_obj - cplex_obj) / cplex_obj) * 100
            print(f"   OR-Tools found {improvement:.2f}% better solution")
        else:
            print("   Both solvers found similar solutions")


def main():
    """Run the complete benchmark."""
    print("🚀 CPLEX vs OR-Tools Performance Benchmark")
    print("=" * 80)
    print("This benchmark will run the same optimization problem using both")
    print("CPLEX and OR-Tools to compare performance and solution quality.")
    print("=" * 80)
    
    # Create results directory
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    benchmark_results = {
        'benchmark_start': time.time(),
        'cplex': None,
        'ortools': None
    }
    
    # Run CPLEX benchmark
    cplex_result = run_cplex_benchmark()
    benchmark_results['cplex'] = cplex_result
    
    # Small delay between runs
    time.sleep(2)
    
    # Run OR-Tools benchmark  
    ortools_result = run_ortools_benchmark()
    benchmark_results['ortools'] = ortools_result
    
    benchmark_results['benchmark_end'] = time.time()
    benchmark_results['total_time'] = benchmark_results['benchmark_end'] - benchmark_results['benchmark_start']
    
    # Analyze results
    analyze_results()
    
    # Save benchmark results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    benchmark_file = results_dir / f"cplex_vs_ortools_benchmark_{timestamp}.json"
    
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\\n💾 Benchmark results saved to: {benchmark_file}")
    
    print("\\n" + "=" * 80)
    print("🏁 Benchmark Complete!")
    
    # Summary
    if cplex_result['status'] == 'success' and ortools_result['status'] == 'success':
        cplex_time = cplex_result['solve_time']
        ortools_time = ortools_result['solve_time']
        
        if cplex_time < ortools_time:
            winner = "CPLEX"
            ratio = ortools_time / cplex_time
        else:
            winner = "OR-Tools"
            ratio = cplex_time / ortools_time
            
        print(f"🏆 Winner: {winner} ({ratio:.2f}x faster)")
    else:
        print("⚠️  Unable to determine winner due to solver failures")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
