#!/usr/bin/env python3
"""
Simple runner script that shows optimization progress in real-time.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_with_live_output():
    """Run the optimizer and show output in real-time."""
    
    print("=" * 60)
    print("OR-TOOLS REFINERY OPTIMIZER - LIVE RUN")
    print("=" * 60)
    
    # Default command
    cmd = [
        sys.executable, "ortools_refinery_optimizer.py",
        "--config", "config.json",
        "--vessels", "6", 
        "--optimization", "throughput",
        "--demurrage-limit", "10",
        "--max-transitions", "11",
        "--time-limit", "3600",
        "--data-path", "./test_data",
        "--output-dir", "./results"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output as it comes
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        
        # Wait for completion
        return_code = process.wait()
        
        print("=" * 60)
        print(f"Process completed with return code: {return_code}")
        
        # Show log files created
        import glob
        log_files = glob.glob("./results/*optimization_log*.log")
        if log_files:
            print(f"Log files created:")
            for log_file in sorted(log_files):
                print(f"  - {log_file}")
        
        return return_code
        
    except KeyboardInterrupt:
        print("\n=== INTERRUPTED BY USER ===")
        process.terminate()
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(run_with_live_output())
