#!/usr/bin/env python3
"""
Test script to verify logging functionality works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_logging():
    """Test the logging functionality with a quick optimization run."""
    print("Testing logging functionality...")
    
    try:
        # Import and run with small dataset
        import subprocess
        
        cmd = [
            sys.executable, "ortools_refinery_optimizer.py",
            "--config", "config_small.json",
            "--vessels", "2", 
            "--optimization", "throughput",
            "--demurrage-limit", "5",
            "--max-transitions", "5",
            "--time-limit", "60",  # Short time limit for testing
            "--data-path", "./test_data",
            "--output-dir", "./results"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        print("=== STDOUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== STDERR ===") 
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        # Check if log file was created
        import glob
        log_files = glob.glob("./results/*optimization_log*.log")
        if log_files:
            print(f"✓ Log file created: {log_files[0]}")
            return True
        else:
            print("✗ No log file found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_logging()
    exit(0 if success else 1)
