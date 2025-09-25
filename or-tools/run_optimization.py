#!/usr/bin/env python3
"""
Simple startup script for running the OR-Tools refinery optimizer.
Designed for easy deployment to EC2.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ortools>=9.5.2237", "pandas>=1.5.0", "numpy>=1.21.0"
        ])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def run_optimization():
    """Run the optimization with default parameters."""
    print("Starting optimization...")
    
    # Set up paths
    script_dir = Path(__file__).parent
    optimizer_script = script_dir / "ortools_refinery_optimizer.py"
    
    if not optimizer_script.exists():
        print(f"✗ Optimizer script not found: {optimizer_script}")
        return False
    
    # Default parameters for production run
    cmd = [
        sys.executable, str(optimizer_script),
        "--config", "config.json",
        "--vessels", "6", 
        "--optimization", "throughput",
        "--demurrage-limit", "10",
        "--max-transitions", "11",
        "--time-limit", "3600",
        "--data-path", "./test_data",
        "--output-dir", "./results"
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Optimization completed successfully!")
            return True
        else:
            print(f"✗ Optimization failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running optimization: {e}")
        return False

def main():
    """Main startup function."""
    print("=== OR-Tools Refinery Optimizer Startup ===")
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Run optimization
    if not run_optimization():
        return 1
    
    print("=== All tasks completed successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
