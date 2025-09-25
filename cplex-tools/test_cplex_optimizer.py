#!/usr/bin/env python3
"""
Test script to verify CPLEX refinery optimizer functionality.
This script performs basic validation of data loading and model creation.
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path so we can import our optimizer
sys.path.append(os.getcwd())

from cplex_refinery_optimizer import DataLoader, CplexRefineryOptimizer, setup_logging


def test_data_loading():
    """Test that data loading works correctly."""
    print("Testing data loading...")
    
    try:
        loader = DataLoader("./test_data")
        data = loader.load_scenario_data("config.json")
        
        print(f"✓ Configuration loaded successfully")
        print(f"✓ Found {len(data['crudes'])} crude types: {data['crudes']}")
        print(f"✓ Found {len(data['locations'])} locations: {data['locations']}")
        print(f"✓ Found {len(data['products_info']['product'].tolist())} blends: {data['products_info']['product'].tolist()}")
        print(f"✓ Found {len(data['parcels'])} parcels")
        print(f"✓ Days range: {min(data['days'])} to {max(data['days'])}")
        
        return data
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None


def test_model_creation(data):
    """Test that the CPLEX model can be created."""
    print("\nTesting CPLEX model creation...")
    
    try:
        # Create logger
        output_dir = Path("./results")
        output_dir.mkdir(exist_ok=True)
        logger = setup_logging(output_dir, "test", 2)
        
        # Create optimizer with small vessel count for testing
        optimizer = CplexRefineryOptimizer(data, vessel_count=2, logger=logger)
        
        print("✓ CPLEX optimizer initialized successfully")
        print(f"✓ Model created with CPLEX version: {optimizer.model.get_version()}")
        
        # Test variable creation
        optimizer.create_variables()
        num_vars = optimizer.model.variables.get_num()
        print(f"✓ Created {num_vars} variables")
        
        return optimizer
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_simple_solve(optimizer):
    """Test solving a very simple model."""
    print("\nTesting simple solve...")
    
    try:
        # Add just a few basic constraints for testing
        optimizer.add_vessel_travel_constraints()
        optimizer.add_vessel_loading_constraints()
        
        num_constraints = optimizer.model.linear_constraints.get_num()
        print(f"✓ Added {num_constraints} constraints")
        
        # Set a simple objective (minimize variables)
        num_vars = optimizer.model.variables.get_num()
        obj_coeffs = [1.0] * num_vars  # Simple objective
        optimizer.model.objective.set_linear(enumerate(obj_coeffs))
        optimizer.model.objective.set_sense(optimizer.model.objective.sense.minimize)
        
        print("✓ Objective set successfully")
        
        # Try to solve with short time limit
        optimizer.model.parameters.timelimit.set(30)  # 30 seconds
        
        print("Attempting solve (30 second limit)...")
        result = optimizer.solve(time_limit_seconds=30)
        
        print(f"✓ Solve completed with status: {result['status']}")
        print(f"✓ Solve time: {result['solve_time']:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"✗ Simple solve failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("CPLEX Refinery Optimizer Test Suite")
    print("=" * 50)
    
    # Test 1: Data Loading
    data = test_data_loading()
    if not data:
        print("\n❌ Data loading test failed - stopping here")
        return 1
    
    # Test 2: Model Creation
    optimizer = test_model_creation(data)
    if not optimizer:
        print("\n❌ Model creation test failed - stopping here")
        return 1
    
    # Test 3: Simple Solve
    result = test_simple_solve(optimizer)
    if not result:
        print("\n❌ Simple solve test failed")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("CPLEX optimizer is ready for full optimization runs.")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())
