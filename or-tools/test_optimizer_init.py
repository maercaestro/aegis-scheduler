#!/usr/bin/env python3
"""
Test script to verify the OR-Tools optimizer can be initialized and built.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ortools_refinery_optimizer import DataLoader, OrToolsRefineryOptimizer

def test_optimizer_initialization():
    """Test the optimizer initialization."""
    print("Testing optimizer initialization...")
    
    try:
        # Load data
        loader = DataLoader("./test_data")
        data = loader.load_scenario_data("config_small.json")  # Use small config for faster testing
        
        print("✓ Data loaded successfully!")
        
        # Initialize optimizer
        optimizer = OrToolsRefineryOptimizer(data, vessel_count=3)  # Use fewer vessels for testing
        
        print("✓ Optimizer initialized successfully!")
        print(f"  - Vessels: {optimizer.vessels}")
        print(f"  - Days: {len(optimizer.days)} (from {min(optimizer.days)} to {max(optimizer.days)})")
        print(f"  - Parcels: {len(optimizer.parcels)}")
        print(f"  - Crudes: {len(optimizer.crudes)}")
        print(f"  - Locations: {len(optimizer.locations)}")
        print(f"  - Blends: {len(optimizer.blends)}")
        
        # Test variable creation
        optimizer.create_variables()
        print("✓ Variables created successfully!")
        
        total_vars = sum(len(var_dict) for var_dict in optimizer.variables.values())
        print(f"  - Total variables created: {total_vars}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimizer_initialization()
    exit(0 if success else 1)
