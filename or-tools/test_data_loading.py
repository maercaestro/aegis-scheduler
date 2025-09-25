#!/usr/bin/env python3
"""
Test script to verify data loading for the OR-Tools refinery optimizer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ortools_refinery_optimizer import DataLoader

def test_data_loading():
    """Test the data loading functionality."""
    print("Testing data loading...")
    
    # Initialize loader
    loader = DataLoader("./test_data")
    
    try:
        # Load data
        data = loader.load_scenario_data("config.json")
        
        print("✓ Data loaded successfully!")
        print(f"  - Number of crudes: {len(data['crudes'])}")
        print(f"  - Crudes: {data['crudes']}")
        print(f"  - Number of locations: {len(data['locations'])}")
        print(f"  - Locations: {data['locations']}")
        print(f"  - Source locations: {data['source_locations']}")
        print(f"  - Days range: {data['config']['DAYS']['start']} to {data['config']['DAYS']['end']}")
        print(f"  - Number of blends: {len(data['products_info'])}")
        print(f"  - Blends: {data['products_info']['product'].tolist()}")
        print(f"  - Crude availability windows: {list(data['crude_availability'].keys())}")
        print(f"  - Products capacity: {data['products_capacity']}")
        print(f"  - Opening inventory: {data['opening_inventory_dict']}")
        
        # Test with small config
        print("\nTesting with small config...")
        data_small = loader.load_scenario_data("config_small.json")
        print(f"✓ Small config loaded! Days: {data_small['config']['DAYS']['start']} to {data_small['config']['DAYS']['end']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    exit(0 if success else 1)
