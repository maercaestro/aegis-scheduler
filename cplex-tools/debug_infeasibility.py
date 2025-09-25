#!/usr/bin/env python3
"""
Debug script to identify infeasible constraints in CPLEX model
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from cplex_refinery_optimizer import DataLoader, CplexRefineryOptimizer, setup_logging


def debug_infeasibility():
    """Debug the infeasibility issue."""
    print("🔍 Debugging CPLEX Model Infeasibility...")
    print("=" * 60)
    
    # Load data
    loader = DataLoader("./test_data")
    data = loader.load_scenario_data("config.json")
    
    print(f"Data loaded:")
    print(f"  - Vessels: 6")
    print(f"  - Parcels: {len(data['parcels'])}")
    print(f"  - Days: {len(data['days'])} (from {min(data['days'])} to {max(data['days'])})")
    
    print(f"\\nParcel details:")
    for i, parcel in enumerate(data['parcels']):
        location, crude, window = parcel
        parcel_size = data['parcel_info'][parcel]['size']
        available_days = data['parcel_info'][parcel]['days']
        print(f"  {i+1}. {crude} at {location} ({window}): {parcel_size:,} barrels, days {available_days}")
    
    print(f"\\nVessel capacity: {data['config']['Vessel_max_limit']:,} barrels")
    print(f"Total crude available: {sum(data['parcel_info'][p]['size'] for p in data['parcels']):,} barrels")
    
    # Check if 6 vessels can handle all parcels
    max_vessel_capacity = data['config']['Vessel_max_limit']
    total_crude = sum(data['parcel_info'][p]['size'] for p in data['parcels'])
    min_vessels_needed = total_crude / max_vessel_capacity
    
    print(f"\\nCapacity analysis:")
    print(f"  - Total crude volume: {total_crude:,} barrels")
    print(f"  - Max vessel capacity: {max_vessel_capacity:,} barrels")
    print(f"  - Minimum vessels needed: {min_vessels_needed:.2f}")
    print(f"  - Vessels available: 6")
    
    if min_vessels_needed > 6:
        print(f"  ⚠️  WARNING: Not enough vessel capacity!")
    else:
        print(f"  ✅ Sufficient vessel capacity")
    
    # Check time windows
    print(f"\\nTime window analysis:")
    for parcel in data['parcels']:
        available_days = data['parcel_info'][parcel]['days']
        print(f"  - {parcel[1]} at {parcel[0]}: days {available_days}")
    
    # Check if all parcels can be picked up within their windows
    overlapping_windows = {}
    for parcel in data['parcels']:
        days = tuple(data['parcel_info'][parcel]['days'])
        if days in overlapping_windows:
            overlapping_windows[days].append(parcel)
        else:
            overlapping_windows[days] = [parcel]
    
    print(f"\\nOverlapping time windows:")
    for days, parcels in overlapping_windows.items():
        if len(parcels) > 1:
            print(f"  - Days {days}: {len(parcels)} parcels")
            for parcel in parcels:
                print(f"    * {parcel[1]} at {parcel[0]}")
    
    # Try creating a simpler model
    print(f"\\n🧪 Testing simplified model...")
    
    # Create logger
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    logger = setup_logging(output_dir, "debug", 3)
    
    # Test with fewer vessels first
    optimizer = CplexRefineryOptimizer(data, vessel_count=3, logger=logger)
    
    print(f"✅ Model created successfully with 3 vessels")
    print(f"   - Variables: {optimizer.model.variables.get_num()}")
    
    # Only add basic constraints
    optimizer.create_variables()
    optimizer.add_vessel_travel_constraints()
    
    print(f"   - Constraints after travel: {optimizer.model.linear_constraints.get_num()}")
    
    # Try to solve just travel constraints
    try:
        result = optimizer.solve(time_limit_seconds=60)
        print(f"   - Travel constraints: {result['status']}")
    except Exception as e:
        print(f"   - Travel constraints failed: {e}")
    
    return data


if __name__ == "__main__":
    debug_infeasibility()
