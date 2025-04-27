#!/usr/bin/env python3
"""
Script to run the scheduling phase of the Aegis Scheduler
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

from src.scheduler import schedule_plant_operation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the Aegis Scheduler")
    parser.add_argument(
        "--config", 
        default="configs/scheduler_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--optimize-vessels", 
        action="store_true",
        default=True,
        help="Run vessel optimization"
    )
    parser.add_argument(
        "--multiple-vessel-solutions", 
        action="store_true",
        default=False,
        help="Generate multiple vessel solutions"
    )
    parser.add_argument(
        "--no-optimize-vessels", 
        action="store_false",
        dest="optimize_vessels",
        help="Skip vessel optimization"
    )
    parser.add_argument(
        "--optimize-end-period", 
        action="store_true",
        default=False,  # Changed default to False
        help="Run end-period optimization (note: use run_optimization.py instead for separate optimization)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run scheduler
    logger.info(f"Running scheduler with config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Vessel optimization: {args.optimize_vessels}")
    logger.info(f"End-period optimization: {args.optimize_end_period}")
    
    result = schedule_plant_operation(
        config_path=args.config,
        output_dir=args.output_dir,
        optimize_vessels=args.optimize_vessels,
        multiple_vessel_solutions=args.multiple_vessel_solutions,
        optimize_end_period=args.optimize_end_period
    )
    
    # Output results
    if result["status"] == "success":
        print("\n========================= SCHEDULING RESULTS =========================")
        print(f"Total days scheduled:      {result['days_scheduled']}")
        print(f"Total volume processed:    {result['total_processed']:.2f} kb")
        if args.optimize_end_period:
            print(f"Optimizations applied:     {result['optimizations']}")
        else:
            print(f"Optimizations applied:     none (use run_optimization.py if needed)")
        
        print("\nResults saved to:")
        for file_name, file_path in result.get("files", {}).items():
            print(f"  - {file_name}: {file_path}")
            
        if result.get("vessel_optimization"):
            vessel_opt = result["vessel_optimization"]
            print("\nVessel Optimization:")
            print(f"  - Total voyages:        {vessel_opt.get('total_voyages', 'N/A')}")
            print(f"  - Total parcels:        {vessel_opt.get('total_parcels', 'N/A')}")
            
        print("\nTo run optimization on this schedule, use:")
        print(f"python run_optimization.py --schedule {args.output_dir}/daily_schedule.json")
        print("====================================================================")
    else:
        print("\n⚠️  SCHEDULING FAILED ⚠️")
        print(f"Error: {result.get('message', 'Unknown error')}")
        print("Check scheduler.log for details")
    
    return result["status"] == "success"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)