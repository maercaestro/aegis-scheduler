#!/usr/bin/env python3
"""
Script to run the optimization phase separately after scheduling
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

from src.scheduler import optimize_existing_schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("optimizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run optimization on an existing schedule")
    parser.add_argument(
        "--config", 
        default="configs/scheduler_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--schedule", 
        default="results/daily_schedule.json",
        help="Path to input schedule JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        default=f"results/optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory to save optimized results"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run optimization
    logger.info(f"Running optimization with config: {args.config}")
    logger.info(f"Input schedule: {args.schedule}")
    logger.info(f"Output directory: {args.output_dir}")
    
    result = optimize_existing_schedule(
        config_path=args.config,
        input_schedule_path=args.schedule,
        output_dir=args.output_dir
    )
    
    # Output results
    if result["status"] == "success":
        print("\n========================= OPTIMIZATION RESULTS =========================")
        print(f"Original total processed:   {result['original_processed']:.2f} kb")
        print(f"Optimized total processed:  {result['optimized_processed']:.2f} kb")
        print(f"Improvement:                {result['improvement']:.2f} kb ({result['improvement_percentage']:.2f}%)")
        print(f"Zero processing days:       {result['original_zero_days']} → {result['optimized_zero_days']}")
        print("\nOptimized results saved to:")
        for file_name, file_path in result.get("files", {}).items():
            print(f"  - {file_name}: {file_path}")
        print("======================================================================")
    else:
        print("\n⚠️  OPTIMIZATION FAILED ⚠️")
        print(f"Error: {result.get('message', 'Unknown error')}")
        print("Check optimizer.log for details")
    
    return result["status"] == "success"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)