"""
Main optimization runner for the refinery and vessel scheduling system.
Provides a high-level interface to run the complete optimization pipeline.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict, Any
from tqdm import tqdm

from data_loader import DataLoader
from optimization_model import OptimizationModel
from solver_manager import SolverManager
from result_processor import ResultProcessor


class OptimizationRunner:
    """Main class to orchestrate the optimization process."""
    
    def __init__(self, data_path: str = "test_data", output_path: str = "results"):
        """
        Initialize the optimization runner.
        
        Args:
            data_path: Path to input data directory
            output_path: Path to output results directory
        """
        self.data_path = data_path
        self.output_path = output_path
        self.data_loader = DataLoader(data_path)
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
    def load_data(self) -> Tuple[Dict[str, Any], ...]:
        """Load all required data for optimization."""
        print("📁 Loading optimization data...")
        return self.data_loader.load_all_data()
        
    def create_start_date(self, config: dict) -> pd.Timestamp:
        """Create start date from config."""
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12, 
        }
        month_number = month_map[config["schedule_month"]]
        year = config["schedule_year"]
        return pd.to_datetime(f"{year}-{month_number:02d}-01")
        
    def run_optimization(self, vessel_count: int = 6, optimization_type: str = "throughput", 
                        max_demurrage_limit: int = 10, scenario_name: str = "test") -> Dict[str, Any]:
        """
        Run the complete optimization process.
        
        Args:
            vessel_count: Number of vessels to use
            optimization_type: Type of optimization ("margin" or "throughput")
            max_demurrage_limit: Maximum demurrage days for throughput optimization
            scenario_name: Name of the scenario for output files
            
        Returns:
            Dictionary containing optimization results and file paths
        """
        try:
            # Load data
            (config, crudes, locations, time_of_travel, crude_availability, 
             source_locations, products_info, crude_margins, opening_inventory_dict,
             products_ratio, window_to_days) = self.load_data()
            
            # Create start date
            start_date = self.create_start_date(config)
            
            # Create optimization model
            print("🔧 Creating optimization model...")
            opt_model = OptimizationModel(
                config=config,
                crudes=crudes,
                locations=locations,
                time_of_travel=time_of_travel,
                crude_availability=crude_availability,
                source_locations=source_locations,
                products_info=products_info,
                crude_margins=crude_margins,
                opening_inventory_dict=opening_inventory_dict,
                products_ratio=products_ratio,
                window_to_days=window_to_days,
                vessel_count=vessel_count
            )
            
            # Build model
            model = opt_model.build_model(optimization_type, max_demurrage_limit)
            
            # Solve model
            print("🚀 Solving optimization model...")
            solver_manager = SolverManager(config)
            results = solver_manager.solve_model(
                model=model,
                scenario_name=scenario_name,
                vessel_count=vessel_count,
                optimization_type=optimization_type,
                max_demurrage_limit=max_demurrage_limit
            )
            
            # Process results
            print("📋 Processing optimization results...")
            result_processor = ResultProcessor(model, config, crudes, start_date)
            
            # Extract results
            crude_blending_df = result_processor.extract_crude_blending_results()
            vessel_routing_df = result_processor.extract_vessel_routing_results(crude_availability)
            summary_metrics = result_processor.calculate_summary_metrics(crude_blending_df)
            
            # Generate output file names
            if optimization_type == 'throughput':
                crude_blending_filename = f'crude_blending_{optimization_type}_optimization_{vessel_count}_vessels_{config["DAYS"]["end"]}_days_{config["MaxTransitions"]}_transitions_{max_demurrage_limit}_demurrages.csv'
                vessel_routing_filename = f'vessel_routing_{optimization_type}_optimization_{vessel_count}_vessels_{config["DAYS"]["end"]}_days_{config["MaxTransitions"]}_transitions_{max_demurrage_limit}_demurrages.csv'
            else:
                crude_blending_filename = f'crude_blending_{optimization_type}_optimization_{vessel_count}_vessels_{config["DAYS"]["end"]}_days_{config["MaxTransitions"]}_transitions.csv'
                vessel_routing_filename = f'vessel_routing_{optimization_type}_optimization_{vessel_count}_vessels_{config["DAYS"]["end"]}_days_{config["MaxTransitions"]}_transitions.csv'
            
            # Save results
            crude_blending_path = os.path.join(self.output_path, crude_blending_filename)
            vessel_routing_path = os.path.join(self.output_path, vessel_routing_filename)
            
            crude_blending_df.to_csv(crude_blending_path, index=False)
            vessel_routing_df.to_csv(vessel_routing_path, index=False)
            
            print(f"Results saved:")
            print(f"  Crude blending: {crude_blending_path}")
            print(f"  Vessel routing: {vessel_routing_path}")
            
            # Print summary metrics
            print(f"\nOptimization Summary:")
            print(f"  Total throughput: {summary_metrics['total_throughput']:.1f} k barrels")
            print(f"  Total margin: ${summary_metrics['total_margin']:,.0f}")
            print(f"  Average throughput: {summary_metrics['average_throughput']:.1f} k barrels/day")
            print(f"  Average margin: ${summary_metrics['average_margin']:,.0f}/day")
            print(f"  Total demurrage at Melaka: ${summary_metrics['demurrage_at_melaka']:,.0f}")
            print(f"  Total demurrage at source: ${summary_metrics['demurrage_at_source']:,.0f}")
            print(f"  Total demurrage: ${summary_metrics['total_demurrage']:,.0f}")
            
            return {
                "status": "success",
                "crude_blending_df": crude_blending_df,
                "vessel_routing_df": vessel_routing_df,
                "summary_metrics": summary_metrics,
                "crude_blending_file": crude_blending_path,
                "vessel_routing_file": vessel_routing_path,
                "model": model,
                "solver_results": results
            }
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }


def main():
    """Main function to run optimization with default parameters."""
    
    # Configuration parameters - can be modified as needed
    VESSEL_COUNT = 6
    OPTIMIZATION_TYPE = "throughput"  # "margin" or "throughput"
    MAX_DEMURRAGE_LIMIT = 10
    SCENARIO_NAME = "test_scenario"
    
    print("=== Aegis Scheduler Optimization ===")
    print(f"Vessel count: {VESSEL_COUNT}")
    print(f"Optimization type: {OPTIMIZATION_TYPE}")
    print(f"Max demurrage limit: {MAX_DEMURRAGE_LIMIT}")
    print("=====================================\n")
    
    # Create and run optimization
    runner = OptimizationRunner()
    results = runner.run_optimization(
        vessel_count=VESSEL_COUNT,
        optimization_type=OPTIMIZATION_TYPE,
        max_demurrage_limit=MAX_DEMURRAGE_LIMIT,
        scenario_name=SCENARIO_NAME
    )
    
    if results["status"] == "success":
        print("\n=== Optimization completed successfully! ===")
    else:
        print(f"\n=== Optimization failed: {results['error']} ===")
        exit(1)


if __name__ == "__main__":
    main()
