"""
Multi-tank refinery optimization main runner.
5-tank system with Tank1-4: 250KB each, Tank5: 180KB.
"""

import json
import time
from pathlib import Path
from tqdm import tqdm

from multi_tank_data_loader import MultiTankDataLoader
from multi_tank_model import MultiTankOptimizationModel
from solver_manager import SolverManager
from result_processor import ResultProcessor


class MultiTankOptimizationRunner:
    """Main runner for multi-tank optimization."""
    
    def __init__(self, config_path: str = "test_data/config.json", vessel_count: int = 5):
        """Initialize the multi-tank optimization runner."""
        print("🚢 Multi-Tank Refinery Optimization System")
        print("=" * 50)
        
        self.config_path = config_path
        self.vessel_count = vessel_count
        
        # The config will be loaded and processed by the data loader
        self.config = None
            
    def load_data(self):
        """Load and prepare all data for multi-tank optimization."""
        print("📊 Loading multi-tank data...")
        
        # Define tank capacities (in barrels) - as specified by user
        tank_config = {
            "TANKS": {
                "Tank1": {"capacity": 250000, "allowed_crudes": []},  # Tank 1: 250 KB
                "Tank2": {"capacity": 250000, "allowed_crudes": []},  # Tank 2: 250 KB
                "Tank3": {"capacity": 250000, "allowed_crudes": []},  # Tank 3: 250 KB
                "Tank4": {"capacity": 250000, "allowed_crudes": []},  # Tank 4: 250 KB
                "Tank5": {"capacity": 180000, "allowed_crudes": []}   # Tank 5: 180 KB
            }
        }
        
        # Initialize data loader
        self.data_loader = MultiTankDataLoader("test_data")
        
        # Load all data with progress tracking (let data loader handle config processing)
        data = self.data_loader.load_all_data()
        
        # Get the processed config from data loader
        self.config = data["config"]
        
        # Apply multi-tank specific configuration
        self.config.update(tank_config)
        
        # Initialize allowed_crudes for the updated tanks
        for tank in self.config["TANKS"]:
            if "allowed_crudes" not in self.config["TANKS"][tank]:
                self.config["TANKS"][tank]["allowed_crudes"] = []
            else:
                self.config["TANKS"][tank]["allowed_crudes"] = []  # Reset for reassignment
        
        # Reassign crudes to tanks with the updated capacities
        print("\n🔄 Reassigning crudes with updated tank capacities...")
        self.tank_assignments = self.data_loader.assign_crudes_to_tanks(data["crudes"], self.config)
        
        print(f"🏭 Tank Configuration:")
        total_capacity = 0
        for tank, info in self.config["TANKS"].items():
            print(f"   {tank}: {info['capacity']:,} KB")
            total_capacity += info['capacity']
        print(f"   Total Capacity: {total_capacity:,} KB")
        print()
        
        # Extract data components (use updated tank assignments)
        self.crudes = data["crudes"]
        self.locations = data["locations"]
        self.time_of_travel = data["time_of_travel"]
        self.crude_availability = data["crude_availability"]
        self.source_locations = data["source_locations"]
        self.products_info = data["products_info"]
        self.crude_margins = data["crude_margins"]
        self.opening_inventory_dict = data["opening_inventory_dict"]
        self.products_ratio = data["products_ratio"]
        self.window_to_days = data["window_to_days"]
        # Use the updated tank assignments instead of the original ones
        # self.tank_assignments = data["tank_assignments"]
        
        print(f"✅ Data loading complete!")
        print(f"   {len(self.crudes)} crude types")
        print(f"   {len(self.locations)} locations")
        print(f"   {self.vessel_count} vessels")
        print(f"   Tank assignments: {self.tank_assignments}")
        print()
        
    def build_model(self, optimization_type: str = "throughput", max_demurrage_limit: int = 10):
        """Build the multi-tank optimization model."""
        print("🏗️  Building multi-tank optimization model...")
        
        # Initialize model builder
        self.model_builder = MultiTankOptimizationModel(
            config=self.config,
            crudes=self.crudes,
            locations=self.locations,
            time_of_travel=self.time_of_travel,
            crude_availability=self.crude_availability,
            source_locations=self.source_locations,
            products_info=self.products_info,
            crude_margins=self.crude_margins,
            opening_inventory_dict=self.opening_inventory_dict,
            products_ratio=self.products_ratio,
            window_to_days=self.window_to_days,
            tank_assignments=self.tank_assignments,
            vessel_count=self.vessel_count
        )
        
        # Build model
        self.model = self.model_builder.build_model(optimization_type, max_demurrage_limit)
        print()
        
    def solve_model(self, solver_name: str = "highs", time_limit: int = 3600):
        """Solve the optimization model."""
        print("⚡ Solving multi-tank optimization...")
        
        # Initialize solver manager
        solver_manager = SolverManager(self.config)
        
        # Solve with progress tracking
        start_time = time.time()
        
        with tqdm(total=100, desc="Optimization progress", unit="%") as pbar:
            # Mock progress updates (actual solver progress varies)
            for i in range(10):
                time.sleep(0.1)  # Simulate setup time
                pbar.update(1)
            
            pbar.set_description("Running HiGHS solver")
            
            # Solve the model
            solver_cfg = {"name": solver_name, "options": {}}
            results = solver_manager.solve_model(
                self.model,
                solver_cfg=solver_cfg,
                scenario_name="multi_tank",
                vessel_count=self.vessel_count,
                optimization_type="throughput"
            )
            
            # Complete progress bar
            pbar.update(90)
        
        solve_time = time.time() - start_time
        
        print(f"✅ Optimization complete!")
        print(f"   Solve time: {solve_time:.2f} seconds")
        print(f"   Status: {results.solver.termination_condition}")
        print()
        
        return results
        
    def process_results(self, results):
        """Process and display optimization results."""
        print("📋 Processing multi-tank results...")
        
        # Initialize result processor
        result_processor = ResultProcessor(self.model, self.config)
        
        # Process results
        processed_results = result_processor.process_results(results)
        
        print("✅ Results processing complete!")
        
        # Display summary with tank information
        if processed_results.get("feasible", False):
            print("\n🎯 Multi-Tank Optimization Summary:")
            print(f"   Total throughput: {processed_results.get('total_throughput', 0):,.0f} KB")
            print(f"   Revenue: ${processed_results.get('revenue', 0):,.0f}")
            
            # Tank utilization summary
            if "tank_utilization" in processed_results:
                print("\n🏭 Tank Utilization:")
                for tank, util in processed_results["tank_utilization"].items():
                    print(f"   {tank}: {util:.1f}%")
        else:
            print("❌ Optimization was not feasible")
            
        return processed_results
        
    def run_optimization(self, optimization_type: str = "throughput", 
                        max_demurrage_limit: int = 10, solver_name: str = "highs",
                        time_limit: int = 3600):
        """Run the complete multi-tank optimization process."""
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Build model
            self.build_model(optimization_type, max_demurrage_limit)
            
            # Step 3: Solve model
            results = self.solve_model(solver_name, time_limit)
            
            # Step 4: Process results
            processed_results = self.process_results(results)
            
            total_time = time.time() - start_time
            print(f"\n⏱️  Total runtime: {total_time:.2f} seconds")
            
            return processed_results
            
        except Exception as e:
            print(f"❌ Error during optimization: {str(e)}")
            raise


def main():
    """Main entry point for multi-tank optimization."""
    
    # Check if we're running directly
    if __name__ == "__main__":
        print("🚀 Starting Multi-Tank Refinery Optimization")
        print("Tank Configuration: Tank1-4 (250KB each), Tank5 (180KB)")
        print()
        
        # Initialize and run optimization
        runner = MultiTankOptimizationRunner(vessel_count=5)
        
        try:
            results = runner.run_optimization(
                optimization_type="throughput",
                max_demurrage_limit=10,
                solver_name="highs",
                time_limit=3600
            )
            
            print("\n🎉 Multi-tank optimization completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⏹️  Optimization interrupted by user")
        except Exception as e:
            print(f"\n💥 Optimization failed: {str(e)}")


if __name__ == "__main__":
    main()
