"""
Two-Tank Scheduling Optimization
Extends the working optimization model with two-tank storage constraints.
Focus: Vessel scheduling, crude selection, production planning with tank constraints.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict, Any
from pyomo.environ import *

from data_loader import DataLoader
from optimization_model import OptimizationModel  
from solver_manager import SolverManager
from result_processor import ResultProcessor


class TwoTankSchedulingModel(OptimizationModel):
    """Two-tank scheduling optimization extending the base model."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with two-tank configuration."""
        super().__init__(*args, **kwargs)
        
        # Two-tank configuration 
        self.TANK_CAPACITIES = {
            "Tank1": 600000,  # 600KB
            "Tank2": 580000   # 580KB  
        }
        
        print(f"🏭 Two-Tank Configuration:")
        print(f"   Tank1: {self.TANK_CAPACITIES['Tank1']:,} barrels")
        print(f"   Tank2: {self.TANK_CAPACITIES['Tank2']:,} barrels")
        print(f"   Total: {sum(self.TANK_CAPACITIES.values()):,} barrels")
        print(f"   Original limit: {self.INVENTORY_MAX_VOLUME:,} barrels")
        
    def assign_crudes_to_tanks(self) -> Dict[str, str]:
        """Assign crudes to tanks based on opening inventory."""
        crude_assignments = {}
        tank_usage = {tank: 0 for tank in self.TANK_CAPACITIES.keys()}
        
        print(f"\n📋 Assigning crudes to tanks...")
        
        # Sort crudes by opening inventory (largest first)
        sorted_crudes = sorted(
            self.opening_inventory_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for crude, inventory in sorted_crudes:
            # Find tank with enough capacity
            assigned = False
            for tank, capacity in self.TANK_CAPACITIES.items():
                available_space = capacity - tank_usage[tank]
                if available_space >= inventory:
                    crude_assignments[crude] = tank
                    tank_usage[tank] += inventory
                    print(f"   {crude}: {inventory:,} KB → {tank}")
                    assigned = True
                    break
            
            if not assigned:
                # Assign to tank with most space (may overflow)
                best_tank = max(self.TANK_CAPACITIES.keys(), 
                               key=lambda t: self.TANK_CAPACITIES[t] - tank_usage[t])
                crude_assignments[crude] = best_tank
                tank_usage[best_tank] += inventory
                print(f"   {crude}: {inventory:,} KB → {best_tank} ⚠️ OVERFLOW")
        
        # Display final utilization
        print(f"\n📊 Tank utilization:")
        total_overflow = 0
        for tank, capacity in self.TANK_CAPACITIES.items():
            usage = tank_usage[tank]
            utilization = (usage / capacity) * 100
            if usage > capacity:
                overflow = usage - capacity
                total_overflow += overflow
                status = f"❌ OVERFLOW +{overflow:,}"
            else:
                status = "✅ OK"
            print(f"   {tank}: {usage:,}/{capacity:,} KB ({utilization:.1f}%) {status}")
        
        if total_overflow > 0:
            print(f"⚠️  Total overflow: {total_overflow:,} KB - model may be infeasible")
        
        return crude_assignments
    
    def build_tank_parameters(self):
        """Build tank-specific parameters."""
        model = self.model
        
        # Tank sets
        model.TANKS = Set(initialize=list(self.TANK_CAPACITIES.keys()))
        
        # Tank capacity parameters
        model.TankCapacity = Param(model.TANKS, initialize=self.TANK_CAPACITIES)
        
        # Crude-to-tank assignment
        self.crude_tank_assignment = self.assign_crudes_to_tanks()
        
        # Opening inventory by tank and crude
        opening_inventory_by_tank = {}
        for crude, inventory in self.opening_inventory_dict.items():
            assigned_tank = self.crude_tank_assignment[crude]
            opening_inventory_by_tank[(assigned_tank, crude)] = inventory
        
        # Initialize all tank-crude combinations to 0, then update with actual values
        for tank in model.TANKS:
            for crude in model.CRUDES:
                if (tank, crude) not in opening_inventory_by_tank:
                    opening_inventory_by_tank[(tank, crude)] = 0
        
        model.TankOpeningInventory = Param(model.TANKS, model.CRUDES, 
                                         initialize=opening_inventory_by_tank)
    
    def build_tank_variables(self):
        """Build tank-specific variables."""
        model = self.model
        
        # Tank inventory - inventory of each crude in each tank on each day
        model.TankInventory = Var(model.TANKS, model.CRUDES, model.DAYS, 
                                domain=NonNegativeReals, 
                                bounds=(0, None))
        
        # Crude arrivals to tanks (from vessels)
        model.CrudeArrival = Var(model.TANKS, model.CRUDES, model.DAYS,
                               domain=NonNegativeReals)
        
        # Crude consumption from tanks
        model.CrudeConsumption = Var(model.TANKS, model.CRUDES, model.DAYS,
                                   domain=NonNegativeReals)
    
    def build_tank_constraints(self):
        """Build tank-specific constraints."""
        model = self.model
        
        # Tank capacity constraints
        def tank_capacity_rule(model, tank, day):
            return sum(model.TankInventory[tank, crude, day] for crude in model.CRUDES) <= model.TankCapacity[tank]
        model.TankCapacityConstraint = Constraint(model.TANKS, model.DAYS, rule=tank_capacity_rule)
        
        # Tank inventory balance
        def tank_inventory_balance_rule(model, tank, crude, day):
            if day == self.config["DAYS"]["start"]:
                # First day: opening inventory + arrivals - consumption
                return (model.TankInventory[tank, crude, day] == 
                       model.TankOpeningInventory[tank, crude] + 
                       model.CrudeArrival[tank, crude, day] - 
                       model.CrudeConsumption[tank, crude, day])
            else:
                # Other days: previous inventory + arrivals - consumption  
                return (model.TankInventory[tank, crude, day] == 
                       model.TankInventory[tank, crude, day-1] + 
                       model.CrudeArrival[tank, crude, day] - 
                       model.CrudeConsumption[tank, crude, day])
        
        model.TankInventoryBalance = Constraint(model.TANKS, model.CRUDES, model.DAYS, 
                                              rule=tank_inventory_balance_rule)
        
        # Link crude arrivals to vessel discharges
        def crude_arrival_link_rule(model, tank, crude, day):
            # Only the assigned tank receives crude arrivals for each crude
            if crude in self.crude_tank_assignment and self.crude_tank_assignment[crude] == tank:
                # Arrivals come from vessel discharges (day-5 due to discharge delay)
                if day > 5:
                    return model.CrudeArrival[tank, crude, day] == sum(
                        model.VolumeDischarged[vessel, crude, day-5] 
                        for vessel in model.VESSELS
                    )
                else:
                    return model.CrudeArrival[tank, crude, day] == 0
            else:
                return model.CrudeArrival[tank, crude, day] == 0
        
        model.CrudeArrivalLink = Constraint(model.TANKS, model.CRUDES, model.DAYS,
                                          rule=crude_arrival_link_rule)
        
        # Link tank consumption to total crude consumption
        def tank_consumption_link_rule(model, crude, day):
            # Total crude consumption = sum across all tanks
            total_from_tanks = sum(model.CrudeConsumption[tank, crude, day] for tank in model.TANKS)
            
            # This should equal the blend consumption
            blend_consumption = sum(
                model.BCb[blend] * model.BRcb[blend, crude] * 
                (model.BlendFraction[blend, 2*day-1] + model.BlendFraction[blend, 2*day])
                for blend in model.BLENDS
            )
            
            return total_from_tanks == blend_consumption
        
        model.TankConsumptionLink = Constraint(model.CRUDES, model.DAYS,
                                             rule=tank_consumption_link_rule)
        
        # Crude can only be consumed from assigned tank
        def crude_tank_assignment_rule(model, tank, crude, day):
            if crude in self.crude_tank_assignment and self.crude_tank_assignment[crude] != tank:
                return model.CrudeConsumption[tank, crude, day] == 0
            else:
                return Constraint.Skip
        
        model.CrudeTankAssignment = Constraint(model.TANKS, model.CRUDES, model.DAYS,
                                             rule=crude_tank_assignment_rule)
    
    def replace_inventory_constraints(self):
        """Replace original inventory constraints with tank-based ones."""
        # Remove the original inventory update constraint
        if hasattr(self.model, 'InventoryUpdate'):
            self.model.del_component('InventoryUpdate')
        
        # Replace with tank-based inventory tracking
        def total_inventory_rule(model, crude, day):
            # Total inventory = sum across tanks  
            return model.Inventory[crude, day] == sum(
                model.TankInventory[tank, crude, day] for tank in model.TANKS
            )
        
        self.model.TotalInventoryRule = Constraint(self.model.CRUDES, self.model.DAYS,
                                                 rule=total_inventory_rule)
    
    def build_model(self, optimization_type: str = "throughput", max_demurrage_limit: int = 10):
        """Build the complete two-tank scheduling model."""
        print(f"🔧 Building two-tank scheduling model...")
        
        # Build base model first
        super().build_model(optimization_type, max_demurrage_limit)
        
        # Add tank-specific components
        self.build_tank_parameters()
        self.build_tank_variables() 
        self.build_tank_constraints()
        self.replace_inventory_constraints()
        
        print(f"✅ Two-tank model built successfully!")
        print(f"   Variables: {self.model.nvariables()}")
        print(f"   Constraints: {self.model.nconstraints()}")
        
        return self.model


class TwoTankSchedulingRunner:
    """Runner for two-tank scheduling optimization."""
    
    def __init__(self, data_path: str = "test_data", output_path: str = "results"):
        """Initialize the runner."""
        self.data_path = data_path
        self.output_path = output_path
        self.data_loader = DataLoader(data_path)
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
    
    def load_data(self):
        """Load all required data."""
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
                        max_demurrage_limit: int = 10, scenario_name: str = "two_tank") -> Dict[str, Any]:
        """
        Run two-tank scheduling optimization.
        
        Args:
            vessel_count: Number of vessels
            optimization_type: "throughput" or "margin"
            max_demurrage_limit: Maximum demurrage days
            scenario_name: Scenario name for outputs
            
        Returns:
            Results dictionary
        """
        try:
            # Load data
            (config, crudes, locations, time_of_travel, crude_availability, 
             source_locations, products_info, crude_margins, opening_inventory_dict,
             products_ratio, window_to_days) = self.load_data()
            
            # Create start date
            start_date = self.create_start_date(config)
            
            print(f"\n🚀 Starting Two-Tank Scheduling Optimization")
            print(f"   Vessels: {vessel_count}")
            print(f"   Optimization: {optimization_type}")
            print(f"   Time horizon: {config['DAYS']['start']}-{config['DAYS']['end']} days")
            
            # Create two-tank optimization model
            opt_model = TwoTankSchedulingModel(
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
            print(f"🚀 Solving two-tank scheduling model...")
            solver_manager = SolverManager(config)
            results = solver_manager.solve_model(
                model=model,
                scenario_name=scenario_name,
                vessel_count=vessel_count,
                optimization_type=optimization_type,
                max_demurrage_limit=max_demurrage_limit
            )
            
            if results and results.solver.termination_condition == TerminationCondition.optimal:
                # Process results
                print(f"📋 Processing optimization results...")
                result_processor = ResultProcessor(model, config, crudes, start_date)
                
                # Extract results
                crude_blending_df = result_processor.extract_crude_blending_results()
                vessel_routing_df = result_processor.extract_vessel_routing_results(crude_availability)
                summary_metrics = result_processor.calculate_summary_metrics(crude_blending_df)
                
                # Generate output files
                crude_blending_filename = f'two_tank_crude_blending_{optimization_type}_{vessel_count}vessels_{config["DAYS"]["end"]}days.csv'
                vessel_routing_filename = f'two_tank_vessel_routing_{optimization_type}_{vessel_count}vessels_{config["DAYS"]["end"]}days.csv'
                
                # Save results
                crude_blending_path = os.path.join(self.output_path, crude_blending_filename)
                vessel_routing_path = os.path.join(self.output_path, vessel_routing_filename)
                
                crude_blending_df.to_csv(crude_blending_path, index=False)
                vessel_routing_df.to_csv(vessel_routing_path, index=False)
                
                # Print results
                print(f"\n🎉 Two-Tank Optimization Complete!")
                print(f"📁 Results saved:")
                print(f"   Crude blending: {crude_blending_path}")
                print(f"   Vessel routing: {vessel_routing_path}")
                
                print(f"\n📊 Summary Metrics:")
                print(f"   Total throughput: {summary_metrics['total_throughput']:.1f} k barrels")
                print(f"   Total margin: ${summary_metrics['total_margin']:,.0f}")
                print(f"   Average throughput: {summary_metrics['average_throughput']:.1f} k barrels/day")
                print(f"   Total demurrage: ${summary_metrics['total_demurrage']:,.0f}")
                
                # Tank utilization summary
                print(f"\n🏭 Tank Assignment Summary:")
                for crude, tank in opt_model.crude_tank_assignment.items():
                    inventory = opening_inventory_dict[crude]
                    if inventory > 0:
                        print(f"   {crude}: {inventory:,} KB → {tank}")
                
                return {
                    "status": "success",
                    "crude_blending_df": crude_blending_df,
                    "vessel_routing_df": vessel_routing_df,
                    "summary_metrics": summary_metrics,
                    "crude_blending_file": crude_blending_path,
                    "vessel_routing_file": vessel_routing_path,
                    "tank_assignments": opt_model.crude_tank_assignment,
                    "model": opt_model,
                    "solver_results": results
                }
            
            else:
                # Handle infeasible or failed solution
                termination = results.solver.termination_condition if results else "unknown"
                print(f"\n❌ Optimization failed!")
                print(f"   Termination condition: {termination}")
                
                if termination == TerminationCondition.infeasible:
                    print(f"   Model is infeasible - tank capacity constraints may be too tight")
                    print(f"   Consider:")
                    print(f"   - Increasing tank capacities")
                    print(f"   - Reducing opening inventory") 
                    print(f"   - Using IIS analysis to identify conflicting constraints")
                
                return {
                    "status": "failed",
                    "termination_condition": str(termination),
                    "model": opt_model if 'opt_model' in locals() else None
                }
            
        except Exception as e:
            print(f"❌ Optimization failed with error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


def main():
    """Main function for two-tank scheduling optimization."""
    
    print("=== Two-Tank Scheduling Optimization ===")
    print("Tank 1: 600,000 barrels")
    print("Tank 2: 580,000 barrels")
    print("Focus: Vessel scheduling + Production planning")
    print("========================================\n")
    
    # Configuration
    VESSEL_COUNT = 6
    OPTIMIZATION_TYPE = "throughput"  # or "margin"
    MAX_DEMURRAGE_LIMIT = 10
    SCENARIO_NAME = "two_tank_test"
    
    # Create and run optimization
    runner = TwoTankSchedulingRunner()
    results = runner.run_optimization(
        vessel_count=VESSEL_COUNT,
        optimization_type=OPTIMIZATION_TYPE,
        max_demurrage_limit=MAX_DEMURRAGE_LIMIT,
        scenario_name=SCENARIO_NAME
    )
    
    if results["status"] == "success":
        print(f"\n=== Two-Tank Scheduling Optimization Successful! ===")
    else:
        print(f"\n=== Two-Tank Scheduling Optimization Failed ===")
        if "error" in results:
            print(f"Error: {results['error']}")
        elif "termination_condition" in results:
            print(f"Termination: {results['termination_condition']}")


if __name__ == "__main__":
    main()
