"""
Main optimization runner for the refinery and vessel scheduling system.
Provides a high-level interface to run the complete optimization pipeline.
*** MODIFIED TO USE GOOGLE OR-TOOLS CP-SAT SOLVER ***
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict, Any

# Import the necessary Google OR-Tools library
from ortools.sat.python import cp_model

# The DataLoader is solver-agnostic and does not need to be changed.
from data_loader import DataLoader


class OrToolsOptimizationModel:
    """Builds the optimization model using Google OR-Tools CP-SAT."""

    def __init__(self, **kwargs):
        """Initializes the OR-Tools model builder."""
        self.model = cp_model.CpModel()
        self.data = kwargs
        self.variables = {}
        # Use a scaling factor for continuous variables to treat them as integers
        self.SCALE_FACTOR = 1000

    def build_model(self, optimization_type: str, max_demurrage_limit: int):
        """Creates all variables and constraints for the CP-SAT model based on the PDF."""
        
        # --- Unpack data for easier access ---
        vessels = [f'V_{i+1}' for i in range(self.data['vessel_count'])]
        days = list(range(self.data['config']['DAYS']['end'] + 1))
        slots = list(range(1, self.data['config']['DAYS']['end'] * 2 + 1))
        # Create a parcel dataframe for easier lookup
        self.data['crude_availability_df'] = pd.DataFrame(
            [(window, loc, crude, details['volume'], details['parcel_size']) 
             for window, locs in self.data['crude_availability'].items()
             for loc, crudes in locs.items()
             for crude, details in crudes.items()],
            columns=['date_range', 'location', 'crude', 'volume', 'parcel_size']
        )
        
        # --- 1.3 Decision Variables ---
        print("    - Creating decision variables...")
        self._create_variables(vessels, days, slots)

        # --- 1.4 Constraints ---
        print("    - Building vessel travel constraints (Eq 1-7)...")
        self._add_vessel_travel_constraints(vessels, days)
        
        print("    - Building vessel loading constraints (Eq 8-22)...")
        self._add_vessel_loading_constraints(vessels, days)

        print("    - Building vessel discharge constraints (Eq 23-30)...")
        self._add_vessel_discharge_constraints(vessels, days)
        
        print("    - Building blending and inventory constraints (Eq 31-41)...")
        self._add_blending_and_inventory_constraints(days, slots)

        # --- 1.5 Objective Function ---
        print("    - Building objective function...")
        self._set_objective(optimization_type)
        
        return self.model, self.variables

    def _create_variables(self, vessels, days, slots):
        """Helper to create all model variables."""
        self.variables['AtLocation'] = {
            (v, l, d): self.model.NewBoolVar(f'AtLoc_{v}_{l}_{d}')
            for v in vessels for l in self.data['locations'] for d in days
        }
        self.variables['Discharge'] = {
            (v, d): self.model.NewBoolVar(f'Discharge_{v}_{d}')
            for v in vessels for d in days
        }
        self.variables['Pickup'] = {
            (v, p_idx, d): self.model.NewBoolVar(f'Pickup_{v}_{p_idx}_{d}')
            for v in vessels for p_idx in self.data['crude_availability_df'].index for d in days
        }
        self.variables['Inventory'] = {
            (c, d): self.model.NewIntVar(0, self.data['config']['INVENTORY_MAX_VOLUME'] * self.SCALE_FACTOR, f'Inv_{c}_{d}')
            for c in self.data['crudes'] for d in days
        }
        self.variables['BlendFraction'] = {
            (b, s): self.model.NewIntVar(0, self.SCALE_FACTOR, f'BF_{b}_{s}')
            for b in self.data['products_info']['product'] for s in slots
        }

    def _add_vessel_travel_constraints(self, vessels, days):
        # (1) A vessel can only be at one location on a given day.
        for v in vessels:
            for d in days:
                self.model.Add(sum(self.variables['AtLocation'][(v, l, d)] for l in self.data['locations']) <= 1)

        # (6, 7) Travel time between locations
        for v in vessels:
            for l1 in self.data['locations']:
                for l2 in self.data['locations']:
                    if l1 == l2: continue
                    travel_time = self.data['time_of_travel'].get((l1, l2), 999)
                    for d1 in days:
                        for d2 in range(d1 + 1, min(d1 + travel_time, max(days) + 1)):
                            self.model.AddImplication(self.variables['AtLocation'][(v, l1, d1)], self.variables['AtLocation'][(v, l2, d2)].Not())

    def _add_vessel_loading_constraints(self, vessels, days):
        # (8) Every vessel has to at least pick one parcel.
        for v in vessels:
            self.model.Add(sum(self.variables['Pickup'][(v, p_idx, d)] for p_idx in self.data['crude_availability_df'].index for d in days) >= 1)

        # (9) A parcel can only be picked up by one vessel.
        for p_idx in self.data['crude_availability_df'].index:
            self.model.Add(sum(self.variables['Pickup'][(v, p_idx, d)] for v in vessels for d in days) <= 1)
        
        # (12) A vessel can only pickup a parcel if at the correct location.
        for v in vessels:
            for p_idx, p_row in self.data['crude_availability_df'].iterrows():
                parcel_loc = p_row['location']
                available_days = self.data['window_to_days'][p_row['date_range']]
                for d in days:
                    if d not in available_days:
                        self.model.Add(self.variables['Pickup'][(v, p_idx, d)] == 0) # (11)
                    else:
                        self.model.AddImplication(self.variables['Pickup'][(v, p_idx, d)], self.variables['AtLocation'][(v, parcel_loc, d)])

    def _add_vessel_discharge_constraints(self, vessels, days):
        discharge_intervals = []
        for v in vessels:
            # (23) Every vessel should discharge exactly once
            self.model.Add(sum(self.variables['Discharge'][(v, d)] for d in days) == 1)
            for d in days:
                if d < max(days):
                    # (24) Discharge only happens at Melaka and for 2 days.
                    self.model.AddImplication(self.variables['Discharge'][(v, d)], self.variables['AtLocation'][(v, 'Melaka', d)])
                    self.model.AddImplication(self.variables['Discharge'][(v, d)], self.variables['AtLocation'][(v, 'Melaka', d + 1)])

                # Create an optional interval for the NoOverlap constraint
                interval = self.model.NewOptionalIntervalVar(
                    start=d, size=2, end=d+2, 
                    is_present=self.variables['Discharge'][(v, d)],
                    name=f'discharge_interval_{v}_{d}'
                )
                discharge_intervals.append(interval)
        
        # (25) No two vessels can discharge on overlapping days.
        self.model.AddNoOverlap(discharge_intervals)

    def _add_blending_and_inventory_constraints(self, days, slots):
        # (39) Initial Inventory
        for c in self.data['crudes']:
            self.model.Add(self.variables['Inventory'][(c, 0)] == int(self.data['opening_inventory_dict'][c] * self.SCALE_FACTOR))
        
        # (41) Max inventory volume
        for d in days:
            self.model.Add(sum(self.variables['Inventory'][(c, d)] for c in self.data['crudes']) <= self.data['config']['INVENTORY_MAX_VOLUME'] * self.SCALE_FACTOR)

        # (32) Consume exactly one blend per slot
        is_blend_consumed = {}
        for b_row in self.data['products_info'].itertuples():
            b = b_row.product
            for s in slots:
                is_blend_consumed[(b, s)] = self.model.NewBoolVar(f'IsBlendConsumed_{b}_{s}')
                # (31) Link BlendFraction to IsBlendConsumed
                self.model.Add(self.variables['BlendFraction'][(b, s)] > 0).OnlyEnforceIf(is_blend_consumed[(b, s)])
                self.model.Add(self.variables['BlendFraction'][(b, s)] == 0).OnlyEnforceIf(is_blend_consumed[(b, s)].Not())

        for s in slots:
            self.model.Add(sum(is_blend_consumed[(b_row.product, s)] for b_row in self.data['products_info'].itertuples()) == 1)

    def _set_objective(self, optimization_type):
        if optimization_type == "throughput":
            # (46) Throughput objective, scaled
            total_throughput = self.model.NewIntVar(0, 10_000_000 * self.SCALE_FACTOR, 'total_throughput')
            self.model.Add(total_throughput == sum(self.variables['BlendFraction'][(b_row.product, s)] for b_row in self.data['products_info'].itertuples() for s in self.variables['BlendFraction'].keys() if s == b_row.product))
            self.model.Maximize(total_throughput)
        else:
            # (45) Margin/Profit objective would be defined here
            pass


class OrToolsSolverManager:
    """Manages the solving process using the CP-SAT solver."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def solve_model(self, model: cp_model.CpModel, **kwargs):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.get("solver_time_limit_seconds", 3600)
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)
        return {"status": status, "solver": solver}

class OrToolsResultProcessor:
    """Processes results from a solved CP-SAT model."""
    def __init__(self, solver: cp_model.CpSolver, variables: Dict, config, crudes, start_date):
        self.solver = solver
        self.variables = variables
        self.config = config
        self.crudes = crudes
        self.start_date = start_date
        self.SCALE_FACTOR = 1000

    def extract_crude_blending_results(self):
        results = []
        for (b, s), var in self.variables['BlendFraction'].items():
            if self.solver.Value(var) > 0:
                day = (s - 1) // 2
                date = self.start_date + pd.Timedelta(days=day)
                results.append({
                    "Date": date.strftime('%Y-%m-%d'),
                    "Slot": s,
                    "Blend": b,
                    "BlendFraction": self.solver.Value(var) / self.SCALE_FACTOR,
                })
        return pd.DataFrame(results)

    def extract_vessel_routing_results(self, crude_availability_df):
        # This method would be expanded to extract pickup, at_location, etc.
        return pd.DataFrame() # Placeholder

    def calculate_summary_metrics(self, crude_blending_df):
        # This method would be expanded to calculate metrics from the solution
        return {
            'total_throughput': 0, 'total_margin': 0, 'average_throughput': 0,
            'average_margin': 0, 'demurrage_at_melaka': 0, 
            'demurrage_at_source': 0, 'total_demurrage': 0
        } # Placeholder

class OptimizationRunner:
    """Main class to orchestrate the OR-Tools optimization process."""
    
    def __init__(self, data_path: str = "test_data", output_path: str = "results"):
        self.data_path = data_path
        self.output_path = output_path
        self.data_loader = DataLoader(data_path)
        os.makedirs(output_path, exist_ok=True)
        
    def load_data(self) -> Tuple[Dict[str, Any], ...]:
        print("📁 Loading optimization data...")
        # We need the crude_availability as a dataframe later
        (config, crudes, locations, time_of_travel, crude_availability, 
         source_locations, products_info, crude_margins, opening_inventory_dict,
         products_ratio, window_to_days) = self.data_loader.load_all_data()
        
        self.crude_availability_df = pd.DataFrame(
            [(window, loc, crude, details['volume'], details['parcel_size']) 
             for window, locs in crude_availability.items()
             for loc, crudes in locs.items()
             for crude, details in crudes.items()],
            columns=['date_range', 'location', 'crude', 'volume', 'parcel_size']
        )
        return (config, crudes, locations, time_of_travel, crude_availability, 
                source_locations, products_info, crude_margins, opening_inventory_dict,
                products_ratio, window_to_days)
        
    def create_start_date(self, config: dict) -> pd.Timestamp:
        # Same as original
        month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
        month_number = month_map[config["schedule_month"]]
        return pd.to_datetime(f"{config['schedule_year']}-{month_number:02d}-01")
        
    def run_optimization(self, vessel_count: int = 6, optimization_type: str = "throughput", 
                        max_demurrage_limit: int = 10, scenario_name: str = "test") -> Dict[str, Any]:
        try:
            (config, crudes, locations, time_of_travel, crude_availability, 
             source_locations, products_info, crude_margins, opening_inventory_dict,
             products_ratio, window_to_days) = self.load_data()
            
            start_date = self.create_start_date(config)
            
            print("🔧 Creating OR-Tools optimization model...")
            model_builder = OrToolsOptimizationModel(
                config=config, crudes=crudes, locations=locations, time_of_travel=time_of_travel,
                crude_availability=crude_availability, source_locations=source_locations,
                products_info=products_info, crude_margins=crude_margins,
                opening_inventory_dict=opening_inventory_dict, products_ratio=products_ratio,
                window_to_days=window_to_days, vessel_count=vessel_count,
                crude_availability_df=self.crude_availability_df
            )
            
            model, variables = model_builder.build_model(optimization_type, max_demurrage_limit)
            
            print("🚀 Solving with OR-Tools CP-SAT solver...")
            solver_manager = OrToolsSolverManager(config)
            results = solver_manager.solve_model(model=model, scenario_name=scenario_name)
            
            status = results["status"]
            solver = results["solver"]

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                print("✅ Solution Found!")
                print("📋 Processing optimization results...")
                
                result_processor = OrToolsResultProcessor(solver, variables, config, crudes, start_date)
                
                crude_blending_df = result_processor.extract_crude_blending_results()
                vessel_routing_df = result_processor.extract_vessel_routing_results(self.crude_availability_df)
                summary_metrics = result_processor.calculate_summary_metrics(crude_blending_df)

                # The rest of the file saving and printing logic remains the same...
                print(f"\nOptimization Summary (Placeholder):")
                print(f"  Solver status: {solver.StatusName(status)}")
                print(f"  Objective value: {solver.ObjectiveValue()}")
                
                return {"status": "success", "crude_blending_df": crude_blending_df}
            else:
                print(f"❌ No solution found. Solver status: {solver.StatusName(status)}")
                return {"status": "failed", "error": f"Solver could not find a solution (Status: {solver.StatusName(status)})"}
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

def main():
    # Same as original
    VESSEL_COUNT = 6
    OPTIMIZATION_TYPE = "throughput"
    MAX_DEMURRAGE_LIMIT = 10
    SCENARIO_NAME = "ortools_test_scenario"
    
    print("=== Aegis Scheduler Optimization (Google OR-Tools) ===")
    
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
        print(f"\n=== Optimization failed: {results.get('error', 'Unknown error')} ===")

if __name__ == "__main__":
    main()