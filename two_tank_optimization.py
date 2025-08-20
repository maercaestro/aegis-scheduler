"""
Two-Tank Optimization System with IIS Analysis
Based on the working main.py but enhanced with multi-tank constraints and feasibility analysis.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List
from pyomo.environ import *
import json

from data_loader import DataLoader
from solver_manager import SolverManager


class TwoTankOptimizationModel:
    """Two-tank optimization model with IIS analysis capabilities."""
    
    def __init__(self, config: dict, crudes: list, locations: list, 
                 time_of_travel: dict, crude_availability: dict,
                 source_locations: list, products_info: pd.DataFrame,
                 crude_margins: list, opening_inventory_dict: dict,
                 products_ratio: dict, window_to_days: dict, vessel_count: int):
        """Initialize the two-tank optimization model."""
        
        self.config = config
        self.crudes = crudes
        self.locations = locations
        self.time_of_travel = time_of_travel
        self.crude_availability = crude_availability
        self.source_locations = source_locations
        self.products_info = products_info
        self.crude_margins = crude_margins
        self.opening_inventory_dict = opening_inventory_dict
        self.products_ratio = products_ratio
        self.window_to_days = window_to_days
        self.vessel_count = vessel_count
        
        # Two-tank configuration
        self.TANK_CAPACITIES = {
            "Tank1": 600000,  # 600KB
            "Tank2": 580000   # 580KB
        }
        
        # Total capacity should match original
        total_capacity = sum(self.TANK_CAPACITIES.values())
        print(f"Two-tank total capacity: {total_capacity:,} barrels")
        print(f"Original capacity: {config.get('INVENTORY_MAX_VOLUME', 1180000):,} barrels")
        
        # Constants from config
        self.INVENTORY_MAX_VOLUME = config["INVENTORY_MAX_VOLUME"]
        self.MaxTransitions = config["MaxTransitions"]
        
        # Create vessels list
        config["VESSELS"] = list(range(1, vessel_count + 1))
        
        # Initialize model
        self.model = ConcreteModel()
        
    def assign_crudes_to_tanks(self) -> Dict[str, str]:
        """Assign crudes to tanks based on opening inventory."""
        crude_assignments = {}
        tank_usage = {tank: 0 for tank in self.TANK_CAPACITIES.keys()}
        
        print("\n🏭 Assigning crudes to tanks...")
        
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
                    print(f"  {crude}: {inventory:,} KB → {tank} (remaining: {available_space - inventory:,} KB)")
                    assigned = True
                    break
            
            if not assigned:
                # If no single tank can fit, assign to tank with most space
                best_tank = max(self.TANK_CAPACITIES.keys(), 
                               key=lambda t: self.TANK_CAPACITIES[t] - tank_usage[t])
                crude_assignments[crude] = best_tank
                tank_usage[best_tank] += inventory
                print(f"  {crude}: {inventory:,} KB → {best_tank} (OVERFLOW: {tank_usage[best_tank] - self.TANK_CAPACITIES[best_tank]:,} KB)")
        
        # Display final assignment
        print(f"\n📊 Final tank utilization:")
        for tank, capacity in self.TANK_CAPACITIES.items():
            usage = tank_usage[tank]
            utilization = (usage / capacity) * 100
            status = "✅ OK" if usage <= capacity else "❌ OVERFLOW"
            print(f"  {tank}: {usage:,}/{capacity:,} KB ({utilization:.1f}%) {status}")
        
        return crude_assignments
    
    def build_sets(self):
        """Build Pyomo sets including tank sets."""
        model = self.model
        
        # Original sets
        model.CRUDES = Set(initialize=self.crudes)
        model.LOCATIONS = Set(initialize=self.locations)
        model.SOURCE_LOCATIONS = Set(initialize=self.source_locations)
        model.VESSELS = Set(initialize=self.config["VESSELS"])
        model.DAYS = RangeSet(self.config["DAYS"]["start"], self.config["DAYS"]["end"])
        model.BLENDS = Set(initialize=self.products_info['product'].tolist(), dimen=None)
        model.SLOTS = RangeSet(self.config["DAYS"]["start"], 2 * self.config["DAYS"]["end"])
        
        # Tank sets
        model.TANKS = Set(initialize=list(self.TANK_CAPACITIES.keys()))
        
        # Build parcels set
        parcel_set = set()
        for window, loc_data in self.crude_availability.items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    parcel_set.add((location, crude_type, window))
        model.PARCELS = Set(initialize=parcel_set, dimen=3)
        
    def build_parameters(self):
        """Build Pyomo parameters including tank parameters."""
        model = self.model
        
        # Original parameters
        products_capacity = dict(zip(self.products_info['product'].tolist(), 
                                   self.products_info['max_per_day']))
        crude_margins_dict = dict(zip(self.crudes, self.crude_margins))
        
        model.BCb = Param(model.BLENDS, initialize=products_capacity)
        model.BRcb = Param(model.BLENDS, model.CRUDES, initialize=self.products_ratio, default=0)
        model.MRc = Param(model.CRUDES, initialize=crude_margins_dict)
        
        # Tank parameters
        model.TankCapacity = Param(model.TANKS, initialize=self.TANK_CAPACITIES)
        
        # Crude to tank assignment
        self.crude_tank_assignment = self.assign_crudes_to_tanks()
        
        # Parcel parameters
        parcel_size = {}
        for window, loc_data in self.crude_availability.items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    key = (location, crude_type, window)
                    parcel_size[key] = info["parcel_size"]
        model.PVp = Param(model.PARCELS, initialize=parcel_size)
        
        # Travel time parameters
        travel_time_dict = {}
        for loc1 in self.locations:
            for loc2 in self.locations:
                if loc1 == loc2:
                    travel_time_dict[(loc1, loc2)] = 0
                else:
                    travel_time_dict[(loc1, loc2)] = self.time_of_travel.get((loc1, loc2), 7)
        model.TTij = Param(model.LOCATIONS, model.LOCATIONS, initialize=travel_time_dict)
        
        # Window to days mapping
        window_to_days_dict = {}
        for window, days_list in self.window_to_days.items():
            for day in days_list:
                window_to_days_dict[(window, day)] = 1
        model.WDwd = Param(model.PARCELS, model.DAYS, 
                          initialize=lambda model, l, c, w, d: window_to_days_dict.get((w, d), 0))
        
        # Opening inventory by tank and crude
        opening_inventory_by_tank = {}
        for crude, inventory in self.opening_inventory_dict.items():
            assigned_tank = self.crude_tank_assignment[crude]
            opening_inventory_by_tank[(assigned_tank, crude)] = inventory
        
        # Initialize all combinations to 0, then update with actual values
        for tank in model.TANKS:
            for crude in model.CRUDES:
                if (tank, crude) not in opening_inventory_by_tank:
                    opening_inventory_by_tank[(tank, crude)] = 0
        
        model.OpeningInventory = Param(model.TANKS, model.CRUDES, 
                                     initialize=opening_inventory_by_tank)
    
    def build_variables(self):
        """Build decision variables including tank inventory variables."""
        model = self.model
        
        # Original variables
        model.x = Var(model.PARCELS, model.VESSELS, model.DAYS, domain=Binary)
        model.y = Var(model.VESSELS, model.LOCATIONS, model.DAYS, domain=Binary)
        model.z = Var(model.BLENDS, model.DAYS, domain=NonNegativeReals)
        model.w = Var(model.CRUDES, model.DAYS, domain=NonNegativeReals)
        
        # Tank inventory variables - inventory of each crude in each tank on each day
        model.tank_inventory = Var(model.TANKS, model.CRUDES, model.DAYS, 
                                 domain=NonNegativeReals, 
                                 bounds=(0, None))
        
        # Crude consumption from tanks
        model.crude_consumed = Var(model.TANKS, model.CRUDES, model.DAYS,
                                 domain=NonNegativeReals)
    
    def build_constraints(self):
        """Build all constraints including tank-specific constraints."""
        model = self.model
        
        # Original constraints (vessel and routing)
        self.build_vessel_constraints()
        self.build_routing_constraints()
        
        # Tank-specific constraints
        self.build_tank_constraints()
        
        # Production constraints
        self.build_production_constraints()
    
    def build_tank_constraints(self):
        """Build tank-specific constraints."""
        model = self.model
        
        # Tank capacity constraints
        def tank_capacity_rule(model, tank, day):
            return sum(model.tank_inventory[tank, crude, day] for crude in model.CRUDES) <= model.TankCapacity[tank]
        model.TankCapacityConstraint = Constraint(model.TANKS, model.DAYS, rule=tank_capacity_rule)
        
        # Tank inventory balance constraints
        def tank_inventory_balance_rule(model, tank, crude, day):
            # Get crude arrivals for this tank
            crude_arrivals = 0
            if crude in self.crude_tank_assignment and self.crude_tank_assignment[crude] == tank:
                # Sum arrivals from all parcels and vessels for this crude
                for location, crude_type, window in model.PARCELS:
                    if crude_type == crude:  # crude type matches
                        for vessel in model.VESSELS:
                            crude_arrivals += (model.x[(location, crude_type, window), vessel, day] * 
                                             model.PVp[(location, crude_type, window)] * 
                                             model.WDwd[(location, crude_type, window), day])
            
            if day == self.config["DAYS"]["start"]:
                # First day: opening inventory + arrivals - consumption
                return (model.tank_inventory[tank, crude, day] == 
                       model.OpeningInventory[tank, crude] + crude_arrivals - 
                       model.crude_consumed[tank, crude, day])
            else:
                # Other days: previous inventory + arrivals - consumption
                return (model.tank_inventory[tank, crude, day] == 
                       model.tank_inventory[tank, crude, day-1] + crude_arrivals - 
                       model.crude_consumed[tank, crude, day])
        
        model.TankInventoryBalance = Constraint(model.TANKS, model.CRUDES, model.DAYS, 
                                              rule=tank_inventory_balance_rule)
        
        # Link tank consumption to total crude usage
        def crude_consumption_link_rule(model, crude, day):
            return (model.w[crude, day] == 
                   sum(model.crude_consumed[tank, crude, day] for tank in model.TANKS))
        model.CrudeConsumptionLink = Constraint(model.CRUDES, model.DAYS, 
                                              rule=crude_consumption_link_rule)
        
        # Crude can only be consumed from assigned tank
        def crude_tank_assignment_rule(model, tank, crude, day):
            if crude in self.crude_tank_assignment and self.crude_tank_assignment[crude] != tank:
                return model.crude_consumed[tank, crude, day] == 0
            else:
                return Constraint.Skip
        model.CrudeTankAssignment = Constraint(model.TANKS, model.CRUDES, model.DAYS,
                                             rule=crude_tank_assignment_rule)
    
    def build_vessel_constraints(self):
        """Build vessel-related constraints from original model."""
        model = self.model
        
        # Vessel capacity constraints
        def vessel_capacity_constraint_rule(model, vessel, day):
            total_volume = sum(model.x[(location, crude, window), vessel, day] * model.PVp[(location, crude, window)] * 
                             model.WDwd[(location, crude, window), day] 
                             for location, crude, window in model.PARCELS)
            return total_volume <= self.config.get("vessel_max_limit", 700000)
        model.VesselCapacityConstraint = Constraint(model.VESSELS, model.DAYS, 
                                                   rule=vessel_capacity_constraint_rule)
        
        # One location per vessel per day
        def one_location_per_vessel_rule(model, vessel, day):
            return sum(model.y[vessel, location, day] for location in model.LOCATIONS) <= 1
        model.OneLocationPerVessel = Constraint(model.VESSELS, model.DAYS, 
                                              rule=one_location_per_vessel_rule)
        
        # Vessel can only pick parcels from current location
        def parcel_location_rule(model, location, crude, window, vessel, day):
            return model.x[(location, crude, window), vessel, day] <= model.y[vessel, location, day]
        model.ParcelLocation = Constraint(model.PARCELS, model.VESSELS, model.DAYS,
                                        rule=parcel_location_rule)
    
    def build_routing_constraints(self):
        """Build routing constraints from original model."""
        model = self.model
        
        # Travel time constraints (simplified)
        def travel_time_rule(model, vessel, loc1, loc2, day):
            if day < self.config["DAYS"]["end"]:
                travel_days = model.TTij[loc1, loc2]
                if travel_days > 0 and day + travel_days <= self.config["DAYS"]["end"]:
                    return (model.y[vessel, loc1, day] + model.y[vessel, loc2, day + travel_days] <= 1)
            return Constraint.Skip
        model.TravelTime = Constraint(model.VESSELS, model.LOCATIONS, model.LOCATIONS, model.DAYS,
                                    rule=travel_time_rule)
    
    def build_production_constraints(self):
        """Build production-related constraints."""
        model = self.model
        
        # Blend production constraints
        def blend_production_rule(model, blend, day):
            return model.z[blend, day] <= model.BCb[blend]
        model.BlendProduction = Constraint(model.BLENDS, model.DAYS, rule=blend_production_rule)
        
        # Blend ratio constraints
        def blend_ratio_rule(model, blend, crude, day):
            return model.w[crude, day] >= model.BRcb[blend, crude] * model.z[blend, day]
        model.BlendRatio = Constraint(model.BLENDS, model.CRUDES, model.DAYS,
                                    rule=blend_ratio_rule)
        
        # Default production capacity
        def default_capacity_rule(model, day):
            default_capacity = self.config.get("default_capacity", 96000)
            return sum(model.z[blend, day] for blend in model.BLENDS) <= default_capacity
        model.DefaultCapacity = Constraint(model.DAYS, rule=default_capacity_rule)
    
    def build_objective(self, optimization_type: str = "throughput"):
        """Build objective function."""
        model = self.model
        
        if optimization_type == "throughput":
            # Maximize total production
            def throughput_objective_rule(model):
                return sum(model.z[blend, day] for blend in model.BLENDS for day in model.DAYS)
            model.Objective = Objective(rule=throughput_objective_rule, sense=maximize)
            
        elif optimization_type == "margin":
            # Maximize margin (revenue - crude costs)
            def margin_objective_rule(model):
                revenue = sum(model.z[blend, day] for blend in model.BLENDS for day in model.DAYS) * 50  # Simplified
                crude_cost = sum(model.w[crude, day] * model.MRc[crude] 
                               for crude in model.CRUDES for day in model.DAYS)
                return revenue - crude_cost
            model.Objective = Objective(rule=margin_objective_rule, sense=maximize)
        
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    def build_model(self, optimization_type: str = "throughput"):
        """Build the complete optimization model."""
        print("🔧 Building two-tank optimization model...")
        
        # Build model components
        self.build_sets()
        self.build_parameters()
        self.build_variables()
        self.build_constraints()
        self.build_objective(optimization_type)
        
        print(f"✅ Model built successfully!")
        print(f"   Variables: {self.model.nvariables()}")
        print(f"   Constraints: {self.model.nconstraints()}")
        
        return self.model
    
    def perform_iis_analysis(self, solver_name: str = "gurobi") -> Dict[str, Any]:
        """
        Perform Irreducible Inconsistent Subsystem (IIS) analysis.
        
        Args:
            solver_name: Solver to use for IIS analysis (gurobi, cplex)
            
        Returns:
            Dictionary with IIS analysis results
        """
        print(f"\n🔍 Performing IIS Analysis with {solver_name}...")
        
        try:
            # Try Gurobi first (best IIS support)
            if solver_name.lower() == "gurobi":
                opt = SolverFactory('gurobi')
                if not opt.available():
                    print("❌ Gurobi not available, trying CPLEX...")
                    return self.perform_iis_analysis("cplex")
                
                # Solve to check feasibility
                result = opt.solve(self.model, tee=False)
                
                if result.solver.termination_condition == TerminationCondition.infeasible:
                    print("❌ Model is infeasible - computing IIS...")
                    
                    # Compute IIS
                    opt.solve(self.model, tee=False, 
                            options={'IISMethod': 1, 'ResultFile': 'model.ilp'})
                    
                    print("📋 IIS analysis completed. Check model.ilp file for details.")
                    return {
                        "status": "infeasible",
                        "iis_computed": True,
                        "iis_file": "model.ilp",
                        "message": "IIS computed successfully"
                    }
                else:
                    print("✅ Model is feasible!")
                    return {
                        "status": "feasible",
                        "objective_value": result.Problem[0].Upper_bound,
                        "message": "Model solved successfully"
                    }
            
            # Try CPLEX as backup
            elif solver_name.lower() == "cplex":
                opt = SolverFactory('cplex')
                if not opt.available():
                    print("❌ CPLEX not available, using basic feasibility check...")
                    return self.basic_feasibility_check()
                
                result = opt.solve(self.model, tee=False)
                
                if result.solver.termination_condition == TerminationCondition.infeasible:
                    print("❌ Model is infeasible - computing IIS...")
                    
                    # CPLEX IIS
                    opt.solve(self.model, tee=False, 
                            options={'write': 'model.lp',
                                   'conflict': 'conflict.clp'})
                    
                    return {
                        "status": "infeasible", 
                        "iis_computed": True,
                        "iis_file": "conflict.clp",
                        "message": "Conflict file generated"
                    }
                else:
                    print("✅ Model is feasible!")
                    return {
                        "status": "feasible",
                        "objective_value": result.Problem[0].Upper_bound,
                        "message": "Model solved successfully"
                    }
            
            else:
                print(f"❌ Unsupported solver for IIS: {solver_name}")
                return self.basic_feasibility_check()
                
        except Exception as e:
            print(f"❌ IIS analysis failed: {e}")
            return self.basic_feasibility_check()
    
    def basic_feasibility_check(self) -> Dict[str, Any]:
        """Basic feasibility check with available solvers."""
        print("\n🔍 Performing basic feasibility check...")
        
        # Try available solvers
        solvers_to_try = ['glpk', 'highs', 'scip']
        
        for solver_name in solvers_to_try:
            try:
                opt = SolverFactory(solver_name)
                if opt.available():
                    print(f"   Testing with {solver_name}...")
                    result = opt.solve(self.model, tee=False)
                    
                    if result.solver.termination_condition == TerminationCondition.optimal:
                        print(f"✅ Model is feasible with {solver_name}!")
                        return {
                            "status": "feasible",
                            "solver": solver_name,
                            "objective_value": value(self.model.Objective),
                            "message": f"Successfully solved with {solver_name}"
                        }
                    elif result.solver.termination_condition == TerminationCondition.infeasible:
                        print(f"❌ Model is infeasible with {solver_name}")
                        return {
                            "status": "infeasible",
                            "solver": solver_name,
                            "message": f"Infeasible with {solver_name} - check constraints manually"
                        }
            except Exception as e:
                print(f"   {solver_name} failed: {e}")
                continue
        
        return {
            "status": "unknown",
            "message": "No suitable solver available for feasibility check"
        }


class TwoTankOptimizationRunner:
    """Runner for two-tank optimization with IIS analysis."""
    
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
    
    def run_optimization_with_iis(self, vessel_count: int = 5, 
                                 optimization_type: str = "throughput",
                                 iis_solver: str = "gurobi") -> Dict[str, Any]:
        """
        Run optimization with IIS analysis if infeasible.
        
        Args:
            vessel_count: Number of vessels
            optimization_type: "throughput" or "margin"
            iis_solver: Solver for IIS analysis ("gurobi", "cplex")
            
        Returns:
            Results dictionary
        """
        try:
            # Load data
            (config, crudes, locations, time_of_travel, crude_availability, 
             source_locations, products_info, crude_margins, opening_inventory_dict,
             products_ratio, window_to_days) = self.load_data()
            
            print(f"\n🚀 Starting Two-Tank Optimization")
            print(f"   Vessels: {vessel_count}")
            print(f"   Optimization: {optimization_type}")
            print(f"   IIS Solver: {iis_solver}")
            
            # Create optimization model
            opt_model = TwoTankOptimizationModel(
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
            model = opt_model.build_model(optimization_type)
            
            # Perform IIS analysis
            iis_results = opt_model.perform_iis_analysis(iis_solver)
            
            return {
                "status": "completed",
                "model": opt_model,
                "iis_results": iis_results,
                "crude_assignments": opt_model.crude_tank_assignment
            }
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }


def main():
    """Main function to run two-tank optimization with IIS analysis."""
    
    print("=== Two-Tank Optimization with IIS Analysis ===")
    print("Tank 1: 600,000 barrels")
    print("Tank 2: 580,000 barrels") 
    print("================================================\n")
    
    # Create and run optimization
    runner = TwoTankOptimizationRunner()
    results = runner.run_optimization_with_iis(
        vessel_count=5,
        optimization_type="throughput", 
        iis_solver="gurobi"  # Change to "cplex" or remove if Gurobi unavailable
    )
    
    if results["status"] == "completed":
        print(f"\n=== Analysis Complete ===")
        iis_results = results["iis_results"]
        print(f"Model Status: {iis_results['status']}")
        print(f"Message: {iis_results['message']}")
        
        if iis_results["status"] == "feasible":
            print(f"Objective Value: {iis_results.get('objective_value', 'N/A')}")
        elif iis_results["status"] == "infeasible" and iis_results.get("iis_computed"):
            print(f"IIS File: {iis_results.get('iis_file', 'N/A')}")
            
        print(f"\nCrude Tank Assignments:")
        for crude, tank in results["crude_assignments"].items():
            print(f"  {crude} → {tank}")
            
    else:
        print(f"\n=== Analysis Failed ===")
        print(f"Error: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
