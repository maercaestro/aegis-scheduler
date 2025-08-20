#!/usr/bin/env python3
"""
Two-Stage Decomposition Optimizer for Refinery Scheduling
Phase 1: Basic N-Solution Approach

Stage 1: Vessel & Logistics Optimizer
Stage 2: Refinery & Blending Optimizer
"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Any
from multi_tank_data_loader import MultiTankDataLoader
from solver_manager import SolverManager
import time


class DecomposedOptimizer:
    """Main orchestrator for the two-stage optimization approach."""
    
    def __init__(self, config_path: str, vessel_count: int = 5):
        self.config_path = config_path
        self.vessel_count = vessel_count
        self.data_loader = MultiTankDataLoader("test_data")
        
        # Load config for solver manager
        with open(config_path, 'r') as f:
            import json
            self.config = json.load(f)
        
        self.solver_manager = SolverManager(self.config)
        
        # Results storage
        self.vessel_solutions = []
        self.refinery_solutions = []
        self.best_solution = None
        
    def run_decomposed_optimization(self, n_solutions: int = 5):
        """
        Run the two-stage decomposed optimization.
        
        Args:
            n_solutions: Number of near-optimal vessel schedules to generate
        """
        print("🚀 Starting Two-Stage Decomposed Optimization")
        print("=" * 55)
        
        # Load data
        print("📊 Loading data...")
        data = self.load_data()
        
        # Stage 1: Generate multiple vessel schedules
        print(f"\n🚢 Stage 1: Generating {n_solutions} vessel schedules...")
        vessel_schedules = self.generate_vessel_schedules(data, n_solutions)
        
        if not vessel_schedules:
            print("❌ No feasible vessel schedules found!")
            return None
        
        print(f"✅ Generated {len(vessel_schedules)} feasible vessel schedules")
        
        # Stage 2: Evaluate each schedule with refinery optimizer
        print(f"\n🏭 Stage 2: Evaluating refinery optimization for each schedule...")
        results = self.evaluate_refinery_schedules(data, vessel_schedules)
        
        # Select best overall solution
        best_result = self.select_best_solution(results)
        
        print(f"\n🎯 Optimization Complete!")
        print(f"Best total value: ${best_result['total_value']:,.2f}")
        print(f"Logistics cost: ${best_result['logistics_cost']:,.2f}")
        print(f"Refinery profit: ${best_result['refinery_profit']:,.2f}")
        
        return best_result
    
    def load_data(self):
        """Load and prepare all necessary data."""
        # Load base data
        data = self.data_loader.load_all_data()
        
        # Apply tank configuration
        tank_config = {
            "TANKS": {
                "Tank1": {"capacity": 250000, "allowed_crudes": []},
                "Tank2": {"capacity": 250000, "allowed_crudes": []},
                "Tank3": {"capacity": 250000, "allowed_crudes": []},
                "Tank4": {"capacity": 250000, "allowed_crudes": []},
                "Tank5": {"capacity": 180000, "allowed_crudes": []}
            }
        }
        data["config"].update(tank_config)
        
        # Assign crudes to tanks
        tank_assignments = self.data_loader.assign_crudes_to_tanks(data["crudes"], data["config"])
        
        return {
            **data,
            "tank_assignments": tank_assignments
        }
    
    def generate_vessel_schedules(self, data: Dict, n_solutions: int) -> List[Dict]:
        """
        Generate multiple near-optimal vessel schedules using different strategies.
        """
        schedules = []
        
        for i in range(n_solutions):
            print(f"  Generating schedule {i+1}/{n_solutions}...")
            
            try:
                # Create vessel optimizer with variation
                vessel_optimizer = VesselOptimizer(data, variation_seed=i, solver_manager=self.solver_manager, vessel_count=self.vessel_count)
                schedule = vessel_optimizer.solve()
                
                if schedule and schedule.get('feasible', False):
                    schedules.append(schedule)
                    print(f"    ✅ Schedule {i+1}: Cost = ${schedule['logistics_cost']:,.2f}")
                else:
                    print(f"    ❌ Schedule {i+1}: Infeasible")
                    
            except Exception as e:
                print(f"    ❌ Schedule {i+1}: Error - {e}")
        
        return schedules
    
    def evaluate_refinery_schedules(self, data: Dict, vessel_schedules: List[Dict]) -> List[Dict]:
        """
        Evaluate each vessel schedule with the refinery optimizer.
        """
        results = []
        
        for i, vessel_schedule in enumerate(vessel_schedules):
            print(f"  Evaluating refinery plan {i+1}/{len(vessel_schedules)}...")
            
            try:
                # Create refinery optimizer with fixed vessel schedule
                refinery_optimizer = RefineryOptimizer(data, vessel_schedule, solver_manager=self.solver_manager)
                refinery_result = refinery_optimizer.solve()
                
                if refinery_result and refinery_result.get('feasible', False):
                    total_value = refinery_result['refinery_profit'] - vessel_schedule['logistics_cost']
                    
                    result = {
                        'vessel_schedule': vessel_schedule,
                        'refinery_result': refinery_result,
                        'logistics_cost': vessel_schedule['logistics_cost'],
                        'refinery_profit': refinery_result['refinery_profit'],
                        'total_value': total_value,
                        'production_schedule': refinery_result.get('production_schedule', {}),
                        'crude_consumption': refinery_result.get('crude_consumption', {}),
                        'tank_inventory': refinery_result.get('tank_inventory', {}),
                        'crude_deliveries': vessel_schedule.get('crude_deliveries', {}),
                        'feasible': True
                    }
                    
                    results.append(result)
                    print(f"    ✅ Plan {i+1}: Profit = ${refinery_result['refinery_profit']:,.2f}, Total = ${total_value:,.2f}")
                else:
                    print(f"    ❌ Plan {i+1}: Refinery infeasible")
                    
            except Exception as e:
                print(f"    ❌ Plan {i+1}: Error - {e}")
        
        return results
    
    def select_best_solution(self, results: List[Dict]) -> Dict:
        """Select the solution with highest total value."""
        if not results:
            raise ValueError("No feasible solutions found!")
        
        best_result = max(results, key=lambda x: x['total_value'])
        self.best_solution = best_result
        
        return best_result


class VesselOptimizer:
    """Stage 1: Vessel & Logistics Optimizer"""
    
    def __init__(self, data: Dict, variation_seed: int = 0, solver_manager=None, vessel_count: int = 5):
        self.data = data
        self.variation_seed = variation_seed
        self.solver_manager = solver_manager
        self.vessel_count = vessel_count
        self.model = None
        
    def solve(self) -> Dict:
        """Solve the vessel scheduling optimization."""
        try:
            # Build model
            self.build_model()
            
            # Add variation for multiple solutions
            self.add_solution_variation()
            
            # Solve
            solver = self.data['config'].get('solver_name', 'glpk')
            
            print(f"    Using solver: {solver}")
            
            # Use direct solving since SolverManager has complex signature
            try:
                opt = SolverFactory(solver)
                if not opt.available():
                    print(f"    Solver {solver} not available, trying glpk...")
                    opt = SolverFactory('glpk')
                
                result = opt.solve(self.model, tee=False)
                solver_status = result.solver.termination_condition
                
                if solver_status == TerminationCondition.optimal:
                    return self.extract_vessel_solution()
                else:
                    print(f"    Solver status: {solver_status}")
                    return {'feasible': False, 'logistics_cost': float('inf')}
                    
            except Exception as solver_error:
                print(f"    Solver error: {solver_error}")
                return {'feasible': False, 'logistics_cost': float('inf')}
                
        except Exception as e:
            print(f"    Vessel optimizer error: {e}")
            return {'feasible': False, 'logistics_cost': float('inf')}
    
    def build_model(self):
        """Build the vessel scheduling MILP model."""
        self.model = ConcreteModel()
        
        # Sets
        self.model.VESSELS = Set(initialize=range(1, self.vessel_count + 1))
        self.model.DAYS = Set(initialize=range(1, 41))  # 40 days
        self.model.LOCATIONS = Set(initialize=self.data['locations'])
        self.model.CRUDES = Set(initialize=self.data['crudes'])
        
        # Parameters
        self.model.CrudeAvailability = Param(self.model.CRUDES, self.model.DAYS, 
                                           initialize=self.get_crude_availability_param())
        self.model.TravelTime = Param(self.model.LOCATIONS, self.model.LOCATIONS,
                                    initialize=self.get_travel_time_param())
        self.model.DemurrageCost = Param(initialize=1000, mutable=True)  # Cost per day per vessel
        self.model.FuelCost = Param(initialize=500, mutable=True)  # Cost per location move
        
        # Decision Variables
        self.model.VesselAtLocation = Var(self.model.VESSELS, self.model.LOCATIONS, self.model.DAYS, 
                                        domain=Binary)
        self.model.CrudePickup = Var(self.model.VESSELS, self.model.CRUDES, self.model.DAYS,
                                   domain=NonNegativeReals)
        self.model.CrudeDelivery = Var(self.model.CRUDES, self.model.DAYS,
                                     domain=NonNegativeReals)
        
        # Objective: Minimize logistics costs
        def logistics_cost_rule(model):
            demurrage = sum(model.DemurrageCost * model.VesselAtLocation[v, l, d]
                           for v in model.VESSELS for l in model.LOCATIONS for d in model.DAYS)
            
            # Simplified fuel cost (movement between locations)
            fuel = sum(model.FuelCost * model.VesselAtLocation[v, l, d]
                      for v in model.VESSELS for l in model.LOCATIONS for d in model.DAYS)
            
            return demurrage + fuel
        
        self.model.LogisticsCost = Objective(rule=logistics_cost_rule, sense=minimize)
        
        # Key Constraints
        self.add_vessel_constraints()
        
    def add_vessel_constraints(self):
        """Add essential vessel scheduling constraints."""
        
        # Vessel can only be at one location per day
        def one_location_rule(model, v, d):
            return sum(model.VesselAtLocation[v, l, d] for l in model.LOCATIONS) == 1
        self.model.OneLocationPerVessel = Constraint(self.model.VESSELS, self.model.DAYS, 
                                                   rule=one_location_rule)
        
        # Crude delivery constraint (simplified)
        def crude_delivery_rule(model, c, d):
            if d <= 5:  # Delivery delay
                return model.CrudeDelivery[c, d] == 0
            else:
                return model.CrudeDelivery[c, d] <= sum(
                    model.CrudePickup[v, c, d-5] for v in model.VESSELS)
        self.model.CrudeDeliveryRule = Constraint(self.model.CRUDES, self.model.DAYS,
                                                rule=crude_delivery_rule)
        
        # Availability constraint
        def availability_rule(model, c, d):
            return sum(model.CrudePickup[v, c, d] for v in model.VESSELS) <= model.CrudeAvailability[c, d]
        self.model.AvailabilityRule = Constraint(self.model.CRUDES, self.model.DAYS,
                                               rule=availability_rule)
    
    def add_solution_variation(self):
        """Add variation to generate different solutions."""
        if self.variation_seed > 0:
            # Add random cost perturbations
            np.random.seed(self.variation_seed)
            
            # Slightly randomize demurrage costs
            base_cost = value(self.model.DemurrageCost)
            variation = np.random.uniform(0.9, 1.1)
            self.model.DemurrageCost.set_value(base_cost * variation)
    
    def get_crude_availability_param(self):
        """Convert crude availability to Pyomo parameter format."""
        param_dict = {}
        crude_availability = self.data['crude_availability']
        
        for crude in self.data['crudes']:
            for day in range(1, 41):  # 40 days
                if crude in crude_availability and day in crude_availability[crude]:
                    param_dict[(crude, day)] = crude_availability[crude][day]
                else:
                    param_dict[(crude, day)] = 0
        
        return param_dict
    
    def get_travel_time_param(self):
        """Convert travel times to Pyomo parameter format."""
        param_dict = {}
        travel_times = self.data['time_of_travel']
        
        for loc1 in self.data['locations']:
            for loc2 in self.data['locations']:
                if loc1 == loc2:
                    param_dict[(loc1, loc2)] = 0
                elif (loc1, loc2) in travel_times:
                    param_dict[(loc1, loc2)] = travel_times[(loc1, loc2)]
                else:
                    param_dict[(loc1, loc2)] = 7  # Default travel time
        
        return param_dict
    
    def extract_vessel_solution(self):
        """Extract solution from solved model."""
        logistics_cost = value(self.model.LogisticsCost)
        
        # Extract crude delivery schedule
        crude_deliveries = {}
        for crude in self.model.CRUDES:
            crude_deliveries[crude] = {}
            for day in self.model.DAYS:
                delivery = value(self.model.CrudeDelivery[crude, day])
                if delivery > 0.1:  # Threshold for numerical precision
                    crude_deliveries[crude][day] = delivery
        
        return {
            'feasible': True,
            'logistics_cost': logistics_cost,
            'crude_deliveries': crude_deliveries,
            'variation_seed': self.variation_seed
        }


class RefineryOptimizer:
    """Stage 2: Refinery & Blending Optimizer"""
    
    def __init__(self, data: Dict, vessel_schedule: Dict, solver_manager=None):
        self.data = data
        self.vessel_schedule = vessel_schedule
        self.solver_manager = solver_manager
        self.model = None
        
    def solve(self) -> Dict:
        """Solve the refinery optimization with fixed crude deliveries."""
        try:
            # Build model
            self.build_model()
            
            # Solve
            solver = self.data['config'].get('solver_name', 'glpk')
            
            # Use direct solving since SolverManager has complex signature
            try:
                opt = SolverFactory(solver)
                if not opt.available():
                    opt = SolverFactory('glpk')
                
                result = opt.solve(self.model, tee=False)
                solver_status = result.solver.termination_condition
                
                if solver_status == TerminationCondition.optimal:
                    return self.extract_refinery_solution()
                else:
                    return {'feasible': False, 'refinery_profit': -float('inf')}
                    
            except Exception as solver_error:
                print(f"    Refinery solver error: {solver_error}")
                return {'feasible': False, 'refinery_profit': -float('inf')}
                
        except Exception as e:
            print(f"    Refinery optimizer error: {e}")
            return {'feasible': False, 'refinery_profit': -float('inf')}
    
    def build_model(self):
        """Build the refinery optimization model with fixed crude arrivals."""
        self.model = ConcreteModel()
        
        # Sets
        self.model.TANKS = Set(initialize=list(self.data['config']['TANKS'].keys()))
        self.model.CRUDES = Set(initialize=self.data['crudes'])
        self.model.DAYS = Set(initialize=range(1, 41))  # 40 days
        
        # Fixed parameters from vessel schedule
        self.model.CrudeArrivals = Param(self.model.CRUDES, self.model.DAYS,
                                       initialize=self.get_crude_arrivals_param())
        
        # Tank parameters
        self.model.TankCapacity = Param(self.model.TANKS,
                                      initialize=self.get_tank_capacity_param())
        
        # Decision Variables
        self.model.TankInventory = Var(self.model.TANKS, self.model.CRUDES, self.model.DAYS,
                                     domain=NonNegativeReals)
        self.model.CrudeConsumed = Var(self.model.CRUDES, self.model.DAYS,
                                     domain=NonNegativeReals)
        self.model.Production = Var(self.model.DAYS, domain=NonNegativeReals)
        
        # Objective: Maximize refinery profit
        def refinery_profit_rule(model):
            revenue = sum(50000 * model.Production[d] for d in model.DAYS)  # Simplified revenue
            crude_cost = sum(30000 * model.CrudeConsumed[c, d] 
                           for c in model.CRUDES for d in model.DAYS)
            holding_cost = sum(10 * model.TankInventory[t, c, d]
                             for t in model.TANKS for c in model.CRUDES for d in model.DAYS)
            
            return revenue - crude_cost - holding_cost
        
        self.model.RefineryProfit = Objective(rule=refinery_profit_rule, sense=maximize)
        
        # Add refinery constraints
        self.add_refinery_constraints()
    
    def add_refinery_constraints(self):
        """Add essential refinery constraints."""
        
        # Tank capacity constraints
        def tank_capacity_rule(model, t, d):
            return sum(model.TankInventory[t, c, d] for c in model.CRUDES) <= model.TankCapacity[t]
        self.model.TankCapacityRule = Constraint(self.model.TANKS, self.model.DAYS,
                                               rule=tank_capacity_rule)
        
        # Inventory balance with crude splitting support
        def inventory_balance_rule(model, t, c, d):
            crude_splits = self.data['config'].get('CRUDE_SPLITS', {})
            
            if d == 1:
                # Opening inventory
                if c in crude_splits:
                    opening = crude_splits[c].get(t, 0)
                elif self.data['tank_assignments'].get(c) == t:
                    opening = self.data.get('opening_inventory_dict', {}).get(c, 0)
                else:
                    opening = 0
                
                arrivals = model.CrudeArrivals[c, d] if self.data['tank_assignments'].get(c) == t else 0
                consumption = model.CrudeConsumed[c, d] if self.data['tank_assignments'].get(c) == t else 0
                
                return model.TankInventory[t, c, d] == opening + arrivals - consumption
            else:
                arrivals = model.CrudeArrivals[c, d] if self.data['tank_assignments'].get(c) == t else 0
                consumption = model.CrudeConsumed[c, d] if self.data['tank_assignments'].get(c) == t else 0
                
                return (model.TankInventory[t, c, d] == 
                       model.TankInventory[t, c, d-1] + arrivals - consumption)
        
        self.model.InventoryBalanceRule = Constraint(self.model.TANKS, self.model.CRUDES, self.model.DAYS,
                                                   rule=inventory_balance_rule)
        
        # Production capacity
        def production_capacity_rule(model, d):
            default_capacity = self.data['config'].get('default_capacity', 96000)
            return model.Production[d] <= default_capacity
        self.model.ProductionCapacityRule = Constraint(self.model.DAYS,
                                                     rule=production_capacity_rule)
        
        # Production-Consumption link: Production requires crude input
        def production_consumption_rule(model, d):
            # Assume 1 barrel of production requires 1.2 barrels of crude input (simplified)
            crude_input_ratio = 1.2
            total_crude_consumed = sum(model.CrudeConsumed[c, d] for c in model.CRUDES)
            return model.Production[d] <= total_crude_consumed / crude_input_ratio
        self.model.ProductionConsumptionRule = Constraint(self.model.DAYS,
                                                        rule=production_consumption_rule)
    
    def get_crude_arrivals_param(self):
        """Convert vessel schedule to crude arrivals parameter."""
        param_dict = {}
        crude_deliveries = self.vessel_schedule.get('crude_deliveries', {})
        
        for crude in self.data['crudes']:
            for day in range(1, 41):  # 40 days
                if crude in crude_deliveries and day in crude_deliveries[crude]:
                    param_dict[(crude, day)] = crude_deliveries[crude][day]
                else:
                    param_dict[(crude, day)] = 0
        
        return param_dict
    
    def get_tank_capacity_param(self):
        """Get tank capacity parameters."""
        return {tank: self.data['config']['TANKS'][tank]['capacity'] 
                for tank in self.data['config']['TANKS']}
    
    def extract_refinery_solution(self):
        """Extract solution from solved refinery model."""
        refinery_profit = value(self.model.RefineryProfit)
        
        # Extract production schedule
        production_schedule = {}
        for day in self.model.DAYS:
            production = value(self.model.Production[day])
            if production > 0.1:
                production_schedule[day] = production
        
        # Extract crude consumption schedule
        crude_consumption = {}
        for crude in self.model.CRUDES:
            crude_consumption[crude] = {}
            for day in self.model.DAYS:
                consumption = value(self.model.CrudeConsumed[crude, day])
                if consumption > 0.1:  # Threshold for numerical precision
                    crude_consumption[crude][day] = consumption
        
        # Extract tank inventory details
        tank_inventory = {}
        for tank in self.model.TANKS:
            tank_inventory[tank] = {}
            for crude in self.model.CRUDES:
                tank_inventory[tank][crude] = {}
                for day in self.model.DAYS:
                    inventory = value(self.model.TankInventory[tank, crude, day])
                    if inventory > 0.1:
                        tank_inventory[tank][crude][day] = inventory
        
        return {
            'feasible': True,
            'refinery_profit': refinery_profit,
            'production_schedule': production_schedule,
            'crude_consumption': crude_consumption,
            'tank_inventory': tank_inventory
        }


if __name__ == "__main__":
    # Run decomposed optimization
    optimizer = DecomposedOptimizer("test_data/config.json", vessel_count=5)
    result = optimizer.run_decomposed_optimization(n_solutions=3)
    
    if result:
        print(f"\n🎉 Decomposed optimization successful!")
        print(f"Selected vessel schedule variation: {result['vessel_schedule']['variation_seed']}")
    else:
        print(f"\n❌ Decomposed optimization failed!")
