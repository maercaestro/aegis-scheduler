# -*- coding: utf-8 -*-
"""
Refinery Scheduling Optimization Script (CPLEX Version)

This script consolidates the vessel routing and crude blending optimization model.
It is designed to determine the optimal schedule for vessels and refinery operations
to maximize either profit or throughput, based on user-defined parameters.

Key functionalities include:
1. Loading scenario-specific data from CSV and JSON files.
2. Building a Mixed-Integer Linear Programming (MILP) model using Pyomo.
3. Defining sets, parameters, decision variables, and constraints.
4. Selecting an objective function (margin or throughput).
5. Solving the optimization problem using the CPLEX solver.
6. Processing and saving the results into CSV files.

Requirements:
- Python 3.10 (for CPLEX compatibility)
- CPLEX solver installed and licensed
- Pyomo optimization framework
- Pandas for data handling

To run with CPLEX environment:
    source cplex/bin/activate
    python cplex_refinery_optimizer.py
"""

# 1. Import Necessary Libraries
import os
import sys
import json
import ast
import pickle
from contextlib import redirect_stdout
from datetime import datetime

import pandas as pd
from pyomo.environ import *

# =============================================================================
# 2. Main Optimization Function
# =============================================================================

def run_refinery_optimization(scenario_num, vessel_count, optimization_type, max_demurrage_limit=10):
    """
    Main function to run the refinery scheduling optimization.

    Args:
        scenario_num (int): The scenario number to load data for.
        vessel_count (int): The number of vessels available for scheduling.
        optimization_type (str): The optimization objective ('margin' or 'throughput').
        max_demurrage_limit (int, optional): The maximum number of demurrage days
                                             allowed when optimizing for throughput.
                                             Defaults to 10.
    """
    print("--- Starting Refinery Scheduling Optimization ---")
    print(f"Scenario: {scenario_num}, Vessels: {vessel_count}, Objective: {optimization_type}")

    # =========================================================================
    # 3. Data Loading and Preprocessing
    # =========================================================================
    print("\n--- Loading and Preprocessing Data ---")

    def load_all_scenario_data(scenario):
        # Use test_data directory for local execution
        base_path = f"../test_data/"
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Test data directory not found: {base_path}")

        with open(os.path.join(base_path, "config.json"), "r") as f:
            config = json.load(f)

        crude_availability_df = pd.read_csv(os.path.join(base_path, "crude_availability.csv"))
        crude_availability = {}
        for _, row in crude_availability_df.iterrows():
            crude_availability.setdefault(row["date_range"], {}).setdefault(row["location"], {})[row["crude"]] = {
                "volume": int(row["volume"]),
                "parcel_size": int(row["parcel_size"])
            }

        time_of_travel_df = pd.read_csv(os.path.join(base_path, "time_of_travel.csv"))
        time_of_travel = {
            (row["from"], row["to"]): int(row["time_in_days"]) + 1
            for _, row in time_of_travel_df.iterrows()
        }

        products_info_df = pd.read_csv(os.path.join(base_path, "products_info.csv"))
        crudes_info_df = pd.read_csv(os.path.join(base_path, "crudes_info.csv"))
        crudes = crudes_info_df["crudes"]
        locations = set(time_of_travel_df["from"]) | set(time_of_travel_df["to"])
        source_location = crudes_info_df["origin"].unique().tolist()
        crude_margins = crudes_info_df['margin'].tolist()
        opening_inventory = crudes_info_df['opening_inventory'].tolist()
        opening_inventory_dict = dict(zip(crudes.tolist(), opening_inventory))

        return config, list(crudes), list(locations), time_of_travel, crude_availability, source_location, products_info_df, crude_margins, opening_inventory_dict

    def extract_window_to_days(crude_availability):
        window_to_days = {}
        for window in crude_availability:
            parts = window.split()[0]
            if '-' in parts:
                start_day, end_day = map(int, parts.split('-'))
                days = list(range(start_day, end_day + 1))
            else:
                days = [int(parts)]
            window_to_days[window] = days
        return window_to_days

    def extract_products_ratio(df):
        return {
            (row['product'], crude): ratio
            for _, row in df.iterrows()
            for crude, ratio in zip(ast.literal_eval(row['crudes']), ast.literal_eval(row['ratios']))
        }

    config, crudes, locations, time_of_travel, crude_availability, source_locations, products_info, crude_margins, opening_inventory_dict = load_all_scenario_data(scenario_num)
    window_to_days = extract_window_to_days(crude_availability)
    products_ratio = extract_products_ratio(products_info)

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12,
    }
    month_number = month_map[config["schedule_month"]]
    year = config["schedule_year"]
    start_date = pd.to_datetime(f"{year}-{month_number:02d}-01")
    print("Data loaded successfully.")

    # =========================================================================
    # 5. Pyomo Model Initialization
    # =========================================================================
    print("\n--- Initializing Pyomo Optimization Model ---")
    model = ConcreteModel()

    # =========================================================================
    # 6. Defining Sets
    # =========================================================================
    model.CRUDES = Set(initialize=crudes)
    model.LOCATIONS = Set(initialize=locations)
    model.SOURCE_LOCATIONS = Set(initialize=source_locations)
    config["VESSELS"] = list(range(1, vessel_count + 1))
    model.VESSELS = Set(initialize=config["VESSELS"])
    model.DAYS = RangeSet(config["DAYS"]["start"], config["DAYS"]["end"])
    model.BLENDS = Set(initialize=products_info['product'].tolist())
    model.SLOTS = RangeSet(config["DAYS"]["start"], 2 * config["DAYS"]["end"])

    parcel_set = set()
    for window, loc_data in crude_availability.items():
        for location, crude_dict in loc_data.items():
            for crude_type, info in crude_dict.items():
                parcel_set.add((location, crude_type, window))
    model.PARCELS = Set(initialize=parcel_set, dimen=3)
    print("Sets defined.")

    # =========================================================================
    # 7. Defining Parameters
    # =========================================================================
    products_capacity = dict(zip(products_info['product'].tolist(), products_info['max_per_day']))
    crude_margins_dict = dict(zip(crudes, crude_margins))
    model.BCb = Param(model.BLENDS, initialize=products_capacity)
    model.BRcb = Param(model.BLENDS, model.CRUDES, initialize=products_ratio, default=0)
    model.MRc = Param(model.CRUDES, initialize=crude_margins_dict)

    parcel_size_dict = {
        (loc, c_type, win): info["parcel_size"]
        for win, loc_data in crude_availability.items()
        for loc, c_dict in loc_data.items()
        for c_type, info in c_dict.items()
    }
    model.PVp = Param(model.PARCELS, initialize=parcel_size_dict)
    
    # Other parameters
    model.PCp = Param(model.PARCELS, within=Any, initialize=lambda m, *p: p[1]) # Crude type of a parcel
    model.PLp = Param(model.PARCELS, within=Any, initialize=lambda m, *p: p[0]) # Location of a parcel
    model.PDp = Param(model.PARCELS, within=Any, initialize=lambda m, *p: window_to_days[p[2]]) # Available days of a parcel
    model.Travel_Time = Param(model.LOCATIONS, model.LOCATIONS, initialize=time_of_travel, default=999)

    # Refinery capacity parameter
    days_list = list(range(config["DAYS"]["start"], config["DAYS"]["end"] + 1))
    capacity_dict = {day: config['default_capacity'] for day in days_list}
    for entry in config.get('plant_capacity_reduction_window', []):
        for day in range(entry['start_date'], entry['end_date'] + 1):
            if day in capacity_dict:
                capacity_dict[day] = entry['max_capacity']
    model.RCd = Param(model.DAYS, initialize=capacity_dict)
    print("Parameters defined.")

    # =========================================================================
    # 8. Defining Decision Variables
    # =========================================================================
    # Core variables
    model.AtLocation = Var(model.VESSELS, model.LOCATIONS, model.DAYS, domain=Binary)
    model.Pickup = Var(model.VESSELS, model.PARCELS, model.DAYS, domain=Binary)
    model.Discharge = Var(model.VESSELS, model.DAYS, domain=Binary)
    model.Inventory = Var(model.CRUDES, model.DAYS, domain=NonNegativeReals)
    model.BlendFraction = Var(model.BLENDS, model.SLOTS, domain=NonNegativeReals, bounds=(0,1))
    
    # Auxiliary variables
    model.Departure = Var(model.VESSELS, model.LOCATIONS, model.DAYS, domain=Binary)
    model.CrudeInVessel = Var(model.VESSELS, model.CRUDES, domain=Binary)
    model.NumGrades12 = Var(model.VESSELS, domain=Binary) # Vessel carries 1 or 2 crude grades
    model.NumGrades3 = Var(model.VESSELS, domain=Binary) # Vessel carries 3 crude grades
    model.VolumeOnboard = Var(model.VESSELS, model.CRUDES, domain=NonNegativeReals)
    model.VolumeDischarged = Var(model.VESSELS, model.CRUDES, model.DAYS, domain=NonNegativeReals)
    model.IsBlendConsumed = Var(model.BLENDS, model.SLOTS, domain=Binary)
    model.IsTransition = Var(model.BLENDS, model.SLOTS, domain=Binary)
    model.DischargeDay = Var(model.VESSELS, domain=PositiveIntegers)
    model.Ullage = Var(model.DAYS, domain=NonNegativeReals)
    print("Decision variables defined.")

    # =========================================================================
    # 9. Defining Constraints
    # =========================================================================
    print("\n--- Defining Model Constraints ---")
    
    # --- 9.1 Vessel Travel Constraints ---
    def vessel_single_location_rule(model, v, d):
        return sum(model.AtLocation[v, l, d] for l in model.LOCATIONS) <= 1
    model.VesselSingleLocation = Constraint(model.VESSELS, model.DAYS, rule=vessel_single_location_rule)

    def departure_logic_rule(model, v, l, d):
        if d < config["DAYS"]["end"]:
            # This set of constraints defines Departure[v,l,d] = 1 if AtLocation[v,l,d]=1 and AtLocation[v,l,d+1]=0
            yield model.Departure[v, l, d] >= model.AtLocation[v, l, d] - model.AtLocation[v, l, d + 1]
            yield model.Departure[v, l, d] <= model.AtLocation[v, l, d]
            yield model.Departure[v, l, d] <= 1 - model.AtLocation[v, l, d + 1]
        else:
            yield model.Departure[v, l, d] == 0
    model.DepartureLogic = ConstraintList()
    for v in model.VESSELS:
        for l in model.LOCATIONS:
            for d in model.DAYS:
                for c in departure_logic_rule(model, v, l, d):
                    model.DepartureLogic.add(c)
    
    def single_departure_per_location_rule(model, v, l):
        return sum(model.Departure[v, l, d] for d in model.DAYS) <= 1
    model.SingleDeparturePerLocation = Constraint(model.VESSELS, model.LOCATIONS, rule=single_departure_per_location_rule)

    def enforce_travel_time_rule(model, v, l, d):
        if l in model.SOURCE_LOCATIONS:
            # If a vessel departs, it must arrive at a destination after the specified travel time.
            destinations = [
                model.AtLocation[v, l2, d + model.Travel_Time[l, l2]]
                for l2 in model.LOCATIONS
                if (l, l2) in model.Travel_Time and d + model.Travel_Time[l, l2] in model.DAYS
            ]
            if destinations:
                return model.Departure[v, l, d] <= sum(destinations)
        return Constraint.Skip
    model.EnforceTravelTime = Constraint(model.VESSELS, model.LOCATIONS, model.DAYS, rule=enforce_travel_time_rule)

    def no_early_arrival_rule(model, v, l1, l2, d1, d2):
        if l1 != l2 and (l1, l2) in model.Travel_Time:
            travel_time = model.Travel_Time[l1, l2]
            if d2 > d1 and d2 < d1 + travel_time:
                return model.AtLocation[v, l1, d1] + model.AtLocation[v, l2, d2] <= 1
        return Constraint.Skip
    model.NoEarlyArrival = Constraint(model.VESSELS, model.LOCATIONS, model.LOCATIONS, model.DAYS, model.DAYS, rule=no_early_arrival_rule)
    
    # --- 9.2 Vessel Loading Constraints ---
    def one_ship_for_one_parcel_pickup_rule(model, *p):
        return sum(model.Pickup[v, p, d] for v in model.VESSELS for d in model.DAYS) <= 1
    model.OneVesselParcel = Constraint(model.PARCELS, rule=one_ship_for_one_parcel_pickup_rule)

    def one_pickup_per_day_rule(model, v, d):
        return sum(model.Pickup[v, p, d] for p in model.PARCELS) <= 1
    model.OnePickupDayVessel = Constraint(model.VESSELS, model.DAYS, rule=one_pickup_per_day_rule)
    
    def pickup_day_limit_rule(model, v, *p):
        return sum(model.Pickup[v, p, d] for d in model.DAYS if d not in model.PDp[p]) == 0
    model.PickupDayLimit = Constraint(model.VESSELS, model.PARCELS, rule=pickup_day_limit_rule)

    def parcel_location_bound_rule(model, v, d, *p):
        return model.Pickup[v, p, d] <= model.AtLocation[v, model.PLp[p], d]
    model.ParcelLocationBound = Constraint(model.VESSELS, model.DAYS, model.PARCELS, rule=parcel_location_bound_rule)

    def crude_in_vessel_logic_rule(model, v, c):
        # Links Pickup decision to the auxiliary CrudeInVessel variable
        M = len(model.PARCELS)
        pickups_for_crude = sum(model.Pickup[v, p, d] for p in model.PARCELS if model.PCp[p] == c for d in model.DAYS)
        return pickups_for_crude <= M * model.CrudeInVessel[v, c]
    model.CrudeInVesselLogic = Constraint(model.VESSELS, model.CRUDES, rule=crude_in_vessel_logic_rule)
    
    def min_crude_in_vessel_rule(model, v, c):
        pickups_for_crude = sum(model.Pickup[v, p, d] for p in model.PARCELS if model.PCp[p] == c for d in model.DAYS)
        return pickups_for_crude >= model.CrudeInVessel[v, c]
    model.MinCrudeInVessel = Constraint(model.VESSELS, model.CRUDES, rule=min_crude_in_vessel_rule)

    def max_3_crudes_limit_rule(model, v):
        return sum(model.CrudeInVessel[v, c] for c in model.CRUDES) <= 3
    model.Max3CrudesLimit = Constraint(model.VESSELS, rule=max_3_crudes_limit_rule)
    
    # Vessel capacity constraints based on number of crude grades
    def crude_grade_group_rule(model, v):
        return model.NumGrades12[v] + model.NumGrades3[v] <= 1
    model.CrudeGradeGroup = Constraint(model.VESSELS, rule=crude_grade_group_rule)

    def link_grades_to_count_rule(model, v):
        num_crudes = sum(model.CrudeInVessel[v, c] for c in model.CRUDES)
        yield num_crudes <= 2 + model.NumGrades3[v] # If NumGrades3=0, num_crudes<=2. If NumGrades3=1, num_crudes<=3
        yield num_crudes >= 3 * model.NumGrades3[v] # If NumGrades3=1, num_crudes>=3
        yield num_crudes >= model.NumGrades12[v]
    model.LinkGradesToCount = ConstraintList()
    for v in model.VESSELS:
        for c in link_grades_to_count_rule(model,v):
            model.LinkGradesToCount.add(c)

    def vessel_volume_limit_rule(model, v):
        total_volume = sum(model.PVp[p] * model.Pickup[v, p, d] for p in model.PARCELS for d in model.DAYS)
        capacity = config['Two_crude'] * model.NumGrades12[v] + \
                   config['Three_crude'] * model.NumGrades3[v] + \
                   config['Two_crude'] * (1-model.NumGrades12[v]-model.NumGrades3[v]) # handles 0 or 1 parcel case
        return total_volume <= capacity
    model.VesselVolumeLimit = Constraint(model.VESSELS, rule=vessel_volume_limit_rule)
    
    # --- 9.3 Vessel Discharge Constraints ---
    def unique_vessel_discharge_day_rule(model, v):
        return sum(model.Discharge[v, d] for d in model.DAYS) == 1
    model.UniqueVesselDischargeDay = Constraint(model.VESSELS, rule=unique_vessel_discharge_day_rule)
    
    def discharge_at_melaka_rule(model, v, d):
        # Discharge takes 2 days and must happen at Melaka
        if d < config["DAYS"]["end"]:
            return 2 * model.Discharge[v, d] <= model.AtLocation[v, "Melaka", d] + model.AtLocation[v, "Melaka", d + 1]
        return model.Discharge[v, d] <= model.AtLocation[v, "Melaka", d]
    model.DischargeAtMelaka = Constraint(model.VESSELS, model.DAYS, rule=discharge_at_melaka_rule)
    
    def no_concurrent_discharge_rule(model, d):
        if d < config["DAYS"]["end"]:
            return sum(model.Discharge[v, d] + model.Discharge[v, d + 1] for v in model.VESSELS) <= 1
        return sum(model.Discharge[v, d] for v in model.VESSELS) <= 1
    model.NoConcurrentDischarge = Constraint(model.DAYS, rule=no_concurrent_discharge_rule)
    
    def volume_onboard_rule(model, v, c):
        return model.VolumeOnboard[v, c] == sum(model.PVp[p] * model.Pickup[v, p, d] for p in model.PARCELS if model.PCp[p] == c for d in model.DAYS)
    model.VolumeOnboardDef = Constraint(model.VESSELS, model.CRUDES, rule=volume_onboard_rule)

    def volume_discharged_logic_rule(model, v, c, d):
        # Links VolumeDischarged to Discharge decision using big-M method
        M = config['Vessel_max_limit']
        yield model.VolumeDischarged[v, c, d] <= M * model.Discharge[v, d]
        yield model.VolumeDischarged[v, c, d] <= model.VolumeOnboard[v, c]
        yield model.VolumeDischarged[v, c, d] >= model.VolumeOnboard[v, c] - M * (1 - model.Discharge[v, d])
    model.VolumeDischargedLogic = ConstraintList()
    for v in model.VESSELS:
      for c in model.CRUDES:
        for d in model.DAYS:
          for constr in volume_discharged_logic_rule(model, v, c, d):
            model.VolumeDischargedLogic.add(constr)

    # --- 9.4 Vessel Ordering (Symmetry Breaking) ---
    def discharge_day_calc_rule(model, v):
        return model.DischargeDay[v] == sum(d * model.Discharge[v, d] for d in model.DAYS)
    model.CalcDischargeDay = Constraint(model.VESSELS, rule=discharge_day_calc_rule)

    def symmetry_breaking_rule(model, v):
        if v < vessel_count:
            return model.DischargeDay[v] + 1 <= model.DischargeDay[v + 1]
        return Constraint.Skip
    model.SymmetryBreak = Constraint(model.VESSELS, rule=symmetry_breaking_rule)
    
    # --- 9.5 Crude Blending Constraints ---
    def one_blend_per_slot_rule(model, s):
        # IsBlendConsumed is 1 if BlendFraction > 0
        for b in model.BLENDS:
            model.OneBlendPerSlot.add(model.IsBlendConsumed[b, s] >= model.BlendFraction[b, s])
        return sum(model.IsBlendConsumed[b, s] for b in model.BLENDS) <= 1
    model.OneBlendPerSlot = ConstraintList()
    for s in model.SLOTS:
       model.OneBlendPerSlot.add(one_blend_per_slot_rule(model, s))

    def blend_fraction_daily_sum_rule(model, d):
        # The sum of fractions over two slots in a day must be <= 1
        s1, s2 = 2*d - 1, 2*d
        if s2 in model.SLOTS:
            return sum(model.BlendFraction[b, s1] + model.BlendFraction[b, s2] for b in model.BLENDS) <= 1
        return Constraint.Skip
    model.BlendFractionDailySum = Constraint(model.DAYS, rule=blend_fraction_daily_sum_rule)

    def plant_capacity_rule(model, d):
        s1, s2 = 2*d - 1, 2*d
        if s2 in model.SLOTS:
            daily_production = sum(model.BCb[b] * (model.BlendFraction[b, s1] + model.BlendFraction[b, s2]) for b in model.BLENDS)
            return daily_production <= model.RCd[d]
        return Constraint.Skip
    model.PlantCapacityConstraint = Constraint(model.DAYS, rule=plant_capacity_rule)
    
    def min_plant_capacity_rule(model,d):
        s1, s2 = 2*d - 1, 2*d
        if s2 in model.SLOTS:
            daily_production = sum(model.BCb[b] * (model.BlendFraction[b, s1] + model.BlendFraction[b, s2]) for b in model.BLENDS)
            return daily_production >= config.get('turn_down_capacity', 0)
        return Constraint.Skip
    model.MinPlantCapacity = Constraint(model.DAYS, rule=min_plant_capacity_rule)

    # --- 9.5b Transition Constraints ---
    def transition_logic_rule(model, b, s):
        # IsTransition[b,s] = 1 if IsBlendConsumed[b,s]=1 and IsBlendConsumed[b,s+1]=0 (ending a blend)
        if s + 1 in model.SLOTS:
            # Transition lower bound: IsTransition >= IsBlendConsumed[b,s] - IsBlendConsumed[b,s+1]
            yield model.IsTransition[b, s] >= model.IsBlendConsumed[b, s] - model.IsBlendConsumed[b, s + 1]
            # Transition upper bound 2: IsTransition <= 1 - IsBlendConsumed[b,s+1]
            yield model.IsTransition[b, s] <= 1 - model.IsBlendConsumed[b, s + 1]
        
        # Transition upper bound 1: IsTransition <= IsBlendConsumed[b,s]
        yield model.IsTransition[b, s] <= model.IsBlendConsumed[b, s]
    
    model.TransitionLogic = ConstraintList()
    for b in model.BLENDS:
        for s in model.SLOTS:
            for constr in transition_logic_rule(model, b, s):
                model.TransitionLogic.add(constr)
    
    def max_transitions_rule(model):
        return sum(model.IsTransition[b, s] for b in model.BLENDS for s in model.SLOTS) <= config["MaxTransitions"]
    model.MaxTransitionsConstraint = Constraint(rule=max_transitions_rule)


    # --- 9.6 Inventory Constraints ---
    def inventory_update_rule(model, c, d):
        # Inventory balance equation
        discharged_today = 0
        # Discharge takes 5 days to reach inventory
        if d > 5:
            discharged_today = sum(model.VolumeDischarged[v, c, d-5] for v in model.VESSELS)
        
        s1, s2 = 2*d - 1, 2*d
        consumed_today = 0
        if s2 in model.SLOTS:
            consumed_today = sum(model.BCb[b] * model.BRcb[b, c] * (model.BlendFraction[b, s1] + model.BlendFraction[b, s2]) for b in model.BLENDS)
        
        if d == 1:
            return model.Inventory[c, d] == opening_inventory_dict[c] + discharged_today - consumed_today
        else:
            return model.Inventory[c, d] == model.Inventory[c, d-1] + discharged_today - consumed_today
    model.InventoryUpdate = Constraint(model.CRUDES, model.DAYS, rule=inventory_update_rule)

    def max_inventory_limit_rule(model, d):
        return sum(model.Inventory[c, d] for c in model.CRUDES) <= config["INVENTORY_MAX_VOLUME"]
    model.MaxInventoryLimit = Constraint(model.DAYS, rule=max_inventory_limit_rule)
    
    print("Constraints defined.")

    # =========================================================================
    # 10. Defining Objective Function
    # =========================================================================
    print("\n--- Defining Objective Function ---")
    
    # --- Expressions for Objective components ---
    def demurrage_at_source_expr(model):
        # Demurrage is charged for days at a source location without loading
        return config['Demurrage'] * (
            sum(model.AtLocation[v, l, d] for v in model.VESSELS for l in model.SOURCE_LOCATIONS for d in model.DAYS) - 
            sum(model.Pickup[v, p, d] for v in model.VESSELS for p in model.PARCELS for d in model.DAYS)
        )
    model.DemurrageAtSource = Expression(rule=demurrage_at_source_expr)

    def demurrage_at_melaka_expr(model):
        # Demurrage at Melaka is charged for days beyond the 2-day discharge period
        return config['Demurrage'] * sum(
            sum(model.AtLocation[v, 'Melaka', d] for d in model.DAYS) - 2 * sum(model.Discharge[v, d] for d in model.DAYS)
            for v in model.VESSELS
        )
    model.DemurrageAtMelaka = Expression(rule=demurrage_at_melaka_expr)

    def total_profit_expr(model):
        return sum(
            model.MRc[c] * model.BRcb[b, c] * model.BCb[b] * model.BlendFraction[b, s]
            for c in model.CRUDES for b in model.BLENDS for s in model.SLOTS
        )
    model.TotalProfit = Expression(rule=total_profit_expr)

    def total_throughput_expr(model):
        return sum(model.BCb[b] * model.BlendFraction[b, s] for b in model.BLENDS for s in model.SLOTS)
    model.Throughput = Expression(rule=total_throughput_expr)

    # --- Set Objective based on input ---
    if optimization_type == 'margin':
        def net_profit_objective_rule(model):
            return model.TotalProfit - (model.DemurrageAtSource + model.DemurrageAtMelaka)
        model.objective = Objective(rule=net_profit_objective_rule, sense=maximize)
        print("Objective set to: Maximize Margin")
        
    elif optimization_type == 'throughput':
        model.DemurrageLimitConstraint = Constraint(
            expr=model.DemurrageAtSource + model.DemurrageAtMelaka <= max_demurrage_limit * config["Demurrage"]
        )
        model.objective = Objective(rule=total_throughput_expr, sense=maximize)
        print("Objective set to: Maximize Throughput")
        
    else:
        raise ValueError("Invalid optimization_type. Choose 'margin' or 'throughput'.")
        
    # =========================================================================
    # 11. Solving the Model with CPLEX
    # =========================================================================
    print("\n--- Configuring and Running Solver ---")
    
    # Configure solver to CPLEX Direct (Python API)
    solver = SolverFactory('cplex_direct')
    
    # Check if CPLEX is available
    if not solver.available():
        print("CPLEX Direct solver is not available. Trying standard CPLEX...")
        solver = SolverFactory('cplex')
        if not solver.available():
            raise RuntimeError("CPLEX solver is not available. Please check your CPLEX installation and license.")
    
    # Set solver options for better performance
    solver.options['timelimit'] = 14400  # Time limit in seconds
    solver.options['mipgap'] = 0.05  # Relative MIP gap tolerance
    solver.options['threads'] = 0  # Use all available threads
    
    print(f"Solver: CPLEX Direct (Python API)")
    print(f"CPLEX available: {solver.available()}")
    print(f"Key solver options: Time limit={solver.options.get('timelimit', 'default')}, MIP gap={solver.options.get('mipgap', 'default')}")

    # Define output paths
    base_output_path = f"./results/"
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
        
    if optimization_type == 'throughput':
        file_suffix = f'{optimization_type}_{vessel_count}vessels_{config["DAYS"]["end"]}days_{max_demurrage_limit}demurrage'
    else:
        file_suffix = f'{optimization_type}_{vessel_count}vessels_{config["DAYS"]["end"]}days'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(base_output_path, f'optimization_log_{file_suffix}_{timestamp}.txt')

    # Solve the model
    try:
        print(f"\nSolving... Log file will be saved to: {log_file_path}")
        with open(log_file_path, "w") as f:
            with redirect_stdout(f):
                results = solver.solve(model, tee=True)
        print("Solver finished.")
        
        # Check solver status
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            print("Solution is optimal.")
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print("Error: No feasible solution found.")
            return
        else:
            print(f"Solver Status: {results.solver.status}")

    except Exception as e:
        if "infeasible" in str(e).lower():
            print("Error: No feasible solution exists for the given configuration.")
            return
        print(f"An error occurred during solving: {e}")
        return

    # =========================================================================
    # 12. Post-Processing and Saving Results
    # =========================================================================
    print("\n--- Processing and Saving Results ---")
    
    # --- Create Crude Blending Schedule DataFrame ---
    records = []
    for d in model.DAYS:
        s1, s2 = 2*d - 1, 2*d
        
        # Process slot 1
        for b in model.BLENDS:
            if value(model.BlendFraction[b, s1]) > 1e-6:
                quantity = value(model.BCb[b] * model.BlendFraction[b, s1])
                profit = sum(value(model.MRc[c] * model.BRcb[b, c] * model.BCb[b] * model.BlendFraction[b, s1]) for c in model.CRUDES)
                record = {
                    "Date": start_date + pd.Timedelta(days=d - 1),
                    "Slot": 1,
                    "Final Product": b,
                    "Quantity Produced (kb)": quantity / 1000,
                    "Profit": profit,
                    **{f"Crude {c} Blended (kb)": value(model.BRcb[b, c] * quantity) / 1000 for c in model.CRUDES},
                }
                records.append(record)

        # Process slot 2
        if s2 in model.SLOTS:
            for b in model.BLENDS:
                 if value(model.BlendFraction[b, s2]) > 1e-6:
                    quantity = value(model.BCb[b] * model.BlendFraction[b, s2])
                    profit = sum(value(model.MRc[c] * model.BRcb[b, c] * model.BCb[b] * model.BlendFraction[b, s2]) for c in model.CRUDES)
                    record = {
                        "Date": start_date + pd.Timedelta(days=d - 1),
                        "Slot": 2,
                        "Final Product": b,
                        "Quantity Produced (kb)": quantity / 1000,
                        "Profit": profit,
                        **{f"Crude {c} Blended (kb)": value(model.BRcb[b, c] * quantity) / 1000 for c in model.CRUDES},
                    }
                    records.append(record)

    blending_df = pd.DataFrame(records)
    
    # Add inventory and ullage info
    inv_data = []
    for d in model.DAYS:
        total_inv = sum(value(model.Inventory[c, d]) for c in model.CRUDES)
        inv_data.append({
            "Date": start_date + pd.Timedelta(days=d-1),
            **{f"Inventory {c} (kb)": value(model.Inventory[c,d])/1000 for c in model.CRUDES},
            "Total Inventory (kb)": total_inv/1000,
            "Ullage (kb)": (config["INVENTORY_MAX_VOLUME"] - total_inv)/1000
        })
    inventory_df = pd.DataFrame(inv_data)
    
    # Merge blending and inventory data
    if not blending_df.empty:
        crude_blending_df = pd.merge(blending_df, inventory_df, on="Date", how="left")
    else: # Handle case with no production
        crude_blending_df = inventory_df
        crude_blending_df['Quantity Produced (kb)'] = 0
        crude_blending_df['Profit'] = 0

    # --- Create Vessel Routing Schedule DataFrame ---
    vessel_records = []
    for v in model.VESSELS:
        for d in model.DAYS:
            # Check for Loading
            for p in model.PARCELS:
                if value(model.Pickup[v, p, d]) > 0.5:
                    vessel_records.append({
                        "Date": start_date + pd.Timedelta(days=d - 1),
                        "Vessel ID": v,
                        "Activity": "Loading",
                        "Location": model.PLp[p],
                        "Crude Type": model.PCp[p],
                        "Volume (kb)": model.PVp[p] / 1000,
                    })
            
            # Check for Discharge
            if value(model.Discharge[v, d]) > 0.5:
                 vessel_records.append({
                    "Date": start_date + pd.Timedelta(days=d - 1),
                    "Vessel ID": v, "Activity": "Discharge Start", "Location": "Melaka"
                })
                 if d+1 in model.DAYS:
                    vessel_records.append({
                        "Date": start_date + pd.Timedelta(days=d),
                        "Vessel ID": v, "Activity": "Discharging", "Location": "Melaka"
                    })

            # Check for Demurrage (at location but not loading/discharging)
            is_active = any(rec['Vessel ID'] == v and rec['Date'].date() == (start_date + pd.Timedelta(days=d - 1)).date() for rec in vessel_records)
            if not is_active:
                for l in model.LOCATIONS:
                    if value(model.AtLocation[v, l, d]) > 0.5:
                         vessel_records.append({
                            "Date": start_date + pd.Timedelta(days=d - 1),
                            "Vessel ID": v, "Activity": "Demurrage", "Location": l
                        })
    
    vessel_df = pd.DataFrame(vessel_records).sort_values(by=["Vessel ID", "Date"]).reset_index(drop=True)
    
    # --- Save files ---
    crude_blending_filename = os.path.join(base_output_path, f'crude_blending_{file_suffix}_{timestamp}.csv')
    vessel_routing_filename = os.path.join(base_output_path, f'vessel_routing_{file_suffix}_{timestamp}.csv')
    
    crude_blending_df.to_csv(crude_blending_filename, index=False)
    vessel_df.to_csv(vessel_routing_filename, index=False)
    
    # Note: Model pickling skipped due to lambda functions in parameters
    print(f"Crude blending schedule saved to: {crude_blending_filename}")
    print(f"Vessel routing schedule saved to: {vessel_routing_filename}")

    # =========================================================================
    # 13. Logging Final Metrics
    # =========================================================================
    total_throughput = crude_blending_df['Quantity Produced (kb)'].sum()
    total_margin = crude_blending_df['Profit'].sum()
    demurrage_source = value(model.DemurrageAtSource)
    demurrage_melaka = value(model.DemurrageAtMelaka)
    net_profit = total_margin - (demurrage_source + demurrage_melaka)
    
    print("\n--- Optimization Summary ---")
    print(f"Total Throughput: {total_throughput:.2f} kb")
    print(f"Total Margin: ${total_margin:,.2f}")
    print(f"Demurrage Cost (Source): ${demurrage_source:,.2f}")
    print(f"Demurrage Cost (Melaka): ${demurrage_melaka:,.2f}")
    print(f"Net Profit: ${net_profit:,.2f}")
    print(f"Crude blending output saved to: {crude_blending_filename}")
    print(f"Vessel routing output saved to: {vessel_routing_filename}")

    print("\n--- Optimization Run Finished Successfully ---")


# =============================================================================
# 14. Script Execution
# =============================================================================
if __name__ == "__main__":
    # --- Define your pipeline parameters here ---
    # Update these values according to your test scenario requirements
    
    SCENARIO_ID = 1  # Not used in current implementation, but kept for future use
    VESSEL_COUNT = 4  # Number of vessels available for scheduling
    OPTIMIZATION_GOAL = "throughput"  # Options: "margin" or "throughput"
    MAX_DEMURRAGE_DAYS_LIMIT = 20  # Only used if OPTIMIZATION_GOAL is "throughput"

    print("Starting CPLEX-based Refinery Optimization")
    print(f"Configuration: {VESSEL_COUNT} vessels, optimizing for {OPTIMIZATION_GOAL}")
    
    # --- Run the optimization ---
    run_refinery_optimization(
        scenario_num=SCENARIO_ID,
        vessel_count=VESSEL_COUNT,
        optimization_type=OPTIMIZATION_GOAL,
        max_demurrage_limit=MAX_DEMURRAGE_DAYS_LIMIT
    )