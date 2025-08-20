"""
Multi-tank optimization model builder for refinery and vessel scheduling.
Implements the Pyomo optimization model with tank-specific constraints.
"""

from pyomo.environ import *
import pandas as pd
from tqdm import tqdm


class MultiTankOptimizationModel:
    """Builds and manages the Pyomo optimization model for multi-tank system."""
    
    def __init__(self, config: dict, crudes: list, locations: list, 
                 time_of_travel: dict, crude_availability: dict,
                 source_locations: list, products_info: pd.DataFrame,
                 crude_margins: list, opening_inventory_dict: dict,
                 products_ratio: dict, window_to_days: dict, 
                 tank_assignments: dict, vessel_count: int):
        """
        Initialize the multi-tank optimization model with data.
        """
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
        self.tank_assignments = tank_assignments
        self.vessel_count = vessel_count
        
        # Constants from config
        self.INVENTORY_MAX_VOLUME = config["INVENTORY_MAX_VOLUME"]
        self.MaxTransitions = config["MaxTransitions"]
        self.tanks = list(config["TANKS"].keys())
        self.tank_capacities = {tank: config["TANKS"][tank]["capacity"] 
                               for tank in self.tanks}
        
        # Create vessels list
        config["VESSELS"] = list(range(1, vessel_count + 1))
        
        # Initialize model
        self.model = ConcreteModel()
        
    def build_sets(self):
        """Build Pyomo sets including tank sets."""
        model = self.model
        
        model.CRUDES = Set(initialize=self.crudes)
        model.LOCATIONS = Set(initialize=self.locations)
        model.SOURCE_LOCATIONS = Set(initialize=self.source_locations)
        model.VESSELS = Set(initialize=self.config["VESSELS"])
        model.DAYS = RangeSet(self.config["DAYS"]["start"], self.config["DAYS"]["end"])
        model.BLENDS = Set(initialize=self.products_info['product'].tolist(), dimen=None)
        model.SLOTS = RangeSet(self.config["DAYS"]["start"], 2 * self.config["DAYS"]["end"])
        
        # Tank sets
        model.TANKS = Set(initialize=self.tanks)
        
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
        
        # Product capacities and ratios
        products_capacity = dict(zip(self.products_info['product'].tolist(), 
                                   self.products_info['max_per_day']))
        crude_margins_dict = dict(zip(self.crudes, self.crude_margins))
        
        model.BCb = Param(model.BLENDS, initialize=products_capacity)
        model.BRcb = Param(model.BLENDS, model.CRUDES, initialize=self.products_ratio, default=0)
        model.MRc = Param(model.CRUDES, initialize=crude_margins_dict)
        
        # Tank parameters
        model.TankCapacity = Param(model.TANKS, initialize=self.tank_capacities)
        model.CrudeTankAssignment = Param(model.CRUDES, initialize=self.tank_assignments, within=model.TANKS)
        
        # Parcel parameters
        parcel_size = {}
        for window, loc_data in self.crude_availability.items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    key = (location, crude_type, window)
                    parcel_size[key] = info["parcel_size"]
        model.PVp = Param(model.PARCELS, initialize=parcel_size)
        
        def pc_init(model, *p):
            return p[1]
        model.PCp = Param(model.PARCELS, initialize=pc_init, within=Any)
        
        model.Travel_Time = Param(model.LOCATIONS, model.LOCATIONS, 
                                initialize=self.time_of_travel)
        
        def pdp_init(model, *p):
            window = p[2]
            return self.window_to_days[window]
        model.PDp = Param(model.PARCELS, initialize=pdp_init)
        
        def plp_init(model, *p):
            return p[0]
        model.PLp = Param(model.PARCELS, within=model.SOURCE_LOCATIONS, initialize=plp_init)
        
        # Refinery capacity parameters
        days = list(range(self.config["DAYS"]["start"], self.config["DAYS"]["end"] + 1))
        capacity_dict = {}
        
        # Handle plant capacity reduction windows if they exist
        if 'plant_capacity_reduction_window' in self.config:
            for entry in self.config['plant_capacity_reduction_window']:
                cap = entry['max_capacity']
                start = entry['start_date']
                end = entry['end_date']
                for day in range(start, end + 1):
                    capacity_dict[day] = cap
        
        default_capacity = self.config['default_capacity']
        for day in days:
            capacity_dict.setdefault(day, default_capacity)
        
        model.RCd = Param(model.DAYS, initialize=capacity_dict)
        
    def build_variables(self):
        """Build decision variables including tank-specific variables."""
        model = self.model
        
        # Main decision variables
        model.AtLocation = Var(model.VESSELS, model.LOCATIONS, model.DAYS, domain=Binary)
        model.Discharge = Var(model.VESSELS, model.DAYS, domain=Binary)
        model.Pickup = Var(model.VESSELS, model.PARCELS, model.DAYS, domain=Binary)
        
        # Tank-specific inventory variables
        model.TankInventory = Var(model.TANKS, model.CRUDES, model.DAYS, domain=NonNegativeReals)
        model.TotalInventory = Var(model.CRUDES, model.DAYS, domain=NonNegativeReals)
        
        model.BlendFraction = Var(model.BLENDS, model.SLOTS, domain=NonNegativeReals)
        model.DischargeDay = Var(model.VESSELS, domain=PositiveIntegers)
        
        # Tank ullage variables
        model.TankUllage = Var(model.TANKS, model.DAYS, domain=NonNegativeReals)
        model.TotalUllage = Var(model.DAYS, domain=NonNegativeReals)
        
        # Auxiliary variables
        model.LocationVisited = Var(model.VESSELS, model.LOCATIONS, domain=Binary)
        model.CrudeInVessel = Var(model.VESSELS, model.CRUDES, domain=Binary)
        model.NumGrades12 = Var(model.VESSELS, domain=Binary)
        model.NumGrades3 = Var(model.VESSELS, domain=Binary)
        model.VolumeDischarged = Var(model.VESSELS, model.CRUDES, model.DAYS, 
                                   domain=NonNegativeReals)
        model.VolumeOnboard = Var(model.VESSELS, model.CRUDES, domain=NonNegativeReals)
        model.IsBlendConsumed = Var(model.BLENDS, model.SLOTS, domain=Binary)
        model.IsTransition = Var(model.BLENDS, model.SLOTS, domain=Binary)
        model.Departure = Var(model.VESSELS, model.LOCATIONS, model.DAYS, domain=Binary)
        
    def build_tank_constraints(self):
        """Build tank-specific constraints."""
        model = self.model
        
        # Tank capacity constraints
        def tank_capacity_rule(model, tank, day):
            assigned_crudes = [crude for crude in model.CRUDES if model.CrudeTankAssignment[crude] == tank]
            if not assigned_crudes:
                return Constraint.Feasible  # Tank has no assigned crudes, so constraint is always satisfied
            return sum(
                model.TankInventory[tank, crude, day]
                for crude in assigned_crudes
            ) <= model.TankCapacity[tank]
        model.TankCapacityConstraint = Constraint(model.TANKS, model.DAYS, rule=tank_capacity_rule)
        
        # Total inventory equals sum of tank inventories
        def total_inventory_rule(model, crude, day):
            assigned_tank = model.CrudeTankAssignment[crude]
            return model.TotalInventory[crude, day] == model.TankInventory[assigned_tank, crude, day]
        model.TotalInventoryRule = Constraint(model.CRUDES, model.DAYS, rule=total_inventory_rule)
        
        # Tank inventory update with crude splitting support
        def tank_inventory_update_rule(model, tank, crude, day):
            # Check if this crude is split across tanks
            crude_splits = self.config.get("CRUDE_SPLITS", {})
            
            if crude in crude_splits:
                # Crude is split - check if this tank has an allocation
                tank_allocation = crude_splits[crude].get(tank, 0)
                if tank_allocation == 0:
                    return model.TankInventory[tank, crude, day] == 0
            else:
                # Traditional single-tank assignment
                if self.tank_assignments[crude] != tank:
                    return model.TankInventory[tank, crude, day] == 0
            
            discharged = 0
            if day <= 5:  
                discharged = 0
            else:
                discharged = sum(model.VolumeDischarged[v, crude, day-5] for v in model.VESSELS)

            # For split crudes, consumption is proportional to tank allocation
            if crude in crude_splits:
                tank_allocation = crude_splits[crude].get(tank, 0)
                total_crude_inventory = sum(crude_splits[crude].values())
                allocation_ratio = tank_allocation / total_crude_inventory if total_crude_inventory > 0 else 0
                
                consumed = allocation_ratio * sum(
                    model.BCb[blend]*model.BRcb[blend,crude]*(model.BlendFraction[blend, 2*day-1] + model.BlendFraction[blend, 2*day])
                    for blend in model.BLENDS
                )
            else:
                # Traditional consumption calculation
                consumed = sum(
                    model.BCb[blend]*model.BRcb[blend,crude]*(model.BlendFraction[blend, 2*day-1] + model.BlendFraction[blend, 2*day])
                    for blend in model.BLENDS
                )
            
            if day == 1:
                # For split crudes, use allocated amount; for others, use full opening inventory
                if crude in crude_splits:
                    opening_inventory = crude_splits[crude].get(tank, 0)
                else:
                    opening_inventory = self.opening_inventory_dict.get(crude, 0)
                return model.TankInventory[tank, crude, day] == opening_inventory + discharged - consumed    
            else:
                return model.TankInventory[tank, crude, day] == model.TankInventory[tank, crude, day-1] + discharged - consumed
        model.TankInventoryUpdate = Constraint(model.TANKS, model.CRUDES, model.DAYS, rule=tank_inventory_update_rule)
        
        # Tank ullage calculation with crude splitting support
        def tank_ullage_rule(model, tank, day):
            # Find all crudes that have inventory in this tank (including split crudes)
            crude_splits = self.config.get("CRUDE_SPLITS", {})
            tank_crudes = []
            
            # Traditional assignments
            for crude in model.CRUDES:
                if self.tank_assignments[crude] == tank:
                    tank_crudes.append(crude)
            
            # Split crude assignments
            for crude, splits in crude_splits.items():
                if tank in splits and crude not in tank_crudes:
                    tank_crudes.append(crude)
            
            if not tank_crudes:
                return model.TankUllage[tank, day] == model.TankCapacity[tank]  # Empty tank has full ullage
            
            return model.TankUllage[tank, day] == model.TankCapacity[tank] - sum(
                model.TankInventory[tank, crude, day]
                for crude in tank_crudes
            )
        model.TankUllageRule = Constraint(model.TANKS, model.DAYS, rule=tank_ullage_rule)
        
        # Total ullage calculation
        def total_ullage_rule(model, day):
            return model.TotalUllage[day] == sum(model.TankUllage[tank, day] for tank in model.TANKS)
        model.TotalUllageRule = Constraint(model.DAYS, rule=total_ullage_rule)
        
    def build_vessel_travel_constraints(self):
        """Build vessel travel constraints."""
        model = self.model
        
        # Constraint 1: A vessel can only be at one location on a given day
        def vessel_single_location_rule(model, v, d):
            return sum(model.AtLocation[v, l, d] for l in model.LOCATIONS) <= 1
        model.VesselSingleLocation = Constraint(model.VESSELS, model.DAYS, 
                                              rule=vessel_single_location_rule)
        
        # Constraint 2: First pickup day equals vessel discharge day
        def first_pickup_day_rule(model, v):
            return model.DischargeDay[v] == sum(
                day * model.Pickup[v, p, day]
                for p in model.PARCELS
                for day in model.DAYS
            )
        model.FirstPickupDay = Constraint(model.VESSELS, rule=first_pickup_day_rule)
        
        # Constraint 3: Enforce travel time (single destination per departure)
        def enforce_travel_time_rule(model, v, l, d):
            valid_destinations = []
            for l2 in model.LOCATIONS:
                if (l, l2) in self.time_of_travel:
                    travel_time = model.Travel_Time[l, l2]
                    arrival_day = d + travel_time
                    if arrival_day in model.DAYS:
                        valid_destinations.append(model.AtLocation[v, l2, arrival_day])
            if valid_destinations:
                return model.Departure[v, l, d] <= sum(valid_destinations)
            else:
                return Constraint.Skip
        model.EnforceTravelTime = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, model.DAYS, 
                                           rule=enforce_travel_time_rule)
        
        # Constraint 4: No early arrival
        def no_early_arrival_rule(model, vessel, source_location, destination_location, start_day, end_day):
            if source_location == destination_location:
                return Constraint.Skip
            
            if (source_location, destination_location) not in self.time_of_travel:
                return Constraint.Skip
            
            if end_day - start_day >= model.Travel_Time[source_location, destination_location]:
                return Constraint.Skip
            
            if end_day <= start_day:
                return Constraint.Skip
            
            return model.AtLocation[vessel, source_location, start_day] + model.AtLocation[vessel, destination_location, end_day] <= 1
        model.NoEarlyArrival = Constraint(model.VESSELS, model.LOCATIONS, 
                                        model.LOCATIONS, model.DAYS, model.DAYS, 
                                        rule=no_early_arrival_rule)
        
        # Constraint 5: Depart after load
        def depart_after_load_rule(model, v, l, d):
            return model.Departure[v, l, d] <= sum(
                model.Pickup[v, p, d]
                for p in model.PARCELS
                if model.PLp[p] == l
            )
        model.DepartAfterLoad = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, model.DAYS, 
                                         rule=depart_after_load_rule)
        
    def build_vessel_loading_constraints(self):
        """Build vessel loading constraints."""
        model = self.model
        
        # At least one parcel per vessel
        def atleast_one_parcel_per_vessel(model, vessel):
            return sum(
                model.Pickup[vessel, parcel, day] 
                for parcel in model.PARCELS
                for day in model.DAYS
            ) >= 1
        model.AtleastOneParcelPerVessel = Constraint(model.VESSELS, rule=atleast_one_parcel_per_vessel)
        
        # One vessel per parcel
        def one_ship_for_one_parcel_pickup(model, *parcel):
            return sum(
                model.Pickup[vessel, parcel, day] 
                for vessel in model.VESSELS
                for day in model.DAYS
            ) <= 1
        model.OneVesselParcel = Constraint(model.PARCELS, rule=one_ship_for_one_parcel_pickup)
        
        # One pickup per day per vessel
        def one_pickup_per_day(model, v, d):
            return sum(
                model.Pickup[v, parcel, d] 
                for parcel in model.PARCELS
            ) <= 1
        model.OnePickupDayVessel = Constraint(model.VESSELS, model.DAYS, rule=one_pickup_per_day)
        
        # Pickup day limits
        def pickup_day_limit(model, vessel, *parcel):
            return sum(
                model.Pickup[vessel, parcel, day]
                for day in model.DAYS
                if day not in model.PDp[parcel]
            ) == 0
        model.PickupDayLimit = Constraint(model.VESSELS, model.PARCELS, rule=pickup_day_limit)
        
        # Parcel location bound
        def parcel_location_bound(model, vessel, day, *parcel):
            return model.Pickup[vessel, parcel, day] <= model.AtLocation[vessel, model.PLp[parcel], day]
        model.ParcelLocationBound = Constraint(model.VESSELS, model.DAYS, model.PARCELS, 
                                             rule=parcel_location_bound)
        
        # Location visited constraints
        M = 30
        def location_visited_constraint_1(model, vessel, location):
            return sum(
                model.AtLocation[vessel, location, day]
                for day in model.DAYS
            ) >= model.LocationVisited[vessel, location]
        model.LocationConstraint1 = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, 
                                             rule=location_visited_constraint_1)
        
        def location_visited_constraint_2(model, vessel, location):
            return sum(
                model.AtLocation[vessel, location, day]
                for day in model.DAYS
            ) <= M * model.LocationVisited[vessel, location]
        model.LocationConstraint2 = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, 
                                             rule=location_visited_constraint_2)
                                             
    def build_vessel_discharge_constraints(self):
        """Build vessel discharge constraints."""
        model = self.model
        
        # Unique discharge day
        def unique_vessel_discharge_day(model, v):
            return sum(model.Discharge[v, d] for d in model.DAYS) == 1
        model.UniqueVesselDischargeDay = Constraint(model.VESSELS, rule=unique_vessel_discharge_day)
        
        # Discharge at Melaka (refinery location)
        def discharge_at_melaka_rule(model, v, d):
            if d == model.DAYS.last():
                return 2 * model.Discharge[v, d] <= model.AtLocation[v, "Melaka", d]
            else:
                return 2 * model.Discharge[v, d] <= model.AtLocation[v, "Melaka", d] + model.AtLocation[v, "Melaka", d + 1]
        model.DischargeAtMelaka = Constraint(model.VESSELS, model.DAYS, rule=discharge_at_melaka_rule)
        
        # No two vessels discharge same or adjacent day
        def no_two_vessels_discharge_same_or_adjacent_day_rule(model, d):
            if d == model.DAYS.last():
                return sum(model.Discharge[v, d] for v in model.VESSELS) <= 1
            else:
                return sum(model.Discharge[v, d] for v in model.VESSELS) + sum(model.Discharge[v, d+1] for v in model.VESSELS) <= 1
        model.NoTwoVesselsDischarge = Constraint(model.DAYS, rule=no_two_vessels_discharge_same_or_adjacent_day_rule)
        
        # Volume onboard calculation
        def volume_onboard_rule(model, v, c):
            return model.VolumeOnboard[v, c] == sum(
                model.PVp[p] * model.Pickup[v, p, d]
                for p in model.PARCELS
                for d in model.DAYS
                if model.PCp[p] == c
            )
        model.VolumeOnboardRule = Constraint(model.VESSELS, model.CRUDES, rule=volume_onboard_rule)
        
        # Volume discharged constraints (linear formulation)
        def volume_discharged_upper_bound_discharge(model, v, c, d):
            return model.VolumeDischarged[v, c, d] <= self.config['vessel_max_limit'] * model.Discharge[v, d]
        model.VolumeDischargedUpperBoundDischarge = Constraint(model.VESSELS, model.CRUDES, model.DAYS, 
                                                             rule=volume_discharged_upper_bound_discharge)
        
        def volume_discharged_upper_bound_onboard(model, v, c, d):
            return model.VolumeDischarged[v, c, d] <= model.VolumeOnboard[v, c]
        model.VolumeDischargedUpperBoundOnboard = Constraint(model.VESSELS, model.CRUDES, model.DAYS, 
                                                           rule=volume_discharged_upper_bound_onboard)
        
        def volume_discharged_lower_bound(model, v, c, d):
            return model.VolumeDischarged[v, c, d] >= model.VolumeOnboard[v, c] - self.config['vessel_max_limit'] * (1 - model.Discharge[v, d])
        model.VolumeDischargedLowerBound = Constraint(model.VESSELS, model.CRUDES, model.DAYS, 
                                                    rule=volume_discharged_lower_bound)
        
        # Crude in vessel constraints
        def crude_in_vessel_rule_1(model, v, c):
            return sum(
                model.Pickup[v, p, d]
                for p in model.PARCELS
                for d in model.DAYS
                if model.PCp[p] == c
            ) >= model.CrudeInVessel[v, c]
        model.CrudeInVessel1 = Constraint(model.VESSELS, model.CRUDES, rule=crude_in_vessel_rule_1)
        
        def crude_in_vessel_rule_2(model, v, c):
            return sum(
                model.Pickup[v, p, d]
                for p in model.PARCELS
                for d in model.DAYS
                if model.PCp[p] == c
            ) <= 5 * model.CrudeInVessel[v, c]
        model.CrudeInVessel2 = Constraint(model.VESSELS, model.CRUDES, rule=crude_in_vessel_rule_2)
        
    def build_vessel_ordering_constraints(self):
        """Build vessel ordering constraints."""
        model = self.model
        
        # Number of grades constraints
        def num_grades_12_rule(model, v):
            return sum(model.CrudeInVessel[v, c] for c in model.CRUDES) == 2 * model.NumGrades12[v] + 3 * model.NumGrades3[v]
        model.NumGrades12Rule = Constraint(model.VESSELS, rule=num_grades_12_rule)
        
        def sum_num_grades_rule(model, v):
            return model.NumGrades12[v] + model.NumGrades3[v] == 1
        model.SumNumGradesRule = Constraint(model.VESSELS, rule=sum_num_grades_rule)
        
        # Vessel capacity constraints
        def vessel_capacity_rule_12(model, v):
            return sum(model.VolumeOnboard[v, c] for c in model.CRUDES) <= (
                self.config["two_parcel_vessel_capacity"] * model.NumGrades12[v] + 
                self.config["three_parcel_vessel_capacity"] * model.NumGrades3[v]
            )
        model.VesselCapacity12Rule = Constraint(model.VESSELS, rule=vessel_capacity_rule_12)
        
    def build_crude_blending_constraints(self):
        """Build crude blending constraints."""
        model = self.model
        
        # Product consumption constraints
        def product_consumption_rule(model, b, s):
            return model.BlendFraction[b, s] <= model.BCb[b] * model.IsBlendConsumed[b, s]
        model.ProductConsumption = Constraint(model.BLENDS, model.SLOTS, rule=product_consumption_rule)
        
        # Transition constraints
        def transition_rule_1(model, b, s):
            if s == 1:
                return Constraint.Skip
            return model.IsTransition[b, s] >= (
                model.IsBlendConsumed[b, s] - model.IsBlendConsumed[b, s-1]
            )
        model.Transition1 = Constraint(model.BLENDS, model.SLOTS, rule=transition_rule_1)
        
        def transition_rule_2(model, b, s):
            if s == max(model.SLOTS):
                return Constraint.Skip
            return model.IsTransition[b, s] >= (
                model.IsBlendConsumed[b, s] - model.IsBlendConsumed[b, s+1]
            )
        model.Transition2 = Constraint(model.BLENDS, model.SLOTS, rule=transition_rule_2)
        
        # Maximum transitions
        def max_transitions_rule(model, b):
            return sum(model.IsTransition[b, s] for s in model.SLOTS) <= self.MaxTransitions
        model.MaxTransitions = Constraint(model.BLENDS, rule=max_transitions_rule)
        
    def build_objectives(self):
        """Build objective functions."""
        model = self.model
        
        # Revenue objective  
        def revenue_objective_rule(model):
            return sum(
                model.MRc[c] * model.VolumeDischarged[v, c, d]
                for v in model.VESSELS
                for c in model.CRUDES
                for d in model.DAYS
            )
        model.RevenueObjective = Objective(rule=revenue_objective_rule, sense=maximize)
        
        # Throughput objective
        def throughput_objective_rule(model):
            return sum(
                model.VolumeDischarged[v, c, d]
                for v in model.VESSELS
                for c in model.CRUDES
                for d in model.DAYS
            )
        model.ThroughputObjective = Objective(rule=throughput_objective_rule, sense=maximize)
        
        # Demurrage objective
        def demurrage_objective_rule(model):
            return sum(
                self.config["demurrage_cost"] * model.DischargeDay[v]
                for v in model.VESSELS
            )
        model.DemurrageObjective = Objective(rule=demurrage_objective_rule, sense=minimize)
        
    def set_objective(self, optimization_type: str, max_demurrage_limit: int = 10):
        """Set the objective function based on optimization type."""
        model = self.model
        
        # Deactivate all objectives first
        model.RevenueObjective.deactivate()
        model.ThroughputObjective.deactivate()
        model.DemurrageObjective.deactivate()
        
        if optimization_type == "revenue":
            model.RevenueObjective.activate()
        elif optimization_type == "throughput":
            model.ThroughputObjective.activate()
        elif optimization_type == "demurrage":
            model.DemurrageObjective.activate()
        
        # Add demurrage limit constraint if not demurrage optimization
        if optimization_type != "demurrage":
            def demurrage_limit_rule(model):
                return sum(model.DischargeDay[v] for v in model.VESSELS) <= max_demurrage_limit
            model.DemurrageLimit = Constraint(rule=demurrage_limit_rule)
        
    def build_model(self, optimization_type: str = "throughput", max_demurrage_limit: int = 10):
        """Build the complete multi-tank optimization model."""
        
        # Define build steps for progress tracking
        build_steps = [
            ("Building sets", self.build_sets),
            ("Building parameters", self.build_parameters),
            ("Building variables", self.build_variables),
            ("Building tank constraints", self.build_tank_constraints),
            ("Building vessel travel constraints", self.build_vessel_travel_constraints),
            ("Building vessel loading constraints", self.build_vessel_loading_constraints),
            ("Building vessel discharge constraints", self.build_vessel_discharge_constraints),
            ("Building vessel ordering constraints", self.build_vessel_ordering_constraints),
            ("Building crude blending constraints", self.build_crude_blending_constraints),
            ("Building objectives", self.build_objectives),
        ]
        
        # Execute build steps with progress bar
        with tqdm(total=len(build_steps) + 1, desc="Building multi-tank model", unit="step") as pbar:
            for step_name, step_func in build_steps:
                pbar.set_description(step_name)
                step_func()
                pbar.update(1)
            
            # Set objective
            pbar.set_description("Setting objective")
            self.set_objective(optimization_type, max_demurrage_limit)
            pbar.update(1)
        
        print("✅ Multi-tank model building complete!")
        print(f"🏭 Tank Configuration:")
        for tank, capacity in self.tank_capacities.items():
            assigned_crudes = [crude for crude, assigned_tank in self.tank_assignments.items() if assigned_tank == tank]
            print(f"   {tank}: {capacity:,} KB - {assigned_crudes}")
        
        return self.model
