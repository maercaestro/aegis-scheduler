"""
Optimization model module for vessel routing and crude blending.
Contains Pyomo model definition with all constraints and objectives.
"""

from pyomo.environ import *
from typing import Dict, List, Any, Optional
import pandas as pd


class OptimizationModel:
    """Creates and manages the Pyomo optimization model."""
    
    def __init__(self, data: Dict[str, Any], vessel_count: int, optimization_type: str = "throughput", 
                 max_demurrage_limit: int = 10):
        """
        Initialize the optimization model with data and parameters.
        
        Args:
            data: Dictionary containing all loaded data
            vessel_count: Number of vessels to optimize
            optimization_type: Type of optimization ('margin' or 'throughput')
            max_demurrage_limit: Maximum demurrage limit for throughput optimization
        """
        self.data = data
        self.vessel_count = vessel_count
        self.optimization_type = optimization_type
        self.max_demurrage_limit = max_demurrage_limit
        self.model = None
        
        # Extract frequently used data
        self.config = data['config']
        self.crudes = data['crudes']
        self.locations = data['locations']
        self.source_location = data['source_location']
        self.products_info = data['products_info']
        self.crude_availability = data['crude_availability']
        self.time_of_travel = data['time_of_travel']
        self.window_to_days = data['window_to_days']
        self.products_ratio = data['products_ratio']
        self.crude_margins = data['crude_margins']
        self.opening_inventory_dict = data['opening_inventory_dict']
        self.capacity_dict = data['capacity_dict']
        
        # Update vessel configuration
        self.config["VESSELS"] = list(range(1, vessel_count + 1))
    
    def create_model(self) -> ConcreteModel:
        """
        Create the complete Pyomo optimization model.
        
        Returns:
            ConcreteModel: The complete Pyomo model
        """
        model = ConcreteModel()
        self.model = model
        
        # Create sets, parameters, variables, and constraints
        self._create_sets()
        self._create_parameters()
        self._create_variables()
        self._create_constraints()
        self._create_objective()
        
        return model
    
    def _create_sets(self):
        """Create all required sets for the model."""
        model = self.model
        
        model.CRUDES = Set(initialize=self.crudes)
        model.LOCATIONS = Set(initialize=self.locations)
        model.SOURCE_LOCATIONS = Set(initialize=self.source_location)
        model.VESSELS = Set(initialize=self.config["VESSELS"])
        model.DAYS = RangeSet(self.config["DAYS"]["start"], self.config["DAYS"]["end"])
        model.BLENDS = Set(initialize=self.products_info['product'].tolist(), dimen=None)
        model.SLOTS = RangeSet(self.config["DAYS"]["start"], 2 * self.config["DAYS"]["end"])
        
        # Create parcels set
        parcel_set = set()
        for window, loc_data in self.crude_availability.items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    parcel_set.add((location, crude_type, window))
        model.PARCELS = Set(initialize=parcel_set, dimen=3)
    
    def _create_parameters(self):
        """Create all required parameters for the model."""
        model = self.model
        
        # Product capacity and ratios
        products_capacity = dict(zip(self.products_info['product'].tolist(), 
                                   self.products_info['max_per_day']))
        crude_margins_dict = dict(zip(self.crudes, self.crude_margins))
        
        model.BCb = Param(model.BLENDS, initialize=products_capacity)
        model.BRcb = Param(model.BLENDS, model.CRUDES, initialize=self.products_ratio, default=0)
        model.MRc = Param(model.CRUDES, initialize=crude_margins_dict)
        
        # Parcel parameters
        parcel_size = {}
        for window, loc_data in self.crude_availability.items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    key = (location, crude_type, window)
                    parcel_size[key] = info["parcel_size"]
        model.PVp = Param(model.PARCELS, initialize=parcel_size)
        
        # Parcel crude type  
        def pc_init(model, *p):
            return p[1]
        model.PCp = Param(model.PARCELS, initialize=pc_init)
        
        # Travel time
        model.Travel_Time = Param(model.LOCATIONS, model.LOCATIONS, initialize=self.time_of_travel)
        
        # Parcel days
        def pdp_init(model, *p):
            window = p[2]
            return self.window_to_days[window]
        model.PDp = Param(model.PARCELS, initialize=pdp_init)
        
        # Parcel location
        def plp_init(model, *p):
            return p[0]
        model.PLp = Param(model.PARCELS, within=model.SOURCE_LOCATIONS, initialize=plp_init)
        
        # Refinery capacity
        model.RCd = Param(model.DAYS, initialize=self.capacity_dict)
    
    def _create_variables(self):
        """Create all decision and auxiliary variables."""
        model = self.model
        
        # Decision variables
        model.AtLocation = Var(model.VESSELS, model.LOCATIONS, model.DAYS, domain=Binary)
        model.Discharge = Var(model.VESSELS, model.DAYS, domain=Binary)
        model.Pickup = Var(model.VESSELS, model.PARCELS, model.DAYS, domain=Binary)
        model.Inventory = Var(model.CRUDES, model.DAYS, domain=NonNegativeReals)
        model.BlendFraction = Var(model.BLENDS, model.SLOTS, domain=NonNegativeReals)
        model.DischargeDay = Var(model.VESSELS, domain=PositiveIntegers)
        model.Ullage = Var(model.DAYS, domain=NonNegativeReals)
        
        # Auxiliary variables
        model.LocationVisited = Var(model.VESSELS, model.LOCATIONS, domain=Binary)
        model.CrudeInVessel = Var(model.VESSELS, model.CRUDES, domain=Binary)
        model.NumGrades12 = Var(model.VESSELS, domain=Binary)
        model.NumGrades3 = Var(model.VESSELS, domain=Binary)
        model.VolumeDischarged = Var(model.VESSELS, model.CRUDES, model.DAYS, domain=NonNegativeReals)
        model.VolumeOnboard = Var(model.VESSELS, model.CRUDES, domain=NonNegativeReals)
        model.IsBlendConsumed = Var(model.BLENDS, model.SLOTS, domain=Binary)
        model.IsTransition = Var(model.BLENDS, model.SLOTS, domain=Binary)
        model.Departure = Var(model.VESSELS, model.LOCATIONS, model.DAYS, domain=Binary)
    
    def _create_constraints(self):
        """Create all model constraints."""
        self._vessel_travel_constraints()
        self._vessel_loading_constraints()
        self._vessel_discharge_constraints()
        self._vessel_ordering_constraints()
        self._crude_blending_constraints()
        self._inventory_constraints()
    
    def _vessel_travel_constraints(self):
        """Create vessel travel related constraints."""
        model = self.model
        
        # Vessel can only be at one location per day
        def vessel_single_location_rule(model, v, d):
            return sum(model.AtLocation[v, l, d] for l in model.LOCATIONS) <= 1
        model.VesselSingleLocation = Constraint(model.VESSELS, model.DAYS, rule=vessel_single_location_rule)
        
        # Departure constraints
        def departure_lower_bound_rule(model, v, l, d):
            if d == model.DAYS[-1]:
                return Constraint.Skip
            else:
                return model.Departure[v, l, d] >= model.AtLocation[v, l, d] - model.AtLocation[v, l, d + 1]
        model.DepartureLowerBound = Constraint(model.VESSELS, model.LOCATIONS, model.DAYS, rule=departure_lower_bound_rule)
        
        def departure_upper_bound1_rule(model, v, l, d):
            return model.Departure[v, l, d] <= model.AtLocation[v, l, d]
        model.DepartureUpperBound1 = Constraint(model.VESSELS, model.LOCATIONS, model.DAYS, rule=departure_upper_bound1_rule)
        
        def departure_upper_bound2_rule(model, v, l, d):
            if d == model.DAYS[-1]:
                return Constraint.Skip
            else:
                return model.Departure[v, l, d] <= 1 - model.AtLocation[v, l, d + 1]
        model.DepartureUpperBound2 = Constraint(model.VESSELS, model.LOCATIONS, model.DAYS, rule=departure_upper_bound2_rule)
        
        def single_departure_per_location_rule(model, v, l):
            return sum(model.Departure[v, l, d] for d in model.DAYS) <= 1
        model.SingleDeparturePerLocation = Constraint(model.VESSELS, model.LOCATIONS, rule=single_departure_per_location_rule)
        
        # Travel time enforcement
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
        model.EnforceTravelTime = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, model.DAYS, rule=enforce_travel_time_rule)
        
        # No early arrival
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
        model.NoEarlyArrival = Constraint(model.VESSELS, model.LOCATIONS, model.LOCATIONS, model.DAYS, model.DAYS, rule=no_early_arrival_rule)
        
        # Depart after loading
        def depart_after_load_rule(model, v, l, d):
            return model.Departure[v, l, d] <= sum(
                model.Pickup[v, p, d]
                for p in model.PARCELS
                if model.PLp[p] == l
            )
        model.DepartAfterLoad = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, model.DAYS, rule=depart_after_load_rule)
    
    def _vessel_loading_constraints(self):
        """Create vessel loading related constraints."""
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
        
        # Pickup day availability
        def pickup_day_limit(model, vessel, *parcel):
            return sum(
                model.Pickup[vessel, parcel, day]
                for day in model.DAYS
                if day not in model.PDp[parcel]
            ) == 0
        model.PickupDayLimit = Constraint(model.VESSELS, model.PARCELS, rule=pickup_day_limit)
        
        # Parcel location constraint
        def parcel_location_bound(model, vessel, day, *parcel):
            return model.Pickup[vessel, parcel, day] <= model.AtLocation[vessel, model.PLp[parcel], day]
        model.ParcelLocationBound = Constraint(model.VESSELS, model.DAYS, model.PARCELS, rule=parcel_location_bound)
        
        # Location visited constraints
        M = 30
        def location_visited_constraint_1(model, vessel, location):
            return sum(
                model.AtLocation[vessel, location, day]
                for day in model.DAYS
            ) >= model.LocationVisited[vessel, location]
        model.LocationConstraint1 = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, rule=location_visited_constraint_1)
        
        def location_visited_constraint_2(model, vessel, location):
            return sum(
                model.AtLocation[vessel, location, day]
                for day in model.DAYS
            ) <= M * model.LocationVisited[vessel, location]
        model.LocationConstraint2 = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, rule=location_visited_constraint_2)
        
        def location_visited_constraint_3(model, vessel, location):
            return sum(
                sum(
                    model.Pickup[vessel, parcel, day]
                    for day in model.DAYS
                )
                for parcel in model.PARCELS if model.PLp[parcel] == location
            ) >= model.LocationVisited[vessel, location]
        model.LocationConstraint3 = Constraint(model.VESSELS, model.SOURCE_LOCATIONS, rule=location_visited_constraint_3)
        
        # Crude in vessel constraints
        def crude_in_vessel_bound_with_pickup(model, vessel, crude):
            return sum(
                sum(
                    model.Pickup[vessel, parcel, day]
                    for day in model.DAYS
                )
                for parcel in model.PARCELS if model.PCp[parcel] == crude
            ) >= model.CrudeInVessel[vessel, crude]
        model.CrudeInVesselBoundWithPickup = Constraint(model.VESSELS, model.CRUDES, rule=crude_in_vessel_bound_with_pickup)
        
        def crude_in_vessel_lower_bound(model, vessel, crude):
            return sum(
                sum(
                    model.Pickup[vessel, parcel, day]
                    for day in model.DAYS
                )
                for parcel in model.PARCELS if model.PCp[parcel] == crude
            ) <= model.CrudeInVessel[vessel, crude] * M
        model.CrudeInVesselLowerBound = Constraint(model.VESSELS, model.CRUDES, rule=crude_in_vessel_lower_bound)
        
        def max_3_crudes_limit(model, vessel):
            return sum(
                model.CrudeInVessel[vessel, crude]
                for crude in model.CRUDES
            ) <= 3
        model.Max3CrudesLimit = Constraint(model.VESSELS, rule=max_3_crudes_limit)
        
        # Crude group and volume limits
        def crude_group_limit(model, vessel):
            return model.NumGrades12[vessel] + model.NumGrades3[vessel] == 1
        model.CrudeGroupLimit = Constraint(model.VESSELS, rule=crude_group_limit)
        
        def total_crude_upper_limit(model, vessel, crude):
            return 2 * model.NumGrades12[vessel] + 3 * model.NumGrades3[vessel] >= sum(
                model.CrudeInVessel[vessel, crude]
                for crude in model.CRUDES
            )
        model.TotalCrudeUpperLimit = Constraint(model.VESSELS, model.CRUDES, rule=total_crude_upper_limit)
        
        def total_crude_lower_limit(model, vessel, crude):
            return model.NumGrades12[vessel] + 3 * model.NumGrades3[vessel] <= sum(
                model.CrudeInVessel[vessel, crude]
                for crude in model.CRUDES
            )
        model.TotalCrudeLowerLimit = Constraint(model.VESSELS, model.CRUDES, rule=total_crude_lower_limit)
        
        def crude_count_wise_vessel_volume_limit(model, vessel):
            return sum(
                model.PVp[parcel] * sum(
                    model.Pickup[vessel, parcel, day]
                    for day in model.DAYS
                )
                for parcel in model.PARCELS
            ) <= (self.config['two_parcel_vessel_capacity'] * model.NumGrades12[vessel] + 
                  self.config['three_parcel_vessel_capacity'] * model.NumGrades3[vessel])
        model.CrudeCountWiseVesselVolume = Constraint(model.VESSELS, rule=crude_count_wise_vessel_volume_limit)
    
    def _vessel_discharge_constraints(self):
        """Create vessel discharge related constraints."""
        model = self.model
        
        # Unique discharge day per vessel
        def unique_vessel_discharge_day(model, v):
            return sum(model.Discharge[v, d] for d in model.DAYS) == 1
        model.UniqueVesselDischargeDay = Constraint(model.VESSELS, rule=unique_vessel_discharge_day)
        
        # Discharge at Melaka
        def discharge_at_melaka_rule(model, v, d):
            if d == model.DAYS[-1]:
                return 2 * model.Discharge[v, d] <= model.AtLocation[v, "Melaka", d]
            else:
                return 2 * model.Discharge[v, d] <= model.AtLocation[v, "Melaka", d] + model.AtLocation[v, "Melaka", d + 1]
        model.DischargeAtMelaka = Constraint(model.VESSELS, model.DAYS, rule=discharge_at_melaka_rule)
        
        # No two vessels discharge simultaneously
        def no_two_vessels_discharge_same_or_adjacent_day_rule(model, d):
            if d == model.DAYS[-1]:
                return sum(model.Discharge[v, d] for v in model.VESSELS) <= 1
            else:
                return sum(model.Discharge[v, d] + model.Discharge[v, d + 1] for v in model.VESSELS) <= 1
        model.NoTwoDischargeSameOrAdjacent = Constraint(model.DAYS, rule=no_two_vessels_discharge_same_or_adjacent_day_rule)
        
        # Vessel stops after discharge
        def vessel_stops_after_discharge_rule(model, v, l, d1, d2):
            if d2 <= d1 + 1:
                return Constraint.Skip
            return model.AtLocation[v, l, d2] <= 1 - model.Discharge[v, d1]
        model.VesselStopsAfterDischarge = Constraint(model.VESSELS, model.LOCATIONS, model.DAYS, model.DAYS, rule=vessel_stops_after_discharge_rule)
        
        # Volume onboard and discharge constraints
        def volume_onboard_rule(model, v, c):
            return model.VolumeOnboard[v, c] == sum(
                model.PVp[p] * sum(model.Pickup[v, p, d] for d in model.DAYS)
                for p in model.PARCELS
                if model.PCp[p] == c
            )
        model.VolumeOnboardDef = Constraint(model.VESSELS, model.CRUDES, rule=volume_onboard_rule)
        
        def discharge_upper_limit_rule(model, v, c, d):
            return model.VolumeDischarged[v, c, d] <= self.config['vessel_max_limit'] * model.Discharge[v, d]
        model.DischargeUpperLimit = Constraint(model.VESSELS, model.CRUDES, model.DAYS, rule=discharge_upper_limit_rule)
        
        def discharge_no_more_than_onboard_rule(model, v, c, d):
            return model.VolumeDischarged[v, c, d] <= model.VolumeOnboard[v, c]
        model.DischargeNoMoreThanOnboard = Constraint(model.VESSELS, model.CRUDES, model.DAYS, rule=discharge_no_more_than_onboard_rule)
        
        def discharge_lower_bound_rule(model, v, c, d):
            return model.VolumeDischarged[v, c, d] >= model.VolumeOnboard[v, c] - self.config['vessel_max_limit'] * (1 - model.Discharge[v, d])
        model.DischargeLowerBound = Constraint(model.VESSELS, model.CRUDES, model.DAYS, rule=discharge_lower_bound_rule)
    
    def _vessel_ordering_constraints(self):
        """Create vessel ordering (symmetry breaking) constraints."""
        model = self.model
        
        # Discharge day calculation
        def discharge_day_rule(model, v):
            return model.DischargeDay[v] == sum(d * model.Discharge[v, d] for d in model.DAYS)
        model.CalcDischargeDay = Constraint(model.VESSELS, rule=discharge_day_rule)
        
        # Symmetry breaking
        def symmetry_breaking_rule(model, v):
            if v < len(model.VESSELS):
                return model.DischargeDay[v] + 1 <= model.DischargeDay[v + 1]
            return Constraint.Skip
        model.SymmetryBreak = Constraint(model.VESSELS, rule=symmetry_breaking_rule)
    
    def _crude_blending_constraints(self):
        """Create crude blending related constraints."""
        model = self.model
        
        # Blend consumption constraints
        def is_blend_greater_than_fraction_rule(model, s, *b):
            return model.IsBlendConsumed[b, s] >= model.BlendFraction[b, s]
        model.IsBlendVsFraction = Constraint(model.SLOTS, model.BLENDS, rule=is_blend_greater_than_fraction_rule)
        
        def one_blend_per_slot_rule(model, s):
            return sum(model.IsBlendConsumed[b, s] for b in model.BLENDS) == 1
        model.OneBlendPerSlot = Constraint(model.SLOTS, rule=one_blend_per_slot_rule)
        
        # Blend fraction daily bounds
        def blend_fraction_daily_upper_bound_rule(model, s):
            if s % 2 == 1 and s + 1 in model.SLOTS:
                return sum(model.BlendFraction[b, s] + model.BlendFraction[b, s + 1] for b in model.BLENDS) <= 1
            else:
                return Constraint.Skip
        model.BlendFractionDailyBound = Constraint(model.SLOTS, rule=blend_fraction_daily_upper_bound_rule)
        
        # Transition constraints
        def transition_lower_bound_rule(model, s, *b):
            if s + 1 in model.SLOTS:
                return model.IsTransition[b, s] >= model.IsBlendConsumed[b, s] - model.IsBlendConsumed[b, s + 1]
            else:
                return Constraint.Skip
        model.TransitionLowerBound = Constraint(model.SLOTS, model.BLENDS, rule=transition_lower_bound_rule)
        
        def transition_upper_bound1_rule(model, s, *b):
            return model.IsTransition[b, s] <= model.IsBlendConsumed[b, s]
        model.TransitionUpperBound1 = Constraint(model.SLOTS, model.BLENDS, rule=transition_upper_bound1_rule)
        
        def transition_upper_bound2_rule(model, s, *b):
            if s + 1 in model.SLOTS:
                return model.IsTransition[b, s] <= 1 - model.IsBlendConsumed[b, s + 1]
            else:
                return Constraint.Skip
        model.TransitionUpperBound2 = Constraint(model.SLOTS, model.BLENDS, rule=transition_upper_bound2_rule)
        
        def max_transitions_rule(model):
            return sum(model.IsTransition[b, s] for b in model.BLENDS for s in model.SLOTS) <= self.config.get("MaxTransitions", 10)
        model.MaxTransitionsConstraint = Constraint(rule=max_transitions_rule)
        
        # Plant capacity constraints
        def plant_capacity_rule(model, d):
            return sum(
                model.BCb[b] * (model.BlendFraction[b, 2*d - 1] + model.BlendFraction[b, 2*d])
                for b in model.BLENDS
            ) <= model.RCd[d]
        model.PlantCapacityConstraint = Constraint(model.DAYS, rule=plant_capacity_rule)
        
        def minimum_plant_capacity_production_rule(model, d):
            return sum(
                model.BCb[b] * (model.BlendFraction[b, 2*d - 1] + model.BlendFraction[b, 2*d])
                for b in model.BLENDS
            ) >= self.config['turn_down_capacity']
        model.MinimumPlantCapacityProductionConstraint = Constraint(model.DAYS, rule=minimum_plant_capacity_production_rule)
    
    def _inventory_constraints(self):
        """Create inventory related constraints."""
        model = self.model
        
        # Inventory update
        def inventory_update_rule(model, c, d):
            discharged = 0
            if d > 5:
                discharged = sum(model.VolumeDischarged[v, c, d-5] for v in model.VESSELS)

            consumed = sum(
                model.BCb[blend]*model.BRcb[blend,c]*(model.BlendFraction[blend, 2*d-1] + model.BlendFraction[blend, 2*d])
                for blend in model.BLENDS
            )
            if d == 1:
                return model.Inventory[c, d] == self.opening_inventory_dict[c] + discharged - consumed
            else:
                return model.Inventory[c, d] == model.Inventory[c, d-1] + discharged - consumed
        model.InventoryUpdate = Constraint(model.CRUDES, model.DAYS, rule=inventory_update_rule)
        
        # Maximum inventory limit
        def max_inventory_limit(model, day):
            return sum(
                model.Inventory[crude, day]
                for crude in model.CRUDES
            ) <= self.config["INVENTORY_MAX_VOLUME"]
        model.MaxInventoryLimit = Constraint(model.DAYS, rule=max_inventory_limit)
        
        # Ullage update
        def ullage_update_rule(model, d):
            consumed = sum(
                model.BCb[b] * (model.BlendFraction[b, 2*d - 1] + model.BlendFraction[b, 2*d])
                for b in model.BLENDS
            )
            if d == 1:
                return model.Ullage[d] == (self.config["INVENTORY_MAX_VOLUME"] - 
                                         sum(self.opening_inventory_dict[c] for c in model.CRUDES) + consumed)
            
            discharged = 0
            if (d - 1) in model.DAYS:
                discharged = sum(
                    model.VolumeDischarged[v, c, d - 1]
                    for v in model.VESSELS
                    for c in model.CRUDES
                )
            return model.Ullage[d] == model.Ullage[d - 1] - discharged + consumed
        model.UllageUpdate = Constraint(model.DAYS, rule=ullage_update_rule)
    
    def _create_objective(self):
        """Create objective function based on optimization type."""
        model = self.model
        
        # Create expressions for demurrage and profit
        def demurrage_at_source_expr(model):
            return self.config['demurrage_cost'] * (
                sum(
                    model.AtLocation[vessel, location, day] 
                    for vessel in model.VESSELS
                    for location in model.LOCATIONS
                    for day in model.DAYS
                    if location != 'Melaka'
                ) - sum(
                    model.Pickup[vessel, parcel, day]
                    for vessel in model.VESSELS
                    for parcel in model.PARCELS
                    for day in model.DAYS
                )
            )
        model.DemurrageAtSource = Expression(rule=demurrage_at_source_expr)
        
        def demurrage_at_melaka_expr(model):
            return self.config['demurrage_cost'] * (sum(
                sum(
                    model.AtLocation[vessel, 'Melaka', day]
                    for day in model.DAYS
                ) - 2
                for vessel in model.VESSELS
            ))
        model.DemurrageAtMelaka = Expression(rule=demurrage_at_melaka_expr)
        
        def total_profit_expr(model):
            return sum(
                model.MRc[crude]*model.BRcb[blend,crude]*model.BCb[blend]*model.BlendFraction[blend,slot]
                for crude in model.CRUDES
                for blend in model.BLENDS
                for slot in model.SLOTS
            )
        model.TotalProfit = Expression(rule=total_profit_expr)
        
        def total_throughput_expr(model):
            return sum(
                model.BCb[blend]*model.BlendFraction[blend,slot]
                for blend in model.BLENDS
                for slot in model.SLOTS
            )
        model.Throughput = Expression(rule=total_throughput_expr)
        
        # Set objective based on optimization type
        if self.optimization_type == 'margin':
            def net_profit_objective_rule(model):
                return model.TotalProfit - (model.DemurrageAtSource + model.DemurrageAtMelaka)
            model.objective = Objective(rule=net_profit_objective_rule, sense=maximize)
        elif self.optimization_type == 'throughput':
            # Add demurrage limit constraint
            model.DemurrageLimitConstraint = Constraint(
                expr=model.DemurrageAtSource + model.DemurrageAtMelaka <= 
                     self.max_demurrage_limit * self.config["demurrage_cost"]
            )
            def total_throughput_objective_rule(model):
                return model.Throughput
            model.objective = Objective(rule=total_throughput_objective_rule, sense=maximize)
        else:
            raise ValueError(f"Unknown optimization type: {self.optimization_type}")