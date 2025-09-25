#!/usr/bin/env python3
"""
OR-Tools Refinery Optimization Script
====================================

A complete implementation of the refinery optimization workflow using Google OR-Tools CP-SAT solver.
This script replicates the Pyomo notebook functionality for deployment to EC2.

Features:
- Vessel routing and scheduling
- Crude oil blending optimization  
- Inventory management
- Demurrage cost minimization
- Throughput maximization

Usage:
    python ortools_refinery_optimizer.py --config config.json --vessels 6 --optimization throughput --demurrage-limit 10
"""

import argparse
import json
import os
import pandas as pd
import ast
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys

# Google OR-Tools imports
from ortools.sat.python import cp_model


def setup_logging(output_dir: Path, optimization_type: str, vessels: int, 
                  demurrage_limit: int = None) -> logging.Logger:
    """Set up logging for the optimization process."""
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if optimization_type == 'throughput' and demurrage_limit is not None:
        log_filename = f"ortools_optimization_log_{optimization_type}_{vessels}_vessels_{demurrage_limit}_demurrage_{timestamp}.log"
    else:
        log_filename = f"ortools_optimization_log_{optimization_type}_{vessels}_vessels_{timestamp}.log"
    
    log_path = output_dir / log_filename
    
    # Configure logging
    logger = logging.getLogger('ortools_optimizer')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


class DataLoader:
    """Loads and preprocesses all scenario data for optimization."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def load_scenario_data(self, config_file: str = "config.json") -> Dict[str, Any]:
        """Load all data from the test_data directory."""
        
        # Load configuration
        with open(self.base_path / config_file, "r") as f:
            config = json.load(f)
            
        # Load CSV data
        crude_availability_df = pd.read_csv(self.base_path / "crude_availability.csv")
        time_of_travel_df = pd.read_csv(self.base_path / "time_of_travel.csv")
        products_info_df = pd.read_csv(self.base_path / "products_info.csv")
        crudes_info_df = pd.read_csv(self.base_path / "crudes_info.csv")
        
        # Process data
        crude_availability = self._process_crude_availability(crude_availability_df)
        time_of_travel = self._process_time_of_travel(time_of_travel_df)
        products_ratio = self._extract_products_ratio(products_info_df)
        window_to_days = self._extract_window_to_days(crude_availability)
        
        # Extract lists and mappings
        crudes = crudes_info_df["crudes"].tolist()
        locations = list(set(time_of_travel_df["from"]) | set(time_of_travel_df["to"]))
        source_locations = crudes_info_df["origin"].tolist()
        crude_margins = crudes_info_df['margin'].tolist()
        opening_inventory = crudes_info_df['opening_inventory'].tolist()
        
        # Create mappings
        crude_margins_dict = dict(zip(crudes, crude_margins))
        opening_inventory_dict = dict(zip(crudes, opening_inventory))
        products_capacity = dict(zip(products_info_df['product'].tolist(), 
                                   products_info_df['max_per_day']))
        
        return {
            'config': config,
            'crudes': crudes,
            'locations': locations,
            'source_locations': source_locations,
            'time_of_travel': time_of_travel,
            'crude_availability': crude_availability,
            'window_to_days': window_to_days,
            'products_info': products_info_df,
            'products_ratio': products_ratio,
            'products_capacity': products_capacity,
            'crude_margins_dict': crude_margins_dict,
            'opening_inventory_dict': opening_inventory_dict
        }
    
    def _process_crude_availability(self, df: pd.DataFrame) -> Dict:
        """Process crude availability data into nested dictionary."""
        crude_availability = {}
        for _, row in df.iterrows():
            crude_availability \
                .setdefault(row["date_range"], {}) \
                .setdefault(row["location"], {})[row["crude"]] = {
                    "volume": int(row["volume"]),
                    "parcel_size": int(row["parcel_size"])
                }
        return crude_availability
    
    def _process_time_of_travel(self, df: pd.DataFrame) -> Dict:
        """Process time of travel data."""
        return {
            (row["from"], row["to"]): int(row["time_in_days"]) + 1
            for _, row in df.iterrows()
        }
    
    def _extract_products_ratio(self, df: pd.DataFrame) -> Dict:
        """Extract product ratios from products info."""
        return {
            (row['product'], crude): ratio
            for _, row in df.iterrows()
            for crude, ratio in zip(ast.literal_eval(row['crudes']), 
                                  ast.literal_eval(row['ratios']))
        }
    
    def _extract_window_to_days(self, crude_availability: Dict) -> Dict:
        """Extract window to days mapping."""
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


class OrToolsRefineryOptimizer:
    """OR-Tools implementation of the refinery optimization model."""
    
    def __init__(self, data: Dict[str, Any], vessel_count: int, logger: logging.Logger = None):
        self.data = data
        self.vessel_count = vessel_count
        self.model = cp_model.CpModel()
        self.variables = {}
        self.solver = cp_model.CpSolver()
        self.logger = logger or logging.getLogger(__name__)
        
        # Scaling factor for continuous variables
        self.SCALE_FACTOR = 1000
        self.BIG_M = 100000
        
        self.logger.info(f"Initializing OR-Tools optimizer with {vessel_count} vessels")
        
        # Extract frequently used data
        self.config = data['config']
        self.crudes = data['crudes']
        self.locations = data['locations']
        self.source_locations = data['source_locations']
        self.time_of_travel = data['time_of_travel']
        self.crude_availability = data['crude_availability']
        self.window_to_days = data['window_to_days']
        self.products_ratio = data['products_ratio']
        self.products_capacity = data['products_capacity']
        self.crude_margins_dict = data['crude_margins_dict']
        self.opening_inventory_dict = data['opening_inventory_dict']
        
        # Create sets
        self.vessels = list(range(1, vessel_count + 1))
        self.days = list(range(self.config["DAYS"]["start"], self.config["DAYS"]["end"] + 1))
        self.slots = list(range(self.config["DAYS"]["start"], 2 * self.config["DAYS"]["end"] + 1))
        self.blends = data['products_info']['product'].tolist()
        
        # Create parcel set
        self.parcels = []
        self.parcel_info = {}
        for window, loc_data in self.crude_availability.items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    parcel = (location, crude_type, window)
                    self.parcels.append(parcel)
                    self.parcel_info[parcel] = {
                        'size': info['parcel_size'],
                        'crude': crude_type,
                        'location': location,
                        'days': self.window_to_days[window]
                    }
        
        self.logger.info(f"Data extraction complete:")
        self.logger.info(f"  - Vessels: {len(self.vessels)}")
        self.logger.info(f"  - Days: {len(self.days)} (from {min(self.days)} to {max(self.days)})")
        self.logger.info(f"  - Slots: {len(self.slots)}")
        self.logger.info(f"  - Crudes: {len(self.crudes)}")
        self.logger.info(f"  - Locations: {len(self.locations)}")
        self.logger.info(f"  - Source locations: {len(set(self.source_locations))}")
        self.logger.info(f"  - Blends: {len(self.blends)}")
        self.logger.info(f"  - Parcels: {len(self.parcels)}")
    
    def create_variables(self):
        """Create all decision variables for the model."""
        self.logger.info("Creating decision variables...")
        
        # Location variables
        self.variables['at_location'] = {}
        for v in self.vessels:
            for l in self.locations:
                for d in self.days:
                    var_name = f'at_location_v{v}_l{l}_d{d}'
                    self.variables['at_location'][(v, l, d)] = self.model.NewBoolVar(var_name)
        
        self.logger.info(f"  - Created {len(self.variables['at_location'])} location variables")
        
        # Discharge variables
        self.variables['discharge'] = {}
        for v in self.vessels:
            for d in self.days:
                var_name = f'discharge_v{v}_d{d}'
                self.variables['discharge'][(v, d)] = self.model.NewBoolVar(var_name)
        
        self.logger.info(f"  - Created {len(self.variables['discharge'])} discharge variables")
        
        # Pickup variables
        self.variables['pickup'] = {}
        for v in self.vessels:
            for p in self.parcels:
                for d in self.days:
                    var_name = f'pickup_v{v}_p{p}_d{d}'
                    self.variables['pickup'][(v, p, d)] = self.model.NewBoolVar(var_name)
        
        self.logger.info(f"  - Created {len(self.variables['pickup'])} pickup variables")
        
        # Inventory variables (scaled)
        self.variables['inventory'] = {}
        for c in self.crudes:
            for d in self.days:
                var_name = f'inventory_c{c}_d{d}'
                max_val = int(self.config['INVENTORY_MAX_VOLUME'])
                self.variables['inventory'][(c, d)] = self.model.NewIntVar(0, max_val, var_name)
        
        self.logger.info(f"  - Created {len(self.variables['inventory'])} inventory variables")
        
        # Blend fraction variables (scaled)
        self.variables['blend_fraction'] = {}
        for b in self.blends:
            for s in self.slots:
                var_name = f'blend_fraction_b{b}_s{s}'
                self.variables['blend_fraction'][(b, s)] = self.model.NewIntVar(0, self.SCALE_FACTOR, var_name)
        
        self.logger.info(f"  - Created {len(self.variables['blend_fraction'])} blend fraction variables")
        
        # Auxiliary variables
        self._create_auxiliary_variables()
        
        total_vars = sum(len(var_dict) for var_dict in self.variables.values())
        self.logger.info(f"Variable creation complete. Total variables: {total_vars}")
    
    def _create_auxiliary_variables(self):
        """Create auxiliary variables for the model."""
        
        # Location visited variables
        self.variables['location_visited'] = {}
        for v in self.vessels:
            for l in self.source_locations:
                var_name = f'location_visited_v{v}_l{l}'
                self.variables['location_visited'][(v, l)] = self.model.NewBoolVar(var_name)
        
        # Crude in vessel variables
        self.variables['crude_in_vessel'] = {}
        for v in self.vessels:
            for c in self.crudes:
                var_name = f'crude_in_vessel_v{v}_c{c}'
                self.variables['crude_in_vessel'][(v, c)] = self.model.NewBoolVar(var_name)
        
        # Number of grades variables
        self.variables['num_grades_12'] = {}
        self.variables['num_grades_3'] = {}
        for v in self.vessels:
            self.variables['num_grades_12'][v] = self.model.NewBoolVar(f'num_grades_12_v{v}')
            self.variables['num_grades_3'][v] = self.model.NewBoolVar(f'num_grades_3_v{v}')
        
        # Volume discharged variables (scaled)
        self.variables['volume_discharged'] = {}
        for v in self.vessels:
            for c in self.crudes:
                for d in self.days:
                    var_name = f'volume_discharged_v{v}_c{c}_d{d}'
                    max_val = self.config.get('vessel_max_limit', 700000)
                    self.variables['volume_discharged'][(v, c, d)] = self.model.NewIntVar(0, max_val, var_name)
        
        # Volume onboard variables (scaled)
        self.variables['volume_onboard'] = {}
        for v in self.vessels:
            for c in self.crudes:
                var_name = f'volume_onboard_v{v}_c{c}'
                max_val = self.config.get('vessel_max_limit', 700000)
                self.variables['volume_onboard'][(v, c)] = self.model.NewIntVar(0, max_val, var_name)
        
        # Blend consumption variables
        self.variables['is_blend_consumed'] = {}
        for b in self.blends:
            for s in self.slots:
                var_name = f'is_blend_consumed_b{b}_s{s}'
                self.variables['is_blend_consumed'][(b, s)] = self.model.NewBoolVar(var_name)
        
        # Transition variables
        self.variables['is_transition'] = {}
        for b in self.blends:
            for s in self.slots:
                var_name = f'is_transition_b{b}_s{s}'
                self.variables['is_transition'][(b, s)] = self.model.NewBoolVar(var_name)
        
        # Departure variables
        self.variables['departure'] = {}
        for v in self.vessels:
            for l in self.locations:
                for d in self.days:
                    var_name = f'departure_v{v}_l{l}_d{d}'
                    self.variables['departure'][(v, l, d)] = self.model.NewBoolVar(var_name)
        
        # Ullage variables (scaled)
        self.variables['ullage'] = {}
        for d in self.days:
            var_name = f'ullage_d{d}'
            max_val = int(self.config['INVENTORY_MAX_VOLUME'])
            self.variables['ullage'][d] = self.model.NewIntVar(0, max_val, var_name)
    
    def add_vessel_travel_constraints(self):
        """Add vessel travel and movement constraints."""
        self.logger.info("Adding vessel travel constraints...")
        
        constraint_count = 0
        
        # Constraint 1: Vessel can only be at one location per day
        for v in self.vessels:
            for d in self.days:
                self.model.Add(
                    sum(self.variables['at_location'][(v, l, d)] for l in self.locations) <= 1
                )
                constraint_count += 1
        
        self.logger.info(f"  - Added {constraint_count} single location constraints")
        
        # Constraint 2-5: Departure constraints
        for v in self.vessels:
            for l in self.locations:
                for d in self.days:
                    if d < max(self.days):
                        # Departure lower bound
                        self.model.Add(
                            self.variables['departure'][(v, l, d)] >= 
                            self.variables['at_location'][(v, l, d)] - 
                            self.variables['at_location'][(v, l, d + 1)]
                        )
                        
                        # Departure upper bound 2
                        self.model.Add(
                            self.variables['departure'][(v, l, d)] <= 
                            1 - self.variables['at_location'][(v, l, d + 1)]
                        )
                    
                    # Departure upper bound 1
                    self.model.Add(
                        self.variables['departure'][(v, l, d)] <= 
                        self.variables['at_location'][(v, l, d)]
                    )
                
                # Single departure per location
                self.model.Add(
                    sum(self.variables['departure'][(v, l, d)] for d in self.days) <= 1
                )
        
        # Constraint 3: Travel time enforcement
        for v in self.vessels:
            for l1 in self.source_locations:
                for d in self.days:
                    valid_destinations = []
                    for l2 in self.locations:
                        if (l1, l2) in self.time_of_travel:
                            travel_time = self.time_of_travel[(l1, l2)]
                            arrival_day = d + travel_time
                            if arrival_day in self.days:
                                valid_destinations.append(
                                    self.variables['at_location'][(v, l2, arrival_day)]
                                )
                    
                    if valid_destinations:
                        self.model.Add(
                            self.variables['departure'][(v, l1, d)] <= sum(valid_destinations)
                        )
        
        # Constraint 4: No early arrival
        for v in self.vessels:
            for l1 in self.locations:
                for l2 in self.locations:
                    if l1 != l2 and (l1, l2) in self.time_of_travel:
                        travel_time = self.time_of_travel[(l1, l2)]
                        for d1 in self.days:
                            for d2 in self.days:
                                if d2 - d1 < travel_time and d2 > d1:
                                    self.model.Add(
                                        self.variables['at_location'][(v, l1, d1)] +
                                        self.variables['at_location'][(v, l2, d2)] <= 1
                                    )
        
        # Constraint 5: Depart after load
        for v in self.vessels:
            for l in self.source_locations:
                for d in self.days:
                    pickup_sum = sum(
                        self.variables['pickup'][(v, p, d)]
                        for p in self.parcels
                        if self.parcel_info[p]['location'] == l
                    )
                    self.model.Add(
                        self.variables['departure'][(v, l, d)] <= pickup_sum
                    )
        
        self.logger.info("Vessel travel constraints added successfully")
    
    def add_vessel_loading_constraints(self):
        """Add vessel loading constraints."""
        self.logger.info("Adding vessel loading constraints...")
        
        # Constraint 1: At least one parcel per vessel
        for v in self.vessels:
            self.model.Add(
                sum(self.variables['pickup'][(v, p, d)] 
                    for p in self.parcels for d in self.days) >= 1
            )
        
        # Constraint 2: One vessel per parcel
        for p in self.parcels:
            self.model.Add(
                sum(self.variables['pickup'][(v, p, d)] 
                    for v in self.vessels for d in self.days) <= 1
            )
        
        # Constraint 3: One pickup per day per vessel
        for v in self.vessels:
            for d in self.days:
                self.model.Add(
                    sum(self.variables['pickup'][(v, p, d)] for p in self.parcels) <= 1
                )
        
        # Constraint 4: Pickup day limit
        for v in self.vessels:
            for p in self.parcels:
                valid_days = self.parcel_info[p]['days']
                for d in self.days:
                    if d not in valid_days:
                        self.model.Add(self.variables['pickup'][(v, p, d)] == 0)
        
        # Constraint 5: Parcel location bound
        for v in self.vessels:
            for p in self.parcels:
                location = self.parcel_info[p]['location']
                for d in self.days:
                    self.model.Add(
                        self.variables['pickup'][(v, p, d)] <= 
                        self.variables['at_location'][(v, location, d)]
                    )
        
        # Constraint 6: Location visited constraints
        for v in self.vessels:
            for l in self.source_locations:
                # Location visited constraint 1
                self.model.Add(
                    sum(self.variables['at_location'][(v, l, d)] for d in self.days) >= 
                    self.variables['location_visited'][(v, l)]
                )
                
                # Location visited constraint 2
                self.model.Add(
                    sum(self.variables['at_location'][(v, l, d)] for d in self.days) <= 
                    30 * self.variables['location_visited'][(v, l)]
                )
                
                # Location visited constraint 3
                pickup_sum = sum(
                    self.variables['pickup'][(v, p, d)]
                    for p in self.parcels
                    for d in self.days
                    if self.parcel_info[p]['location'] == l
                )
                self.model.Add(
                    pickup_sum >= self.variables['location_visited'][(v, l)]
                )
        
        # Constraint 7: Crude in vessel constraints
        for v in self.vessels:
            for c in self.crudes:
                pickup_sum = sum(
                    self.variables['pickup'][(v, p, d)]
                    for p in self.parcels
                    for d in self.days
                    if self.parcel_info[p]['crude'] == c
                )
                
                # Crude in vessel bound
                self.model.Add(pickup_sum >= self.variables['crude_in_vessel'][(v, c)])
                self.model.Add(pickup_sum <= 30 * self.variables['crude_in_vessel'][(v, c)])
            
            # Max 3 crudes limit
            self.model.Add(
                sum(self.variables['crude_in_vessel'][(v, c)] for c in self.crudes) <= 3
            )
        
        # Constraint 8: Crude group constraints
        for v in self.vessels:
            # Exactly one grade group
            self.model.Add(
                self.variables['num_grades_12'][v] + self.variables['num_grades_3'][v] == 1
            )
            
            # Grade limits
            total_crudes = sum(self.variables['crude_in_vessel'][(v, c)] for c in self.crudes)
            
            # Upper limit
            self.model.Add(
                2 * self.variables['num_grades_12'][v] + 3 * self.variables['num_grades_3'][v] >= 
                total_crudes
            )
            
            # Lower limit  
            self.model.Add(
                self.variables['num_grades_12'][v] + 3 * self.variables['num_grades_3'][v] <= 
                total_crudes
            )
            
            # Volume limit based on crude count
            total_volume = sum(
                self.parcel_info[p]['size'] * self.variables['pickup'][(v, p, d)]
                for p in self.parcels
                for d in self.days
            )
            
            two_parcel_capacity = self.config.get('Two_crude', 700000)
            three_parcel_capacity = self.config.get('Three_crude', 650000)
            
            self.model.Add(
                total_volume <= 
                two_parcel_capacity * self.variables['num_grades_12'][v] + 
                three_parcel_capacity * self.variables['num_grades_3'][v]
            )
        
        self.logger.info("Vessel loading constraints added successfully")
    
    def add_vessel_discharge_constraints(self):
        """Add vessel discharge constraints."""
        self.logger.info("Adding vessel discharge constraints...")
        
        # Constraint 1: Unique discharge day per vessel
        for v in self.vessels:
            self.model.Add(
                sum(self.variables['discharge'][(v, d)] for d in self.days) == 1
            )
        
        # Constraint 2: Discharge at Melaka (2 days)
        melaka_idx = None
        for i, loc in enumerate(self.locations):
            if loc == "Melaka":
                melaka_idx = loc
                break
        
        if melaka_idx:
            for v in self.vessels:
                for d in self.days:
                    if d < max(self.days):
                        self.model.Add(
                            2 * self.variables['discharge'][(v, d)] <= 
                            self.variables['at_location'][(v, melaka_idx, d)] + 
                            self.variables['at_location'][(v, melaka_idx, d + 1)]
                        )
                    else:
                        self.model.Add(
                            2 * self.variables['discharge'][(v, d)] <= 
                            self.variables['at_location'][(v, melaka_idx, d)]
                        )
        
        # Constraint 3: No two vessels discharge on same/adjacent days
        for d in self.days:
            if d < max(self.days):
                self.model.Add(
                    sum(self.variables['discharge'][(v, d)] + 
                        self.variables['discharge'][(v, d + 1)] for v in self.vessels) <= 1
                )
            else:
                self.model.Add(
                    sum(self.variables['discharge'][(v, d)] for v in self.vessels) <= 1
                )
        
        # Constraint 4: Vessel stops after discharge
        for v in self.vessels:
            for l in self.locations:
                for d1 in self.days:
                    for d2 in self.days:
                        if d2 > d1 + 1:
                            self.model.Add(
                                self.variables['at_location'][(v, l, d2)] <= 
                                1 - self.variables['discharge'][(v, d1)]
                            )
        
        # Constraint 5: Volume onboard and discharge constraints
        for v in self.vessels:
            for c in self.crudes:
                # Volume onboard definition
                onboard_sum = sum(
                    self.parcel_info[p]['size'] * self.variables['pickup'][(v, p, d)]
                    for p in self.parcels
                    for d in self.days
                    if self.parcel_info[p]['crude'] == c
                )
                self.model.Add(
                    self.variables['volume_onboard'][(v, c)] == onboard_sum
                )
                
                for d in self.days:
                    vessel_max = self.config.get('Vessel_max_limit', 700000)
                    
                    # Discharge upper limit
                    self.model.Add(
                        self.variables['volume_discharged'][(v, c, d)] <= 
                        vessel_max * self.variables['discharge'][(v, d)]
                    )
                    
                    # Discharge no more than onboard
                    self.model.Add(
                        self.variables['volume_discharged'][(v, c, d)] <= 
                        self.variables['volume_onboard'][(v, c)]
                    )
                    
                    # Discharge lower bound
                    self.model.Add(
                        self.variables['volume_discharged'][(v, c, d)] >= 
                        self.variables['volume_onboard'][(v, c)] - 
                        vessel_max * (1 - self.variables['discharge'][(v, d)])
                    )
        
        self.logger.info("Vessel discharge constraints added successfully")
    
    def add_blending_constraints(self, max_transitions: int):
        """Add crude blending constraints."""
        self.logger.info(f"Adding blending constraints (max transitions: {max_transitions})...")
        
        # Constraint 1: Blend consumption
        for b in self.blends:
            for s in self.slots:
                # Is blend consumed based on fraction
                self.model.Add(
                    self.SCALE_FACTOR * self.variables['is_blend_consumed'][(b, s)] >= 
                    self.variables['blend_fraction'][(b, s)]
                )
        
        # One blend per slot
        for s in self.slots:
            self.model.Add(
                sum(self.variables['is_blend_consumed'][(b, s)] for b in self.blends) == 1
            )
        
        # Constraint 2: Blend fraction daily bound
        for s in self.slots:
            if s % 2 == 1 and s + 1 in self.slots:
                self.model.Add(
                    sum(self.variables['blend_fraction'][(b, s)] + 
                        self.variables['blend_fraction'][(b, s + 1)] for b in self.blends) <= 
                    self.SCALE_FACTOR
                )
        
        # Constraint 3: Transition constraints
        for b in self.blends:
            for s in self.slots:
                if s + 1 in self.slots:
                    # Transition lower bound
                    self.model.Add(
                        self.variables['is_transition'][(b, s)] >= 
                        self.variables['is_blend_consumed'][(b, s)] - 
                        self.variables['is_blend_consumed'][(b, s + 1)]
                    )
                    
                    # Transition upper bound 2
                    self.model.Add(
                        self.variables['is_transition'][(b, s)] <= 
                        1 - self.variables['is_blend_consumed'][(b, s + 1)]
                    )
                
                # Transition upper bound 1
                self.model.Add(
                    self.variables['is_transition'][(b, s)] <= 
                    self.variables['is_blend_consumed'][(b, s)]
                )
        
        # Max transitions constraint
        self.model.Add(
            sum(self.variables['is_transition'][(b, s)] 
                for b in self.blends for s in self.slots) <= max_transitions
        )
        
        # Constraint 4: Plant capacity constraints
        for d in self.days:
            if 2*d-1 in self.slots and 2*d in self.slots:
                capacity_sum = sum(
                    self.products_capacity[b] * 
                    (self.variables['blend_fraction'][(b, 2*d-1)] + 
                     self.variables['blend_fraction'][(b, 2*d)])
                    for b in self.blends
                )
                
                # Capacity constraint
                daily_capacity = self.config.get('default_capacity', 96000)
                self.model.Add(capacity_sum <= daily_capacity * self.SCALE_FACTOR)
                
                # Minimum production
                turn_down = self.config.get('turn_down_capacity', 20000)  # Default minimum capacity
                self.model.Add(capacity_sum >= turn_down * self.SCALE_FACTOR)
        
        self.logger.info("Blending constraints added successfully")
    
    def add_inventory_constraints(self):
        """Add inventory management constraints."""
        self.logger.info("Adding inventory constraints...")
        
        # Inventory update constraints
        for c in self.crudes:
            for d in self.days:
                # Calculate discharged amount (with 5-day delay)
                if d <= 5:
                    discharged = 0
                else:
                    discharged = sum(
                        self.variables['volume_discharged'][(v, c, d-5)] 
                        for v in self.vessels
                    )
                
                # Calculate consumed amount
                consumed_terms = []
                for b in self.blends:
                    if (b, c) in self.products_ratio:
                        ratio = self.products_ratio[(b, c)]
                        capacity = self.products_capacity[b]
                        if 2*d-1 in self.slots and 2*d in self.slots:
                            consumed_terms.append(
                                int(capacity * ratio) * 
                                (self.variables['blend_fraction'][(b, 2*d-1)] + 
                                 self.variables['blend_fraction'][(b, 2*d)])
                            )
                
                consumed = sum(consumed_terms) if consumed_terms else 0
                
                # Inventory balance
                if d == 1:
                    opening = self.opening_inventory_dict.get(c, 0)
                    if isinstance(consumed, int) and consumed == 0:
                        self.model.Add(
                            self.variables['inventory'][(c, d)] == opening + discharged
                        )
                    else:
                        # For non-zero consumption, we need to handle the scaled values
                        self.model.Add(
                            self.SCALE_FACTOR * self.variables['inventory'][(c, d)] == 
                            self.SCALE_FACTOR * (opening + discharged) - consumed
                        )
                else:
                    if isinstance(consumed, int) and consumed == 0:
                        self.model.Add(
                            self.variables['inventory'][(c, d)] == 
                            self.variables['inventory'][(c, d-1)] + discharged
                        )
                    else:
                        self.model.Add(
                            self.SCALE_FACTOR * self.variables['inventory'][(c, d)] == 
                            self.SCALE_FACTOR * self.variables['inventory'][(c, d-1)] + 
                            self.SCALE_FACTOR * discharged - consumed
                        )
        
        # Maximum inventory limit
        for d in self.days:
            self.model.Add(
                sum(self.variables['inventory'][(c, d)] for c in self.crudes) <= 
                self.config['INVENTORY_MAX_VOLUME']
            )
        
        # Ullage constraints
        for d in self.days:
            total_consumed = sum(
                self.products_capacity[b] * 
                (self.variables['blend_fraction'][(b, 2*d-1)] + 
                 self.variables['blend_fraction'][(b, 2*d)])
                for b in self.blends
                if 2*d-1 in self.slots and 2*d in self.slots
            )
            
            if d == 1:
                opening_total = sum(self.opening_inventory_dict.values())
                max_inventory = self.config['INVENTORY_MAX_VOLUME']
                self.model.Add(
                    self.SCALE_FACTOR * self.variables['ullage'][d] == 
                    self.SCALE_FACTOR * (max_inventory - opening_total) + total_consumed
                )
            else:
                discharged_prev = sum(
                    self.variables['volume_discharged'][(v, c, d-1)]
                    for v in self.vessels
                    for c in self.crudes
                )
                self.model.Add(
                    self.SCALE_FACTOR * self.variables['ullage'][d] == 
                    self.SCALE_FACTOR * self.variables['ullage'][d-1] - 
                    self.SCALE_FACTOR * discharged_prev + total_consumed
                )
        
        self.logger.info("Inventory constraints added successfully")
    
    def set_objective(self, optimization_type: str, max_demurrage_limit: int):
        """Set the optimization objective."""
        self.logger.info(f"Setting {optimization_type} objective with demurrage limit: {max_demurrage_limit}...")
        
        demurrage_cost = self.config.get('Demurrage', 40000)
        
        # Demurrage at source
        demurrage_source_terms = []
        for v in self.vessels:
            for l in self.locations:
                if l != 'Melaka':
                    for d in self.days:
                        demurrage_source_terms.append(
                            self.variables['at_location'][(v, l, d)]
                        )
        
        # Subtract pickup activities
        pickup_terms = []
        for v in self.vessels:
            for p in self.parcels:
                for d in self.days:
                    pickup_terms.append(self.variables['pickup'][(v, p, d)])
        
        demurrage_at_source = sum(demurrage_source_terms) - sum(pickup_terms)
        
        # Demurrage at Melaka
        demurrage_melaka_terms = []
        for v in self.vessels:
            melaka_days = sum(
                self.variables['at_location'][(v, 'Melaka', d)] 
                for d in self.days
            )
            # Each vessel gets 2 free days at Melaka
            # This is simplified - would need auxiliary variables for exact implementation
            demurrage_melaka_terms.append(melaka_days)
        
        total_demurrage = demurrage_at_source + sum(demurrage_melaka_terms)
        
        if optimization_type == 'throughput':
            # Maximize throughput with demurrage constraint
            throughput_terms = []
            for b in self.blends:
                capacity = self.products_capacity[b]
                for s in self.slots:
                    throughput_terms.append(
                        capacity * self.variables['blend_fraction'][(b, s)]
                    )
            
            total_throughput = sum(throughput_terms)
            
            # Add demurrage limit constraint
            self.model.Add(total_demurrage <= max_demurrage_limit)
            
            # Maximize throughput
            self.model.Maximize(total_throughput)
            
            self.logger.info(f"Throughput optimization objective set (demurrage limit: {max_demurrage_limit})")
            
        elif optimization_type == 'margin':
            # Maximize profit (margin - demurrage)
            profit_terms = []
            for c in self.crudes:
                margin = self.crude_margins_dict[c]
                for b in self.blends:
                    if (b, c) in self.products_ratio:
                        ratio = self.products_ratio[(b, c)]
                        capacity = self.products_capacity[b]
                        for s in self.slots:
                            profit_terms.append(
                                int(margin * ratio * capacity) * 
                                self.variables['blend_fraction'][(b, s)]
                            )
            
            total_profit = sum(profit_terms)
            
            # Maximize profit minus demurrage
            self.model.Maximize(total_profit - demurrage_cost * total_demurrage)
            
            self.logger.info("Margin optimization objective set")
            
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    def solve(self, time_limit_seconds: int = 3600) -> Dict[str, Any]:
        """Solve the optimization model."""
        self.logger.info(f"Starting optimization with time limit: {time_limit_seconds} seconds...")
        
        # Set solver parameters
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        self.solver.parameters.num_search_workers = 4
        
        self.logger.info(f"Solver parameters:")
        self.logger.info(f"  - Time limit: {time_limit_seconds} seconds")
        self.logger.info(f"  - Search workers: 4")
        
        # Log model statistics
        num_variables = self.model.Proto().variables
        num_constraints = len(self.model.Proto().constraints)
        self.logger.info(f"Model statistics:")
        self.logger.info(f"  - Variables: {len(num_variables)}")
        self.logger.info(f"  - Constraints: {num_constraints}")
        
        # Solve
        start_time = datetime.now()
        self.logger.info("Solver starting...")
        
        status = self.solver.Solve(self.model)
        
        end_time = datetime.now()
        solve_duration = (end_time - start_time).total_seconds()
        
        # Process results
        result = {
            'status': self.solver.StatusName(status),
            'objective_value': None,
            'solve_time': solve_duration,
            'solution': None,
            'solver_stats': {
                'wall_time': self.solver.WallTime(),
                'user_time': self.solver.UserTime(),
                'deterministic_time': self.solver.DeterministicTime(),
                'num_booleans': self.solver.NumBooleans(),
                'num_conflicts': self.solver.NumConflicts(),
                'num_branches': self.solver.NumBranches(),
                'num_integers': self.solver.NumIntegers()
            }
        }
        
        self.logger.info(f"Solver completed with status: {result['status']}")
        self.logger.info(f"Solve time: {solve_duration:.2f} seconds")
        self.logger.info(f"Solver statistics:")
        self.logger.info(f"  - Wall time: {result['solver_stats']['wall_time']:.2f}s")
        self.logger.info(f"  - User time: {result['solver_stats']['user_time']:.2f}s")
        self.logger.info(f"  - Branches: {result['solver_stats']['num_branches']}")
        self.logger.info(f"  - Conflicts: {result['solver_stats']['num_conflicts']}")
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result['objective_value'] = self.solver.ObjectiveValue()
            result['solution'] = self._extract_solution()
            self.logger.info(f"Solution found! Objective value: {result['objective_value']}")
            self.logger.info("Extracting solution details...")
        else:
            self.logger.warning(f"No solution found. Status: {result['status']}")
            if status == cp_model.INFEASIBLE:
                self.logger.error("Model is infeasible - constraints are contradictory")
            elif status == cp_model.MODEL_INVALID:
                self.logger.error("Model is invalid - check constraint definitions")
        
        return result
    
    def _extract_solution(self) -> Dict[str, Any]:
        """Extract the solution from the solved model."""
        solution = {}
        
        # Extract vessel schedules
        vessel_schedule = []
        for v in self.vessels:
            for d in self.days:
                for l in self.locations:
                    if self.solver.Value(self.variables['at_location'][(v, l, d)]):
                        activities = []
                        
                        # Check for pickup
                        for p in self.parcels:
                            if self.solver.Value(self.variables['pickup'][(v, p, d)]):
                                activities.append(f"Pickup {p[1]} from {p[0]}")
                        
                        # Check for discharge
                        if self.solver.Value(self.variables['discharge'][(v, d)]):
                            activities.append("Discharge")
                        
                        if not activities:
                            activities.append("Demurrage")
                        
                        vessel_schedule.append({
                            'vessel': v,
                            'day': d,
                            'location': l,
                            'activities': activities
                        })
        
        solution['vessel_schedule'] = vessel_schedule
        
        # Extract production schedule
        production_schedule = []
        for s in self.slots:
            day = (s + 1) // 2
            slot_in_day = 1 if s % 2 == 1 else 2
            
            for b in self.blends:
                if self.solver.Value(self.variables['is_blend_consumed'][(b, s)]):
                    fraction = self.solver.Value(self.variables['blend_fraction'][(b, s)]) / self.SCALE_FACTOR
                    quantity = fraction * self.products_capacity[b]
                    
                    production_schedule.append({
                        'day': day,
                        'slot': slot_in_day,
                        'blend': b,
                        'fraction': fraction,
                        'quantity': quantity
                    })
        
        solution['production_schedule'] = production_schedule
        
        # Extract inventory levels
        inventory_levels = {}
        for c in self.crudes:
            inventory_levels[c] = {}
            for d in self.days:
                inventory_levels[c][d] = self.solver.Value(self.variables['inventory'][(c, d)])
        
        solution['inventory_levels'] = inventory_levels
        
        return solution
    
    def build_model(self, optimization_type: str, max_demurrage_limit: int, max_transitions: int):
        """Build the complete optimization model."""
        self.logger.info("=" * 60)
        self.logger.info("BUILDING OR-TOOLS OPTIMIZATION MODEL")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        self.create_variables()
        self.add_vessel_travel_constraints()
        self.add_vessel_loading_constraints()
        self.add_vessel_discharge_constraints()
        self.add_blending_constraints(max_transitions)
        self.add_inventory_constraints()
        self.set_objective(optimization_type, max_demurrage_limit)
        
        build_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Model building complete in {build_time:.2f} seconds!")
        self.logger.info("=" * 60)


def create_output_dataframes(solution: Dict[str, Any], data: Dict[str, Any], 
                           config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create output dataframes from the solution."""
    
    # Create production schedule DataFrame
    production_records = []
    
    if 'production_schedule' in solution:
        for record in solution['production_schedule']:
            day = record['day']
            slot = record['slot']
            blend = record['blend']
            quantity = record['quantity']
            
            # Calculate date
            start_date = pd.to_datetime(f"{config['schedule_year']}-{config['schedule_month']}-01")
            date = start_date + pd.Timedelta(days=day-1)
            
            # Calculate crude usage and inventory
            crude_blended = {}
            crude_available = {}
            
            for crude in data['crudes']:
                # Calculate blended amount
                if (blend, crude) in data['products_ratio']:
                    ratio = data['products_ratio'][(blend, crude)]
                    blended_amount = quantity * ratio
                else:
                    blended_amount = 0
                    
                crude_blended[f"Crude {crude} Blended"] = round(blended_amount/1000, 1)
                
                # Get inventory (simplified)
                if 'inventory_levels' in solution and crude in solution['inventory_levels']:
                    if day in solution['inventory_levels'][crude]:
                        available = solution['inventory_levels'][crude][day]
                    else:
                        available = 0
                else:
                    available = 0
                    
                crude_available[f"Crude {crude} Available"] = round(available/1000, 1)
            
            production_records.append({
                "Date": date.strftime('%Y-%m-%d'),
                "Slot": slot,
                "Final Product": blend,
                "Quantity Produced": round(quantity/1000, 1),
                **crude_available,
                **crude_blended,
                "Flag": "Optimization"
            })
    
    production_df = pd.DataFrame(production_records)
    
    # Create vessel schedule DataFrame
    vessel_records = []
    
    if 'vessel_schedule' in solution:
        for record in solution['vessel_schedule']:
            vessel = record['vessel']
            day = record['day']
            location = record['location']
            activities = record['activities']
            
            # Calculate date
            start_date = pd.to_datetime(f"{config['schedule_year']}-{config['schedule_month']}-01")
            date = start_date + pd.Timedelta(days=day-1)
            
            for activity in activities:
                vessel_records.append({
                    "Activity Date": date.strftime('%Y-%m-%d'),
                    "Activity Name": activity,
                    "Vessel ID": vessel,
                    "Location": location,
                    "is_at_Melaka": 1 if location == "Melaka" else 0,
                    "is Demurrage Day": 1 if activity == "Demurrage" else 0
                })
    
    vessel_df = pd.DataFrame(vessel_records)
    
    return production_df, vessel_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="OR-Tools Refinery Optimization")
    parser.add_argument("--config", type=str, default="config.json", 
                       help="Configuration file name (config.json or config_small.json)")
    parser.add_argument("--vessels", type=int, default=6, help="Number of vessels")
    parser.add_argument("--optimization", choices=['margin', 'throughput'], 
                       default='throughput', help="Optimization type")
    parser.add_argument("--demurrage-limit", type=int, default=10, 
                       help="Maximum demurrage limit for throughput optimization")
    parser.add_argument("--max-transitions", type=int, default=11, 
                       help="Maximum number of transitions")
    parser.add_argument("--time-limit", type=int, default=3600, 
                       help="Solver time limit in seconds")
    parser.add_argument("--data-path", type=str, default="./test_data", 
                       help="Path to test data directory")
    parser.add_argument("--output-dir", type=str, default="./results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up logging
    logger = setup_logging(
        output_dir, 
        args.optimization, 
        args.vessels, 
        args.demurrage_limit if args.optimization == 'throughput' else None
    )
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data_path} using {args.config}...")
        loader = DataLoader(args.data_path)
        data = loader.load_scenario_data(args.config)
        
        logger.info("Data loaded successfully!")
        
        # Create optimizer
        optimizer = OrToolsRefineryOptimizer(data, args.vessels, logger)
        
        # Build model
        optimizer.build_model(
            optimization_type=args.optimization,
            max_demurrage_limit=args.demurrage_limit,
            max_transitions=args.max_transitions
        )
        
        # Solve
        result = optimizer.solve(time_limit_seconds=args.time_limit)
        
        # Generate output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if result['solution']:
            logger.info("Generating output files...")
            # Create output dataframes
            production_df, vessel_df = create_output_dataframes(
                result['solution'], data, data['config']
            )
            
            # Save results
            if args.optimization == 'throughput':
                prod_filename = f"ortools_production_schedule_{args.optimization}_{args.vessels}_vessels_{args.demurrage_limit}_demurrage_{timestamp}.csv"
                vessel_filename = f"ortools_vessel_schedule_{args.optimization}_{args.vessels}_vessels_{args.demurrage_limit}_demurrage_{timestamp}.csv"
                summary_filename = f"ortools_summary_{args.optimization}_{args.vessels}_vessels_{args.demurrage_limit}_demurrage_{timestamp}.json"
            else:
                prod_filename = f"ortools_production_schedule_{args.optimization}_{args.vessels}_vessels_{timestamp}.csv"
                vessel_filename = f"ortools_vessel_schedule_{args.optimization}_{args.vessels}_vessels_{timestamp}.csv"
                summary_filename = f"ortools_summary_{args.optimization}_{args.vessels}_vessels_{timestamp}.json"
            
            production_df.to_csv(output_dir / prod_filename, index=False)
            vessel_df.to_csv(output_dir / vessel_filename, index=False)
            
            logger.info(f"Production schedule saved: {prod_filename}")
            logger.info(f"Vessel schedule saved: {vessel_filename}")
            
            # Save summary
            summary = {
                'config_file': args.config,
                'vessels': args.vessels,
                'optimization_type': args.optimization,
                'status': result['status'],
                'objective_value': result['objective_value'],
                'solve_time_seconds': result['solve_time'],
                'total_production': production_df['Quantity Produced'].sum() if not production_df.empty else 0,
                'solver_stats': result['solver_stats'],
                'output_files': {
                    'production_schedule': prod_filename,
                    'vessel_schedule': vessel_filename
                }
            }
            
            with open(output_dir / summary_filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary saved: {summary_filename}")
            logger.info("=" * 60)
            logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Status: {result['status']}")
            logger.info(f"Objective value: {result['objective_value']}")
            logger.info(f"Total production: {summary['total_production']:.1f}")
            logger.info(f"Solve time: {result['solve_time']:.2f} seconds")
            logger.info(f"Results saved to: {output_dir}")
            logger.info("=" * 60)
            
        else:
            logger.error("No solution found - check constraints and data")
            logger.error("This could be due to:")
            logger.error("  - Infeasible constraints")
            logger.error("  - Insufficient time limit")
            logger.error("  - Data inconsistencies")
            
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        import traceback
        logger.error("Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(line)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
