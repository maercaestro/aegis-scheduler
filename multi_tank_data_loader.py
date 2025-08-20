"""
Data loader utilities for the 5-tank refinery and vessel optimization system.
Handles loading and preprocessing of input data from CSV and JSON files.
Modified to support multiple tank configuration.
"""

import pandas as pd
import json
import ast
from pathlib import Path


class MultiTankDataLoader:
    """Handles loading and preprocessing of optimization data for multi-tank system."""
    
    def __init__(self, data_path: str = "test_data"):
        """
        Initialize DataLoader with path to data directory.
        
        Args:
            data_path: Path to directory containing input data files
        """
        self.data_path = Path(data_path)
        
    def load_config(self) -> dict:
        """Load configuration from config.json and add tank configuration."""
        config_path = self.data_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Process config to match notebook expectations
        config["INVENTORY_MAX_VOLUME"] = config.get("INVENTORY_MAX_VOLUME", 1180000)
        config["MaxTransitions"] = config.get("MaxTransitions", 11)
        config["two_parcel_vessel_capacity"] = config.get("Two_crude", 700000)
        config["three_parcel_vessel_capacity"] = config.get("Three_crude", 650000)
        config["demurrage_cost"] = config.get("Demurrage", 40000)
        config["vessel_max_limit"] = config.get("Vessel_max_limit", 700000)
        
        # Add tank configuration
        config["TANKS"] = {
            "Tank1": {"capacity": 250000, "allowed_crudes": []},  # Will be set dynamically
            "Tank2": {"capacity": 250000, "allowed_crudes": []},
            "Tank3": {"capacity": 250000, "allowed_crudes": []},
            "Tank4": {"capacity": 250000, "allowed_crudes": []},
            "Tank5": {"capacity": 180000, "allowed_crudes": []}
        }
        
        # Total capacity check
        total_tank_capacity = sum(tank["capacity"] for tank in config["TANKS"].values())
        print(f"Total tank capacity: {total_tank_capacity:,} barrels")
        print(f"Original single tank capacity: {config['INVENTORY_MAX_VOLUME']:,} barrels")
        
        # Process capacity reduction windows
        config["plant_capacity_reduction_window"] = []
        if "Range" in config:
            for entry in config["Range"]:
                config["plant_capacity_reduction_window"].append({
                    "max_capacity": entry["capacity"],
                    "start_date": entry["start_date"],
                    "end_date": entry["end_date"]
                })
        
        # Set turn down capacity (minimum production)
        config["turn_down_capacity"] = config.get("turn_down_capacity", 30000)
        
        # Set solver time limit
        config["solver_time_limit_seconds"] = 3600
        for solver in config.get("solver", []):
            if solver.get("use", False):
                config["solver_time_limit_seconds"] = solver.get("time_limit", 3600)
                break
        
        return config
    
    def assign_crudes_to_tanks(self, crudes: list, config: dict) -> dict:
        """
        Assign crude types to tanks using a smart capacity-aware strategy with splitting.
        Large crude inventories are automatically split across multiple tanks to ensure feasible assignments.
        """
        # Get opening inventory data
        crudes_info_df = self.load_crudes_info()
        opening_inventory = dict(zip(crudes_info_df["crudes"], crudes_info_df["opening_inventory"]))
        
        # Sort crudes by opening inventory (largest first)
        crudes_by_inventory = sorted(crudes, key=lambda c: opening_inventory.get(c, 0), reverse=True)
        
        # Initialize tank assignments and current usage
        tank_assignments = {}  # crude -> primary tank
        crude_splits = {}      # crude -> {tank: amount} for split assignments
        tanks = list(config["TANKS"].keys())
        tank_usage = {tank: 0 for tank in tanks}
        tank_capacities = {tank: config["TANKS"][tank]["capacity"] for tank in tanks}
        
        print(f"\nSmart Tank Assignment with Crude Splitting:")
        print(f"Opening Inventory: {opening_inventory}")
        
        # Check total capacity
        total_capacity = sum(tank_capacities.values())
        total_inventory = sum(opening_inventory.values())
        
        if total_inventory > total_capacity:
            print(f"WARNING: Total opening inventory ({total_inventory:,}) exceeds total tank capacity ({total_capacity:,})")
            print("Even with splitting, this may cause infeasibility.")
            return {}
        
        # Advanced assignment with crude splitting
        for crude in crudes_by_inventory:
            crude_inventory = opening_inventory.get(crude, 0)
            
            if crude_inventory == 0:
                # Zero inventory crudes - assign to tank with most space
                available_space = {tank: tank_capacities[tank] - tank_usage[tank] for tank in tanks}
                best_tank = max(tanks, key=lambda t: available_space[t])
                tank_assignments[crude] = best_tank
                config["TANKS"][best_tank]["allowed_crudes"].append(crude)
                print(f"   {crude} (0 KB) -> {best_tank} (zero inventory)")
                continue
            
            # Try to fit in a single tank first
            available_space = {tank: tank_capacities[tank] - tank_usage[tank] for tank in tanks}
            best_tank = max(tanks, key=lambda t: available_space[t])
            
            if crude_inventory <= available_space[best_tank]:
                # Fits in single tank
                tank_assignments[crude] = best_tank
                tank_usage[best_tank] += crude_inventory
                config["TANKS"][best_tank]["allowed_crudes"].append(crude)
                print(f"   {crude} ({crude_inventory:,} KB) -> {best_tank} (single tank)")
            else:
                # Need to split across multiple tanks
                print(f"   {crude} ({crude_inventory:,} KB) - SPLITTING:")
                remaining_inventory = crude_inventory
                crude_splits[crude] = {}
                tanks_used = []
                
                # Sort tanks by available space (most space first)
                sorted_tanks = sorted(tanks, key=lambda t: available_space[t], reverse=True)
                
                for tank in sorted_tanks:
                    if remaining_inventory <= 0:
                        break
                    
                    space_available = available_space[tank]
                    if space_available <= 0:
                        continue
                    
                    # Allocate as much as possible to this tank
                    allocation = min(remaining_inventory, space_available)
                    if allocation > 0:
                        crude_splits[crude][tank] = allocation
                        tank_usage[tank] += allocation
                        remaining_inventory -= allocation
                        tanks_used.append(tank)
                        config["TANKS"][tank]["allowed_crudes"].append(crude)
                        print(f"     -> {tank}: {allocation:,} KB (usage: {tank_usage[tank]:,}/{tank_capacities[tank]:,} KB)")
                        
                        # Update available space for next iteration
                        available_space[tank] = tank_capacities[tank] - tank_usage[tank]
                
                if remaining_inventory > 0:
                    print(f"     WARNING: Could not allocate {remaining_inventory:,} KB - insufficient tank capacity!")
                    
                # Set primary tank to the one with largest allocation
                if crude_splits[crude]:
                    primary_tank = max(crude_splits[crude].keys(), key=lambda t: crude_splits[crude][t])
                    tank_assignments[crude] = primary_tank
                    print(f"     Primary tank: {primary_tank}")
        
        # Store crude splits in config for model building
        config["CRUDE_SPLITS"] = crude_splits
        
        print("\nFinal Tank Assignments:")
        total_used = 0
        feasible = True
        for tank, tank_info in config["TANKS"].items():
            crudes_list = tank_info["allowed_crudes"]
            capacity = tank_info["capacity"]
            used = tank_usage[tank]
            utilization = (used / capacity) * 100 if capacity > 0 else 0
            total_used += used
            status = "OK" if used <= capacity else "OVERFLOW"
            if used > capacity:
                feasible = False
            print(f"  {tank} ({capacity:,} KB): {crudes_list} - {status}")
            print(f"    Usage: {used:,} KB ({utilization:.1f}% utilized)")
        
        print(f"\nTotal Opening Inventory: {total_used:,} KB")
        print(f"Total Tank Capacity: {sum(tank_capacities.values()):,} KB")
        print(f"Overall Utilization: {(total_used / sum(tank_capacities.values())) * 100:.1f}%")
        
        if feasible:
            print("\n✅ Assignment is feasible with crude splitting!")
        else:
            print("\n❌ Assignment is still not feasible!")
        
        # Print crude split summary
        if crude_splits:
            print(f"\nCrude Split Summary:")
            for crude, splits in crude_splits.items():
                total_split = sum(splits.values())
                print(f"  {crude} ({total_split:,} KB): {dict(splits)}")
        
        return tank_assignments
    
    def load_crude_availability(self) -> dict:
        """Load crude availability data from CSV and convert to nested dict format."""
        df = pd.read_csv(self.data_path / "crude_availability.csv")
        crude_availability = {}
        
        for _, row in df.iterrows():
            crude_availability \
                .setdefault(row["date_range"], {}) \
                .setdefault(row["location"], {})[row["crude"]] = {
                    "volume": int(row["volume"]),
                    "parcel_size": int(row["parcel_size"])
                }
        
        return crude_availability
    
    def load_time_of_travel(self) -> dict:
        """Load time of travel data from CSV."""
        df = pd.read_csv(self.data_path / "time_of_travel.csv")
        return {
            (row["from"], row["to"]): int(row["time_in_days"]) + 1
            for _, row in df.iterrows()
        }
    
    def load_products_info(self) -> pd.DataFrame:
        """Load products information from CSV."""
        return pd.read_csv(self.data_path / "products_info.csv")
    
    def load_crudes_info(self) -> pd.DataFrame:
        """Load crudes information from CSV."""
        return pd.read_csv(self.data_path / "crudes_info.csv")
    
    def extract_products_ratio(self, products_df: pd.DataFrame) -> dict:
        """Convert products DataFrame to ratio dictionary."""
        return {
            (row['product'], crude): ratio
            for _, row in products_df.iterrows()
            for crude, ratio in zip(ast.literal_eval(row['crudes']), ast.literal_eval(row['ratios']))
        }
    
    def extract_window_to_days(self, crude_availability: dict) -> dict:
        """Extract day ranges from window strings."""
        window_to_days = {}
        
        for window in crude_availability:
            # Split the date range and take only the day parts (ignore month)
            parts = window.split()[0]  # e.g., "1-3"
            if '-' in parts:
                start_day, end_day = map(int, parts.split('-'))
                days = list(range(start_day, end_day + 1))
            else:
                days = [int(parts)]
            window_to_days[window] = days
        
        return window_to_days
    
    def load_all_data(self, config: dict = None) -> dict:
        """
        Load all optimization data for multi-tank system.
        
        Args:
            config: Optional configuration dict to use instead of loading from file
        
        Returns:
            Dictionary containing all loaded data
        """
        # Load basic data
        if config is None:
            config = self.load_config()
        else:
            # Add tank configuration if not present
            if "TANKS" not in config:
                config["TANKS"] = {
                    "Tank1": {"capacity": 250000, "allowed_crudes": []},
                    "Tank2": {"capacity": 250000, "allowed_crudes": []},
                    "Tank3": {"capacity": 250000, "allowed_crudes": []},
                    "Tank4": {"capacity": 250000, "allowed_crudes": []},
                    "Tank5": {"capacity": 180000, "allowed_crudes": []}
                }
            else:
                # Ensure all tanks have allowed_crudes list
                for tank in config["TANKS"]:
                    if "allowed_crudes" not in config["TANKS"][tank]:
                        config["TANKS"][tank]["allowed_crudes"] = []
        
        crude_availability = self.load_crude_availability()
        time_of_travel = self.load_time_of_travel()
        products_info = self.load_products_info()
        crudes_info_df = self.load_crudes_info()
        
        # Extract derived data
        crudes = crudes_info_df["crudes"].tolist()
        locations = list(set(
            key[0] for key in time_of_travel.keys()
        ) | set(
            key[1] for key in time_of_travel.keys()
        ))
        source_locations = crudes_info_df["origin"].tolist()
        crude_margins = crudes_info_df['margin'].tolist()
        opening_inventory = crudes_info_df['opening_inventory'].tolist()
        opening_inventory_dict = dict(zip(crudes, opening_inventory))
        
        # Assign crudes to tanks
        tank_assignments = self.assign_crudes_to_tanks(crudes, config)
        
        # Extract products ratio
        products_ratio = self.extract_products_ratio(products_info)
        
        # Extract window to days mapping
        window_to_days = self.extract_window_to_days(crude_availability)
        
        return {
            "config": config,
            "crudes": crudes,
            "locations": locations,
            "time_of_travel": time_of_travel,
            "crude_availability": crude_availability,
            "source_locations": source_locations,
            "products_info": products_info,
            "crude_margins": crude_margins,
            "opening_inventory_dict": opening_inventory_dict,
            "products_ratio": products_ratio,
            "window_to_days": window_to_days,
            "tank_assignments": tank_assignments
        }
