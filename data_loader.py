"""
Data loader utilities for the refinery and vessel optimization system.
Handles loading and preprocessing of input data from CSV and JSON files.
"""

import pandas as pd
import json
import ast
from pathlib import Path


class DataLoader:
    """Handles loading and preprocessing of optimization data."""
    
    def __init__(self, data_path: str = "test_data"):
        """
        Initialize DataLoader with path to data directory.
        
        Args:
            data_path: Path to directory containing input data files
        """
        self.data_path = Path(data_path)
        
    def load_config(self) -> dict:
        """Load configuration from config.json."""
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
    
    def load_all_data(self) -> tuple:
        """
        Load all optimization data.
        
        Returns:
            Tuple containing:
            (config, crudes, locations, time_of_travel, crude_availability, 
             source_locations, products_info, crude_margins, opening_inventory_dict,
             products_ratio, window_to_days)
        """
        # Load basic data
        config = self.load_config()
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
        
        # Extract products ratio
        products_ratio = self.extract_products_ratio(products_info)
        
        # Extract window to days mapping
        window_to_days = self.extract_window_to_days(crude_availability)
        
        return (config, crudes, locations, time_of_travel, crude_availability, 
                source_locations, products_info, crude_margins, opening_inventory_dict,
                products_ratio, window_to_days)
