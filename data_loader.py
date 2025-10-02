"""
Data loading and preprocessing module for vessel routing optimization.
Handles loading of scenario data, configuration, and data transformations.
"""

import pandas as pd
import json
import ast
from typing import Dict, List, Tuple, Any, Optional


class DataLoader:
    """Handles loading and preprocessing of optimization data."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize DataLoader with base path for data files.
        
        Args:
            base_path: Base directory path for data files. If None, uses test_data/
        """
        self.base_path = base_path or "test_data/"
        if not self.base_path.endswith("/"):
            self.base_path += "/"
    
    def load_scenario_data(self) -> Tuple[Dict, List, List, Dict, Dict, List, pd.DataFrame, List, Dict]:
        """
        Load all scenario data including configuration, crude info, and availability.
        
        Returns:
            Tuple containing:
            - config: Configuration dictionary
            - crudes: List of crude types
            - locations: List of locations
            - time_of_travel: Travel time dictionary
            - crude_availability: Crude availability dictionary
            - source_location: List of source locations
            - products_info: Products information DataFrame
            - crude_margins: List of crude margins
            - opening_inventory_dict: Opening inventory dictionary
        """
        # Load configuration
        with open(f"{self.base_path}config.json", "r") as f:
            config = json.load(f)

        # Load crude availability
        crude_availability_df = pd.read_csv(f"{self.base_path}crude_availability.csv")
        crude_availability = {}
        for _, row in crude_availability_df.iterrows():
            crude_availability \
                .setdefault(row["date_range"], {}) \
                .setdefault(row["location"], {})[row["crude"]] = {
                    "volume": int(row["volume"]),
                    "parcel_size": int(row["parcel_size"])
                }
        
        # Load time of travel
        time_of_travel_df = pd.read_csv(f"{self.base_path}time_of_travel.csv")
        time_of_travel = {
            (row["from"], row["to"]): int(row["time_in_days"]) + 1
            for _, row in time_of_travel_df.iterrows()
        }
        
        # Load products and crudes info
        products_info = pd.read_csv(f"{self.base_path}products_info.csv")
        crudes_info_df = pd.read_csv(f"{self.base_path}crudes_info.csv")
        
        crudes = crudes_info_df["crudes"].tolist()
        locations = list(set(time_of_travel_df["from"]) | set(time_of_travel_df["to"]))
        source_location = crudes_info_df["origin"].tolist()
        crude_margins = crudes_info_df['margin'].tolist()
        
        opening_inventory = crudes_info_df['opening_inventory'].tolist()
        opening_inventory_dict = dict(zip(crudes, opening_inventory))

        return (config, crudes, locations, time_of_travel, crude_availability, 
                source_location, products_info, crude_margins, opening_inventory_dict)
    
    def extract_window_to_days(self, crude_availability: Dict) -> Dict[str, List[int]]:
        """
        Extract days from availability windows.
        
        Args:
            crude_availability: Crude availability dictionary
            
        Returns:
            Dictionary mapping windows to list of days
        """
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
    
    def extract_products_ratio(self, products_info: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Extract product ratios from products info DataFrame.
        
        Args:
            products_info: Products information DataFrame
            
        Returns:
            Dictionary mapping (product, crude) to ratio
        """
        return {
            (row['product'], crude): ratio
            for _, row in products_info.iterrows()
            for crude, ratio in zip(ast.literal_eval(row['crudes']), ast.literal_eval(row['ratios']))
        }
    
    def get_date_mapping(self, config: Dict) -> Tuple[int, int, pd.Timestamp]:
        """
        Get date mapping information from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (month_number, year, start_date)
        """
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        month_number = month_map[config["schedule_month"]]
        year = config["schedule_year"]
        start_date = pd.to_datetime(f"{year}-{month_number:02d}-01")
        
        return month_number, year, start_date
    
    def prepare_capacity_dict(self, config: Dict) -> Dict[int, int]:
        """
        Prepare plant capacity dictionary from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary mapping days to capacity values
        """
        days = list(range(config["DAYS"]["start"], config["DAYS"]["end"] + 1))
        capacity_dict = {}

        for entry in config['plant_capacity_reduction_window']:
            cap = entry['max_capacity']
            start = entry['start_date']
            end = entry['end_date']
            for day in range(start, end + 1):
                capacity_dict[day] = cap

        default_capacity = config['default_capacity']
        for day in days:
            capacity_dict.setdefault(day, default_capacity)

        return capacity_dict


def load_all_data(base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load all required data.
    
    Args:
        base_path: Base directory path for data files
        
    Returns:
        Dictionary containing all loaded data
    """
    loader = DataLoader(base_path)
    
    (config, crudes, locations, time_of_travel, crude_availability, 
     source_location, products_info, crude_margins, opening_inventory_dict) = loader.load_scenario_data()
    
    window_to_days = loader.extract_window_to_days(crude_availability)
    products_ratio = loader.extract_products_ratio(products_info)
    month_number, year, start_date = loader.get_date_mapping(config)
    capacity_dict = loader.prepare_capacity_dict(config)
    
    return {
        'config': config,
        'crudes': crudes,
        'locations': locations,
        'time_of_travel': time_of_travel,
        'crude_availability': crude_availability,
        'source_location': source_location,
        'products_info': products_info,
        'crude_margins': crude_margins,
        'opening_inventory_dict': opening_inventory_dict,
        'window_to_days': window_to_days,
        'products_ratio': products_ratio,
        'month_number': month_number,
        'year': year,
        'start_date': start_date,
        'capacity_dict': capacity_dict
    }