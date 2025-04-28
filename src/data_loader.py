import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pandas as pd

# Use absolute imports instead of relative imports
from src.models.tank import Tank
from src.models.delivery import Delivery
from src.models.blend import BlendRecipe
from src.models.inventory import CrudeInventory


class ConfigValidationError(Exception):
    """Exception raised for errors in the configuration file."""
    pass

# Configure logger
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and validating configuration and input data for the scheduler.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data loader.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration as a dictionary
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Validate the configuration
            self._validate_config()
            
            return self.config
        except FileNotFoundError:
            raise ConfigValidationError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ConfigValidationError(f"Invalid JSON in configuration file: {config_path}")
    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.
        
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        # Check required top-level keys
        required_keys = ['tanks', 'planning_horizon_days']
        for key in required_keys:
            if key not in self.config:
                raise ConfigValidationError(f"Missing required configuration: '{key}'")
        
        # Check if tanks have valid properties
        if not isinstance(self.config['tanks'], dict):
            raise ConfigValidationError("'tanks' must be a dictionary")
        
        for tank_id, tank_config in self.config['tanks'].items():
            if 'capacity' not in tank_config:
                raise ConfigValidationError(f"Tank '{tank_id}' missing required 'capacity' property")
            
            try:
                float(tank_config['capacity'])
            except (ValueError, TypeError):
                raise ConfigValidationError(f"Tank '{tank_id}' capacity must be a number")
        
        # Check planning horizon
        try:
            planning_horizon = int(self.config['planning_horizon_days'])
            if planning_horizon <= 0:
                raise ConfigValidationError("'planning_horizon_days' must be positive")
        except (ValueError, TypeError):
            raise ConfigValidationError("'planning_horizon_days' must be an integer")
    
    def get_tanks(self) -> List[Tank]:
        """
        Get initialized tank objects from the configuration.
        Respects tank capacity limits and converts excess to deliveries for days 1 and 2.
        
        Returns:
            List of Tank objects
        """
        initial_tanks = []
        tanks_cfg = self.config.get("tanks", {})
        init_alloc = self.config.get("initial_tank_allocation", {})
        excess_deliveries = []
        
        for tid, cfg in tanks_cfg.items():
            alloc = init_alloc.get(tid, {"crude": None, "level": 0.0})
            crude = alloc.get("crude")
            requested_level = float(alloc.get("level", 0.0))
            capacity = float(cfg.get("capacity", 0.0))
            
            # Check if the allocation exceeds tank capacity
            if crude and requested_level > capacity:
                excess = requested_level - capacity
                level = capacity  # Set to maximum capacity
                
                # Split excess into two deliveries (day 1 and day 2)
                day1_volume = excess / 2
                day2_volume = excess - day1_volume
                
                # Add excess as deliveries for days 1 and 2
                if day1_volume > 0:
                    excess_deliveries.append({"day": 1, "crude": crude, "volume": day1_volume})
                if day2_volume > 0:
                    excess_deliveries.append({"day": 2, "crude": crude, "volume": day2_volume})
                
                logger.warning(
                    f"Tank {tid} allocation exceeds capacity: requested {requested_level} kb of {crude}, "
                    f"but capacity is {capacity} kb. Excess {excess} kb split into deliveries for days 1-2."
                )
            else:
                # No excess, use requested level
                level = requested_level
            
            # Create the tank with the adjusted level
            tank = Tank(
                tank_id=tid,
                capacity=capacity,
                crude=crude,
                level=level if level > 0 else 0.0,
            )
            initial_tanks.append(tank)
        
        # Add excess deliveries to the configuration
        if excess_deliveries:
            current_deliveries = self.config.get("deliveries", [])
            self.config["deliveries"] = current_deliveries + excess_deliveries
            logger.info(f"Added {len(excess_deliveries)} excess deliveries for days 1-2")
        
        return initial_tanks
    
    def get_inventory(self, tanks: Optional[List[Tank]] = None) -> CrudeInventory:
        """
        Get inventory object initialized from tanks.
        
        Args:
            tanks: Optional list of tanks to initialize from
            
        Returns:
            CrudeInventory object
        """
        inventory = CrudeInventory({})
        
        if tanks:
            inventory.sync_with_tanks({t.id: t for t in tanks})
        
        return inventory
    
    def get_deliveries(self) -> Tuple[List[Delivery], Dict[int, List[Delivery]]]:
        """
        Get delivery objects from the configuration.
        
        Returns:
            Tuple of (all deliveries, deliveries by day dictionary)
        """
        raw_deliv = self.config.get("deliveries", [])
        deliveries = [Delivery.from_dict(d) for d in raw_deliv]
        
        # Organize by day
        future_deliveries = {}
        for d in deliveries:
            future_deliveries.setdefault(d.day, []).append(d)
        
        return deliveries, future_deliveries
    
    def get_blends(self) -> List[BlendRecipe]:
        """
        Get blend recipe objects from the configuration.
        
        Returns:
            List of BlendRecipe objects
        """
        pairing = self.config.get("pairing_and_blending", [])
        blends = []
        
        for item in pairing:
            ratio = item.get("ratio", {})
            # Convert from bpd to kb (1000 barrels)
            cap = item.get("capacity_bpd", 0.0) / 1000.0
            blends.append(BlendRecipe(ratios=ratio, max_capacity=cap))
        
        return blends

    def load_blend_recipes(self) -> List[BlendRecipe]:
        """
        Load blend recipes from configuration.
        Alias for get_blends() to maintain API compatibility.
        
        Returns:
            List of BlendRecipe objects
        """
        return self.get_blends()
        
    def get_scheduler_params(self) -> Dict[str, Any]:
        """
        Get scheduler parameters from the configuration.
        
        Returns:
            Dictionary of scheduler parameters
        """
        return {
            "planning_horizon": int(self.config.get("planning_horizon_days", 40)),
            "lookahead": int(self.config.get("look_ahead_days", 7)),
            "smoothing": float(self.config.get("smoothing_factor", 0.85)),
            "critical": int(self.config.get("critical_coverage_days", 8)),
            "final_days": int(self.config.get("final_days_threshold", 10)),
            "absolute_min": float(self.config.get("absolute_min_daily_processing_kb", 15.0)),
            "min_daily": float(self.config.get("min_daily_processing_kb", 50.0)),
            "increment": float(self.config.get("processing_increment_kb", 5.0)),
            "daily_capacity": float(self.config.get("daily_capacity_kb", 100.0)),
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            The loaded configuration dictionary
        """
        return self.config
    
    def load_vessel_data(self, path: str = None) -> Dict[str, Any]:
        """
        Load vessel data from a separate file.
        
        Args:
            path: Optional path to vessel data file (if None, uses default path)
            
        Returns:
            Dictionary with vessel optimization data
        """
        if path is None:
            # Use default path relative to config file
            config_dir = os.path.dirname(self.config_path)
            path = os.path.join(config_dir, "vessel_data.json")
        
        try:
            with open(path, 'r') as f:
                vessel_data = json.load(f)
            return vessel_data
        except FileNotFoundError:
            raise ConfigValidationError(f"Vessel data file not found: {path}")
        except json.JSONDecodeError:
            raise ConfigValidationError(f"Invalid JSON in vessel data file: {path}")
    
    def save_results(self, data: Dict[str, Any], output_dir: str = '.') -> Dict[str, str]:
        """
        Save results to output files.
        
        Args:
            data: Results data
            output_dir: Directory to save files
            
        Returns:
            Dictionary with file paths
        """
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert data to DataFrames if needed
        if isinstance(data.get('schedule'), list):
            schedule_df = pd.DataFrame(data['schedule'])
        elif isinstance(data.get('schedule'), pd.DataFrame):
            schedule_df = data['schedule']
        else:
            schedule_df = pd.DataFrame()
        
        # Save to CSV and JSON
        paths = {}
        
        if not schedule_df.empty:
            csv_path = output_path / "schedule.csv"
            json_path = output_path / "schedule.json"
            
            schedule_df.to_csv(csv_path, index=False)
            schedule_df.to_json(json_path, orient='records', indent=2)
            
            paths["schedule_csv"] = str(csv_path)
            paths["schedule_json"] = str(json_path)
        
        # Save raw JSON results
        raw_json_path = output_path / "results.json"
        with open(raw_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        paths["results_json"] = str(raw_json_path)
        
        return paths


def load_test_data() -> Dict[str, Any]:
    """
    Load sample test data for development and testing.
    
    Returns:
        Sample configuration dictionary
    """
    return {
        "tanks": {
            "T1": {"capacity": 500.0},
            "T2": {"capacity": 500.0},
            "T3": {"capacity": 600.0},
            "T4": {"capacity": 400.0}
        },
        "initial_tank_allocation": {
            "T1": {"crude": "A", "level": 300.0},
            "T2": {"crude": "B", "level": 250.0},
            "T3": {"crude": "C", "level": 400.0},
            "T4": {"crude": None, "level": 0.0}
        },
        "deliveries": [
            {"day": 5, "crude": "A", "volume": 200.0},
            {"day": 8, "crude": "B", "volume": 300.0},
            {"day": 12, "crude": "C", "volume": 350.0},
            {"day": 15, "crude": "D", "volume": 400.0}
        ],
        "pairing_and_blending": [
            {
                "name": "Blend1",
                "ratio": {"A": 0.3, "B": 0.7},
                "capacity_bpd": 75000.0
            },
            {
                "name": "Blend2",
                "ratio": {"A": 0.4, "B": 0.2, "C": 0.4},
                "capacity_bpd": 60000.0
            }
        ],
        "planning_horizon_days": 30,
        "look_ahead_days": 7,
        "smoothing_factor": 0.85,
        "critical_coverage_days": 8,
        "final_days_threshold": 10,
        "absolute_min_daily_processing_kb": 15.0,
        "min_daily_processing_kb": 50.0,
        "processing_increment_kb": 5.0,
        "daily_capacity_kb": 100.0
    }


if __name__ == "__main__":
    # Example usage
    try:
        # Try loading from a file (this will fail if file doesn't exist)
        config_file = "configs/scheduler_config.json"
        loader = DataLoader(config_file)
        
        # Get components
        tanks = loader.get_tanks()
        inventory = loader.get_inventory(tanks)
        deliveries, deliveries_by_day = loader.get_deliveries()
        blends = loader.get_blends()
        params = loader.get_scheduler_params()
        
        print(f"Loaded configuration with {len(tanks)} tanks, {len(deliveries)} deliveries, {len(blends)} blends")
        print(f"Planning horizon: {params['planning_horizon']} days")
        
    except ConfigValidationError as e:
        print(f"Error loading config: {e}")
        
        # Use test data instead
        print("Loading test data instead...")
        test_config = load_test_data()
        
        # Create a loader with the test data
        loader = DataLoader()
        loader.config = test_config
        
        # Get components from test data
        tanks = loader.get_tanks()
        inventory = loader.get_inventory(tanks)
        deliveries, deliveries_by_day = loader.get_deliveries()
        blends = loader.get_blends()
        params = loader.get_scheduler_params()
        
        print(f"Loaded test config with {len(tanks)} tanks, {len(deliveries)} deliveries, {len(blends)} blends")
        print(f"Planning horizon: {params['planning_horizon']} days")