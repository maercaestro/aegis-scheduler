import os
import sys
import pytest
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use absolute imports instead of relative imports
from src.data_loader import DataLoader
from src.models.tank import Tank
from src.models.inventory import CrudeInventory
from src.models.blend import BlendRecipe
from src.models.delivery import Delivery


@pytest.fixture
def test_config_path():
    """Path to test config file"""
    return os.path.join(os.path.dirname(__file__), "..", "configs", "test_data.json")


@pytest.fixture
def sample_test_data():
    """Sample test data dictionary"""
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

@pytest.fixture
def data_loader(test_config_path, sample_test_data):
    """DataLoader instance with test configuration"""
    # Ensure the test config exists
    config_path = Path(test_config_path)
    if not config_path.exists():
        # Create parent directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Write test config if not exists
        with open(config_path, 'w') as f:
            json.dump(sample_test_data, f, indent=2)
    
    # Create and return data loader
    return DataLoader(str(config_path))

# Additional fixtures for tanks, inventory, deliveries, etc.