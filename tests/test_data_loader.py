import pytest
import os
import json
import tempfile
from src.data_loader import DataLoader, ConfigValidationError


class TestDataLoader:
    def test_load_config(self, test_config_path):
        """Test loading configuration from file"""
        loader = DataLoader(test_config_path)
        assert loader.config is not None
        assert "tanks" in loader.config
        assert "planning_horizon_days" in loader.config

    def test_get_tanks(self, data_loader):
        """Test getting tanks from configuration"""
        tanks = data_loader.get_tanks()
        assert len(tanks) == 7  # Updated to match the number of tanks in our config
        assert tanks[0].id in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    
    def test_invalid_config(self):
        """Test handling of invalid configuration"""
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            # Write invalid JSON
            with open(tmp.name, 'w') as f:
                f.write('{"tanks": "invalid"}')
            
            # Try to load invalid config
            loader = DataLoader()
            with pytest.raises(ConfigValidationError):
                loader.load_config(tmp.name)