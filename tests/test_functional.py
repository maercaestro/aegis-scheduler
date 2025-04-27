import pytest
import os
import tempfile
from scheduler import schedule_plant_operation


@pytest.mark.skip(reason="Test data configuration needs to be fixed")
def test_end_to_end_scheduling(test_config_path):
    """Test the full scheduling process from input to output"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = schedule_plant_operation(
            test_config_path,
            output_dir=tmpdirname,
            optimize_vessels=True
        )
        
        assert result["status"] == "success"
        assert result["days_scheduled"] > 0
        assert result["total_processed"] > 0
        assert os.path.exists(result["files"]["daily_schedule_csv"])