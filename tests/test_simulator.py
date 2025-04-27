import pytest
from models.tank import Tank
from models.blend import BlendRecipe
from models.inventory import CrudeInventory
from models.delivery import Delivery
from simulator import check_tank_inventory, find_max_feasible_amount, simulate_blend


class TestSimulator:
    def test_check_tank_inventory_sufficient(self):
        """Test that check_tank_inventory returns True when tanks have sufficient inventory"""
        tanks = [
            Tank("T1", 500.0, "A", 150.0),
            Tank("T2", 500.0, "B", 350.0),
        ]
        
        # Blend requires 30% A, 70% B
        blend_ratios = {"A": 0.3, "B": 0.7}
        
        # Check for 100 units total (needs 30 A, 70 B)
        result = check_tank_inventory(tanks, blend_ratios, 100.0)
        assert result is True
    
    def test_check_tank_inventory_insufficient(self):
        """Test that check_tank_inventory returns False when tanks don't have enough inventory"""
        tanks = [
            Tank("T1", 500.0, "A", 20.0),  # Not enough A
            Tank("T2", 500.0, "B", 350.0),
        ]
        
        # Blend requires 30% A, 70% B
        blend_ratios = {"A": 0.3, "B": 0.7}
        
        # Check for 100 units total (needs 30 A, 70 B)
        result = check_tank_inventory(tanks, blend_ratios, 100.0)
        assert result is False
    
    def test_check_tank_inventory_zero_amount(self):
        """Test that check_tank_inventory returns True for zero amount"""
        tanks = [
            Tank("T1", 500.0, "A", 0.0),
            Tank("T2", 500.0, "B", 0.0),
        ]
        
        blend_ratios = {"A": 0.3, "B": 0.7}
        result = check_tank_inventory(tanks, blend_ratios, 0.0)
        assert result is True
    
    def test_find_max_feasible_amount_full(self):
        """Test finding max feasible amount when full amount is possible"""
        tanks = [
            Tank("T1", 500.0, "A", 150.0),
            Tank("T2", 500.0, "B", 350.0),
        ]
        
        # Blend requires 30% A, 70% B
        blend_ratios = {"A": 0.3, "B": 0.7}
        
        # With 150 A, we can make at most 150/0.3 = 500 total
        # With 350 B, we can make at most 350/0.7 = 500 total
        # So 200 should be entirely feasible
        result = find_max_feasible_amount(tanks, blend_ratios, 200.0, increment_step=5.0)
        assert result == 200.0
    
    def test_find_max_feasible_amount_partial(self):
        """Test finding max feasible amount when only partial amount is possible"""
        tanks = [
            Tank("T1", 500.0, "A", 30.0),  # Only enough A for 100 total
            Tank("T2", 500.0, "B", 350.0),
        ]
        
        # Blend requires 30% A, 70% B
        blend_ratios = {"A": 0.3, "B": 0.7}
        
        # With 30 A, we can make at most 30/0.3 = 100 total
        result = find_max_feasible_amount(tanks, blend_ratios, 200.0, increment_step=5.0)
        assert result == 100.0
    
    def test_find_max_feasible_amount_zero(self):
        """Test finding max feasible amount when nothing is possible"""
        tanks = [
            Tank("T1", 500.0, "A", 0.0),
            Tank("T2", 500.0, "B", 350.0),
        ]
        
        # Blend requires 30% A, 70% B, but we have no A
        blend_ratios = {"A": 0.3, "B": 0.7}
        
        result = find_max_feasible_amount(tanks, blend_ratios, 200.0, increment_step=5.0)
        assert result == 0.0
    
    def test_simulate_blend_basic(self):
        """Test basic simulate_blend functionality"""
        # Setup tanks and inventory
        tanks = [
            Tank("T1", 500.0, "A", 150.0),
            Tank("T2", 500.0, "B", 350.0),
        ]
        
        inventory = CrudeInventory({"A": 150.0, "B": 350.0})
        
        # Blend requires 30% A, 70% B
        blend = BlendRecipe({"A": 0.3, "B": 0.7}, 100.0)
        
        # No future deliveries
        future_deliveries = {}
        
        # Simulate for 3 days
        window = 3
        
        results = simulate_blend(blend, inventory, tanks, future_deliveries, window)
        
        # With 150 A, 350 B, we can make:
        # A allows 150/0.3 = 500 total
        # B allows 350/0.7 = 500 total
        # Limited by max_capacity = 100 per day
        
        # Should get 100 for each of the 3 days
        assert len(results) == 3
        assert results[0] == 100.0
        assert results[1] == 100.0
        assert results[2] == 100.0
    
    def test_simulate_blend_with_deliveries(self):
        """Test simulate_blend with future deliveries"""
        # Setup tanks and inventory with limited initial amounts
        tanks = [
            Tank("T1", 500.0, "A", 30.0),  # Only enough A for 100 total initially
            Tank("T2", 500.0, "B", 70.0),
        ]
        
        inventory = CrudeInventory({"A": 30.0, "B": 70.0})
        
        # Blend requires 30% A, 70% B
        blend = BlendRecipe({"A": 0.3, "B": 0.7}, 100.0)
        
        # Delivery of more A and B on day 2
        future_deliveries = {
            2: [Delivery(2, "A", 60.0), Delivery(2, "B", 140.0)]
        }
        
        # Simulate for 3 days
        window = 3
        
        results = simulate_blend(blend, inventory, tanks, future_deliveries, window)
        
        # Day 1: With 30 A, 70 B, we can make 30/0.3 = 100 total
        # Day 2: After delivery, we have 60 more A, 140 more B
        # Day 3: Should still have materials for 100 more
        
        assert len(results) == 3
        assert results[0] == 100.0  # Day 1: We use all initial inventory
        assert results[1] == 100.0  # Day 2: We get a delivery and can process 100
        assert results[2] == 100.0  # Day 3: We still have enough for 100 more