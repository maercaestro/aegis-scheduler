import pytest
from models.tank import Tank
from models.inventory import CrudeInventory
from models.blend import BlendRecipe
from models.delivery import Delivery


class TestTank:
    def test_initialization(self):
        """Test tank initialization"""
        tank = Tank("T1", 500.0, "A", 300.0)
        assert tank.id == "T1"
        assert tank.capacity == 500.0
        assert tank.crude == "A"
        assert tank.level == 300.0
        assert tank.composition == {"A": 300.0}

    def test_available_space(self):
        """Test available space calculation"""
        tank = Tank("T1", 500.0, "A", 300.0)
        assert tank.available_space() == 200.0

    def test_is_empty(self):
        """Test is_empty method"""
        empty_tank = Tank("T2", 500.0)
        full_tank = Tank("T1", 500.0, "A", 300.0)
        assert empty_tank.is_empty() == True
        assert full_tank.is_empty() == False

    def test_deposit(self):
        """Test deposit method"""
        tank = Tank("T1", 500.0, "A", 300.0)
        remainder = tank.deposit("A", 150.0)
        assert remainder == 0.0
        assert tank.level == 450.0
        assert tank.composition["A"] == 450.0

    def test_withdraw(self):
        """Test withdraw method"""
        tank = Tank("T1", 500.0, "A", 300.0)
        tank.withdraw("A", 100.0)
        assert tank.level == 200.0
        assert tank.composition["A"] == 200.0