import pytest
from models.tank import Tank
from models.delivery import Delivery
from allocation import allocate_delivery


def test_allocate_to_same_crude():
    """Test allocating to tanks with same crude"""
    tanks = [
        Tank("T1", 500.0, "A", 300.0),
        Tank("T2", 500.0, "B", 250.0)
    ]
    delivery = Delivery(5, "A", 150.0)
    allocation, remainder = allocate_delivery(tanks, delivery)
    
    assert remainder == 0.0
    assert allocation == {"T1": 150.0}
    assert tanks[0].level == 450.0
    assert tanks[1].level == 250.0

def test_allocate_to_empty_tank():
    """Test allocating to empty tank"""
    tanks = [
        Tank("T1", 500.0, "A", 300.0),
        Tank("T2", 500.0, None, 0.0)
    ]
    delivery = Delivery(5, "B", 350.0)
    allocation, remainder = allocate_delivery(tanks, delivery)
    
    assert remainder == 0.0
    assert allocation == {"T2": 350.0}
    assert tanks[1].level == 350.0
    assert tanks[1].crude == "B"

def test_allocation_with_remainder():
    """Test when tanks can't fit full delivery"""
    tanks = [
        Tank("T1", 500.0, "A", 450.0),
        Tank("T2", 500.0, "B", 450.0)
    ]
    delivery = Delivery(5, "A", 100.0)
    allocation, remainder = allocate_delivery(tanks, delivery)
    
    # The function should allocate 50 to T1 (to capacity)
    assert tanks[0].level == 500.0
    assert "T1" in allocation
    assert allocation["T1"] == 50.0