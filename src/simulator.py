from typing import List, Dict, Tuple
from .models.blend import BlendRecipe
from .models.inventory import CrudeInventory
from .models.tank import Tank
from .models.delivery import Delivery
from .allocation import allocate_delivery
import math


def check_tank_inventory(
    tanks: List[Tank],
    blend_ratios: Dict[str, float],
    amount: float
) -> bool:
    """
    Return True if tanks collectively have at least `amount * ratio` of each crude.
    """
    if amount <= 1e-6:
        return True
    for crude, ratio in blend_ratios.items():
        if ratio <= 0:
            continue
        required = amount * ratio
        available = sum(
            tank.composition.get(crude, 0.0) for tank in tanks
        )
        if available < required - 1e-6:
            return False
    return True


def find_max_feasible_amount(
    tanks: List[Tank],
    blend_ratios: Dict[str, float],
    upper_bound: float,
    increment_step: float = 5.0
) -> float:
    """
    Binary search to find the largest feasible amount <= upper_bound that tanks can supply.
    """
    if upper_bound <= 1e-6 or not check_tank_inventory(tanks, blend_ratios, 1e-6):
        return 0.0
    # If full amount is feasible, round down to increment step
    if check_tank_inventory(tanks, blend_ratios, upper_bound):
        return math.floor(upper_bound / increment_step) * increment_step
    low, high = 0.0, upper_bound
    best = 0.0
    for _ in range(20):
        mid = (low + high) / 2
        if check_tank_inventory(tanks, blend_ratios, mid):
            best = mid
            low = mid
        else:
            high = mid
        if high - low < increment_step:
            break
    return math.floor(best / increment_step) * increment_step


def simulate_blend(
    blend: BlendRecipe,
    start_inventory: CrudeInventory,
    start_tanks: List[Tank],
    future_deliveries: Dict[int, List[Delivery]],
    window: int,
    increment: float = 5.0,
) -> List[float]:
    """
    Simulate look-ahead processing for a single blend over a window of days.

    Returns a list of max feasible processing volumes for each day in the window.
    """
    # Copy initial state
    sim_inventory = CrudeInventory(start_inventory.stock.copy())
    sim_tanks = [Tank(t.id, t.capacity, t.crude, t.level) for t in start_tanks]
    max_per_day: List[float] = []

    for day in range(1, window + 1):
        # Apply deliveries
        if day in future_deliveries:
            for d in future_deliveries[day]:
                sim_inventory.add(d.crude, d.volume)
                allocate_delivery(sim_tanks, d)

        # Compute inventory-based upper limit
        day_max = blend.max_capacity
        for crude, ratio in blend.ratios.items():
            if ratio <= 0:
                continue
            inv_available = sim_inventory.get(crude)
            possible = inv_available / ratio
            day_max = min(day_max, possible)

        # Enforce tank constraints
        if not check_tank_inventory(sim_tanks, blend.ratios, day_max):
            day_max = find_max_feasible_amount(
                sim_tanks, blend.ratios, day_max, increment
            )
        max_per_day.append(day_max)

        # Withdraw from inventory and tanks
        if day_max > 1e-6:
            # Update inventory
            for crude, ratio in blend.ratios.items():
                used = day_max * ratio
                sim_inventory.remove(crude, used)
            # Withdraw from tanks proportionally
            for crude, ratio in blend.ratios.items():
                required = day_max * ratio
                remaining_crude = required
                for tank in sim_tanks:
                    avail = tank.composition.get(crude, 0.0)
                    to_withdraw = min(avail, remaining_crude)
                    if to_withdraw > 1e-6:
                        tank.withdraw(crude, to_withdraw)
                        remaining_crude -= to_withdraw
                        if remaining_crude <= 1e-6:
                            break
                # Optionally warn if remaining_crude > tolerance
                if remaining_crude > 1e-6:
                    print(f"Warning: Could not withdraw {remaining_crude:.2f} kb of {crude} on day {day}")

    return max_per_day
