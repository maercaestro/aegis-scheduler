from typing import List, Dict, Tuple
from .models.tank import Tank
from .models.delivery import Delivery


def allocate_delivery(
    tanks: List[Tank],
    delivery: Delivery,
    low_threshold: float = 10.0,
    allow_mixing: bool = True,
    max_mix_ratio: float = 0.3,
) -> Tuple[Dict[str, float], float]:
    """
    Allocate `delivery.volume` of `delivery.crude` into `tanks`.

    Steps:
      1. Fill tanks that already contain this crude or are below low_threshold (treat as empty).
      2. Fill completely empty tanks.
      3. Optionally mix into other-crude tanks up to max_mix_ratio.

    Returns:
      - allocation: dict mapping tank_id -> volume deposited
      - remainder: volume that couldn't be deposited
    """
    remaining = delivery.volume
    allocation: Dict[str, float] = {}

    # 1) Fill same-crude or low-inventory tanks
    for tank in tanks:
        if tank.crude == delivery.crude or tank.level < low_threshold:
            # Treat low-level tanks as empty
            if tank.level < low_threshold:
                tank.clear_if_low(low_threshold)
            # Determine how much to deposit
            deposit_amt = min(remaining, tank.available_space())
            if deposit_amt <= 1e-6:
                continue
            # Deposit and compute actual deposited
            leftover = tank.deposit(delivery.crude, deposit_amt)
            actual = deposit_amt - leftover
            allocation[tank.id] = allocation.get(tank.id, 0.0) + actual
            remaining -= actual
            if remaining <= 1e-6:
                return allocation, 0.0

    # 2) Fill empty tanks
    for tank in tanks:
        if tank.is_empty() and remaining > 1e-6:
            deposit_amt = min(remaining, tank.available_space())
            if deposit_amt <= 1e-6:
                continue
            leftover = tank.deposit(delivery.crude, deposit_amt)
            actual = deposit_amt - leftover
            allocation[tank.id] = allocation.get(tank.id, 0.0) + actual
            remaining -= actual
            if remaining <= 1e-6:
                return allocation, 0.0

    # 3) Mixing into other-crude tanks
    if allow_mixing and remaining > 1e-6:
        for tank in tanks:
            if (
                tank.crude is not None
                and tank.crude != delivery.crude
                and tank.level > 1e-6
                and tank.level < tank.capacity - 1e-6
            ):
                # Calculate max secondary volume allowed by mix ratio
                t0 = tank.level
                max_secondary = (max_mix_ratio * t0) / (1 - max_mix_ratio)
                mix_space = min(max_secondary, tank.capacity - t0)
                deposit_amt = min(remaining, mix_space)
                if deposit_amt <= 1e-6:
                    continue
                leftover = tank.deposit(delivery.crude, deposit_amt)
                actual = deposit_amt - leftover
                allocation[tank.id] = allocation.get(tank.id, 0.0) + actual
                remaining -= actual
                if remaining <= 1e-6:
                    break

    return allocation, remaining
