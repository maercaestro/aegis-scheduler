from typing import Dict, Optional, Tuple


class Tank:
    """
    Represents a storage tank with a fixed capacity, current level, and crude composition.
    """
    def __init__(
        self,
        tank_id: str,
        capacity: float,
        crude: Optional[str] = None,
        level: float = 0.0,
    ):
        self.id: str = tank_id
        self.capacity: float = capacity
        self.level: float = level
        self.crude: Optional[str] = crude
        # composition holds actual volumes per crude in the tank
        self.composition: Dict[str, float] = {}
        if crude and level > 0:
            self.composition[crude] = level

    def available_space(self) -> float:
        """Return available capacity in the tank."""
        return max(self.capacity - self.level, 0.0)

    def is_empty(self) -> bool:
        """Check if the tank is effectively empty."""
        return self.level <= 1e-6

    def clear_if_low(self, threshold: float) -> None:
        """
        If the tank level is below threshold, treat it as empty by clearing contents.
        """
        if self.level < threshold:
            self.crude = None
            self.level = 0.0
            self.composition.clear()

    def deposit(self, crude: str, volume: float) -> float:
        """
        Deposit up to `volume` of crude into the tank until it is full.
        Updates composition and primary crude. Returns the un-deposited remainder.
        """
        if volume <= 0:
            return volume  # nothing to deposit, full remainder

        space = self.available_space()
        if space <= 1e-6:
            return volume  # tank is full, no deposit done

        # amount we can actually deposit
        deposit_amount = min(volume, space)

        # update composition
        if crude in self.composition:
            self.composition[crude] += deposit_amount
        else:
            self.composition[crude] = deposit_amount

        # update tank level and primary crude label
        self.level += deposit_amount
        self.crude = max(self.composition, key=self.composition.get)

        # return leftover
        return volume - deposit_amount

    def withdraw(self, crude: str, volume: float) -> None:
        """
        Withdraw a given volume of crude from the tank. Updates composition and level.
        Raises ValueError if insufficient volume.
        """
        available = self.composition.get(crude, 0.0)
        if volume <= 0:
            return  # no-op for zero or negative
        if available < volume:
            raise ValueError(
                f"Not enough {crude} to withdraw: requested {volume}, available {available}"
            )
        # subtract from composition and level
        self.composition[crude] = available - volume
        self.level -= volume
        # remove entry if depleted
        if self.composition[crude] <= 1e-6:
            del self.composition[crude]
        # update crude label or mark empty
        if self.composition:
            self.crude = max(self.composition, key=self.composition.get)
        else:
            self.crude = None
            self.level = 0.0
