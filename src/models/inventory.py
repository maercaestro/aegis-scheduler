from typing import Dict, Optional, List
from .tank import Tank


class CrudeInventory:
    """
    Represents the total inventory of each crude type, independent of tank storage.
    """
    def __init__(self, initial: Optional[Dict[str, float]] = None):
        # stock maps crude name to total available volume
        self.stock: Dict[str, float] = initial.copy() if initial else {}

    def add(self, crude: str, volume: float) -> None:
        """
        Increase available inventory of a crude by volume.
        """
        if volume <= 0:
            return
        self.stock[crude] = self.stock.get(crude, 0.0) + volume

    def remove(self, crude: str, volume: float) -> None:
        """
        Decrease available inventory of a crude by volume.
        Raises ValueError if insufficient inventory.
        """
        available = self.stock.get(crude, 0.0)
        if volume <= 0:
            return
        
        # Handle floating point precision issues - if we're very close to the available amount
        if abs(available - volume) < 1e-9:
            # Just use all the available amount
            volume = available
        
        if available < volume:
            raise ValueError(f"Not enough inventory for {crude}: requested {volume}, available {available}")
        
        remaining = available - volume
        if remaining <= 1e-6:
            # Remove entry if emptied
            del self.stock[crude]
        else:
            self.stock[crude] = remaining

    def available(self, crude: str, volume: float) -> bool:
        """
        Check if at least `volume` of `crude` is available.
        """
        available = self.stock.get(crude, 0.0)
        # Handle floating point precision issues
        return available >= volume or abs(available - volume) < 1e-9

    def get(self, crude: str) -> float:
        """
        Get current available volume of `crude`.
        Returns 0 if not present.
        """
        return self.stock.get(crude, 0.0)
    
    def get_all_crudes(self) -> List[str]:
        """
        Get a list of all crude types currently in inventory.
        """
        return list(self.stock.keys())

    def sync_with_tanks(self, tanks: Dict[str, Tank]) -> None:
        """
        Synchronize the inventory to match the sum of tank levels per crude.
        Tanks may contain mixed composition; this uses each tank's composition data.
        """
        tank_totals: Dict[str, float] = {}
        for tank in tanks.values():
            for c, vol in tank.composition.items():
                tank_totals[c] = tank_totals.get(c, 0.0) + vol

        # Replace stock with tank_totals, removing zero-volume entries
        self.stock = {c: v for c, v in tank_totals.items() if v > 1e-6}
