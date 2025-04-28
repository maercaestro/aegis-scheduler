from dataclasses import dataclass, field
from typing import Dict, Optional
from .tank import Tank


@dataclass
class Delivery:
    """
    Represents a scheduled delivery of a crude type on a particular day.
    """
    day: int
    crude: str
    volume: float
    original_day: Optional[int] = None
    delay_days: Optional[int] = None
    tank_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Track loading status
    is_loading: bool = False
    loading_days_left: int = 0
    unloaded_volume: float = 0.0
    
    def start_loading(self, loading_days: int = 2) -> None:
        """
        Start the loading process for this delivery.
        
        Args:
            loading_days: Number of days the loading process takes
        """
        self.is_loading = True
        self.loading_days_left = loading_days
        self.unloaded_volume = self.volume
        
    def continue_loading(self, tanks_dict: Dict[str, 'Tank']) -> float:
        """
        Continue the loading process for this delivery.
        Attempts to transfer crude from the ship to available tanks.
        
        Args:
            tanks_dict: Dictionary of tank objects to load into
            
        Returns:
            Amount successfully transferred to tanks
        """
        if not self.is_loading or self.unloaded_volume <= 0:
            return 0.0
        
        # Reduce loading days
        self.loading_days_left = max(0, self.loading_days_left - 1)
        
        # Try to transfer crude to tanks
        transferred = 0.0
        original_unloaded = self.unloaded_volume
        
        # Find tanks that can hold this crude type
        available_tanks = []
        for tank_id, tank in tanks_dict.items():
            if tank.crude is None or tank.crude == self.crude:
                available_tanks.append((tank_id, tank))
        
        # Sort tanks by available space (largest first)
        available_tanks.sort(key=lambda t: t[1].available_space(), reverse=True)
        
        # Try to deposit in each tank
        for tank_id, tank in available_tanks:
            if self.unloaded_volume <= 0:
                break
                
            # Attempt to deposit crude
            before = self.unloaded_volume
            self.unloaded_volume = tank.deposit(self.crude, self.unloaded_volume)
            amount = before - self.unloaded_volume
            
            if amount > 0:
                # Record tank allocation
                self.tank_allocation[tank_id] = self.tank_allocation.get(tank_id, 0.0) + amount
                transferred += amount
        
        # Check if loading is complete (either no more days or no more volume)
        if self.loading_days_left == 0 or self.unloaded_volume <= 0:
            self.is_loading = False
        
        return transferred
    
    def delay(self, days: int = 1) -> None:
        """
        Delay the delivery by the specified number of days.
        
        Args:
            days: Number of days to delay the delivery
        """
        if self.original_day is None:
            self.original_day = self.day
        
        self.day += days
        self.delay_days = (self.delay_days or 0) + days
        
    def is_complete(self) -> bool:
        """
        Check if the delivery is completely unloaded.
        """
        return not self.is_loading and self.unloaded_volume <= 0
    
    def get_delay_status(self) -> str:
        """
        Get a string representation of the delivery delay status.
        
        Returns:
            String with delay information or "On time" if not delayed
        """
        if self.delay_days and self.delay_days > 0:
            return f"Delayed by {self.delay_days} days (original: day {self.original_day})"
        return "On time"

    def to_dict(self) -> Dict:
        """
        Convert the Delivery instance to a serializable dictionary.
        """
        return {
            "day": self.day,
            "crude": self.crude,
            "volume": self.volume,
            "original_day": self.original_day,
            "delay_days": self.delay_days,
            "tank_allocation": self.tank_allocation,
            "is_loading": self.is_loading,
            "unloaded_volume": self.unloaded_volume,
            "loading_days_left": self.loading_days_left
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Delivery":
        """
        Create a Delivery instance from a dictionary.
        """
        delivery = cls(
            day=int(data.get("date", data.get("day"))),
            crude=data["crude"],
            volume=float(data["volume"]),
            original_day=data.get("original_date", data.get("original_day")),
            delay_days=data.get("delay_days"),
            tank_allocation=data.get("tank_allocation", {}),
        )
        
        # Set loading status if available
        if "is_loading" in data:
            delivery.is_loading = data["is_loading"]
        if "unloaded_volume" in data:
            delivery.unloaded_volume = data["unloaded_volume"]
        if "loading_days_left" in data:
            delivery.loading_days_left = data["loading_days_left"]
        
        return delivery
