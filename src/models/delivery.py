from dataclasses import dataclass, field
from typing import Dict, Optional


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
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Delivery":
        """
        Create a Delivery instance from a dictionary.
        """
        return cls(
            day=int(data.get("date", data.get("day"))),
            crude=data["crude"],
            volume=float(data["volume"]),
            original_day=data.get("original_date", data.get("original_day")),
            delay_days=data.get("delay_days"),
            tank_allocation=data.get("tank_allocation", {}),
        )
