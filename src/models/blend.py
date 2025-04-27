from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BlendRecipe:
    """
    Represents a blend recipe: proportions for each crude and a maximum processing capacity (kb/day).

    The `ratios` dict defines relative blend proportions; these are normalized to sum to 1.
    """
    ratios: Dict[str, float] = field(default_factory=dict)
    max_capacity: float = 0.0

    def __post_init__(self):
        # Validate and normalize blend ratios to sum to 1.0
        total = sum(self.ratios.values())
        if total <= 0:
            raise ValueError("Blend ratios must sum to a positive value")
        self.ratios = {c: r / total for c, r in self.ratios.items()}

    def components(self) -> List[str]:
        """Return the list of crude names in this blend."""
        return list(self.ratios.keys())

    def capacity(self) -> float:
        """Return the maximum processing capacity (kb/day) for this blend."""
        return self.max_capacity
