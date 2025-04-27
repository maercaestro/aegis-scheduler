from typing import Dict, List, Tuple


def choose_strategy(
    max_per_day: List[float],
    smoothing_factor: float,
    low_inventory_threshold: float = 10.0
) -> Tuple[float, float, str]:
    """
    Compare a greedy vs. smooth processing strategy.

    - greedy_total = sum(max_per_day)
    - smooth_rate = min(max_per_day)
      smooth_total = smooth_rate * len(max_per_day)

    If smooth_total >= greedy_total * smoothing_factor, choose smooth.
    Otherwise choose greedy (processing max_per_day[0] today).
    
    If the first day's max is very low (below low_inventory_threshold), 
    it indicates we're running low on inventory for this blend.

    Returns:
      (daily_rate, total_processing, strategy_name)
    """
    if not max_per_day:
        return 0.0, 0.0, "none"
    
    # Check if the first day's max is below threshold - indicates low inventory
    if max_per_day[0] < low_inventory_threshold:
        # Return a very low score to discourage using this blend
        return max_per_day[0], max_per_day[0], "low_inventory"
    
    greedy_total = sum(max_per_day)
    smooth_rate = min(max_per_day)
    smooth_total = smooth_rate * len(max_per_day)
    
    if smooth_total >= greedy_total * smoothing_factor and smooth_rate > 1e-6:
        return smooth_rate, smooth_total, "smooth"
    # fallback to greedy
    return max_per_day[0], greedy_total, "greedy"


def calculate_rationing_factor(
    blend_ratios: Dict[str, float],
    day: int,
    inventory_projection: Dict[int, Dict[str, float]],
    critical_days: int
) -> float:
    """
    Calculate a rationing factor [0.5..1.0] based on projected coverage.

    For each crude in blend_ratios, count days until projected inventory <= 0.
    Let min_days_covered = minimum across crudes.

    If min_days_covered < critical_days:
        return max(0.5, min_days_covered / critical_days)
    else:
        return 1.0
    """
    min_days_covered = float('inf')
    horizon = max(inventory_projection.keys(), default=day)
    for crude, ratio in blend_ratios.items():
        if ratio <= 0:
            continue
        days_covered = 0
        for future_day in range(day, horizon + 1):
            inv = inventory_projection.get(future_day, {}).get(crude, 0.0)
            if inv > 0:
                days_covered += 1
            else:
                break
        min_days_covered = min(min_days_covered, days_covered)
    if min_days_covered < critical_days:
        return max(0.5, min_days_covered / critical_days)
    return 1.0


def enforce_final_days(
    daily_rate: float,
    days_remaining: int,
    final_days_threshold: int,
    absolute_min_rate: float
) -> float:
    """
    If days_remaining <= final_days_threshold and daily_rate < absolute_min_rate,
    force up to absolute_min_rate (if feasible).
    """
    if days_remaining <= final_days_threshold and daily_rate < absolute_min_rate:
        return absolute_min_rate
    return daily_rate
