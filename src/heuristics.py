from typing import Dict, List, Tuple, Any


def choose_strategy(
    day: int,
    inventory: Any,
    upcoming_deliveries: List[Any],
    blend_recipes: List[Any],
    params: Dict[str, Any]
) -> Tuple[str, float, int]:
    """
    Choose processing strategy for the current day based on inventory
    and upcoming deliveries.
    
    Args:
        day: Current day in the planning horizon
        inventory: Current inventory state
        upcoming_deliveries: List of upcoming deliveries (Delivery objects)
        blend_recipes: Available blend recipes
        params: Configuration parameters
        
    Returns:
        Tuple of (strategy, processing_rate, blend_index)
    """
    # Default rate from configuration or use 50.0 if not specified
    default_rate = params.get("default_processing_rate", 50.0)
    min_rate = params.get("min_daily", 15.0)
    lookahead = params.get("lookahead", 7)
    critical_days = params.get("critical", 7)
    
    # First find which blend recipes are feasible with current inventory
    feasible_blends = []
    for idx, blend in enumerate(blend_recipes):
        # For each crude in this blend, check if we have enough inventory
        can_process = True
        max_possible = float('inf')
        
        for crude, ratio in blend.ratios.items():
            if ratio <= 0:
                continue
                
            # Check how much we could process based on this crude
            available = inventory.get(crude)
            if available <= 0:
                can_process = False
                break
                
            # Calculate how much we could process with this crude
            possible_with_crude = available / ratio
            max_possible = min(max_possible, possible_with_crude)
        
        # If we can process and have a reasonable amount, add to feasible blends
        if can_process and max_possible >= min_rate:
            # Get the effective max processing rate (min of blend capacity and inventory)
            effective_max = min(max_possible, blend.max_capacity)
            feasible_blends.append((idx, blend, effective_max))
    
    # If no feasible blends, return "none" with zero processing
    if not feasible_blends:
        return "none", 0.0, None
    
    # Calculate projected inventory including upcoming deliveries
    # This helps us prioritize blends that use crude types that will be replenished
    projected_inventory = {}
    for future_day in range(day, day + lookahead + 1):
        projected_inventory[future_day] = {}
        
        # Start with current day's inventory
        if future_day == day:
            for crude in inventory.get_all_crudes():
                projected_inventory[day][crude] = inventory.get(crude)
        else:
            # Copy from previous day
            for crude, amount in projected_inventory[future_day - 1].items():
                projected_inventory[future_day][crude] = amount
        
        # Add upcoming deliveries for this day - using Delivery object attributes directly
        for delivery in upcoming_deliveries:
            # Access attributes directly instead of using .get()
            delivery_day = delivery.day
            if delivery_day == future_day:
                crude = delivery.crude
                volume = delivery.volume
                if crude and volume > 0:
                    projected_inventory[future_day][crude] = projected_inventory[future_day].get(crude, 0.0) + volume
    
    # Score each blend based on inventory availability and upcoming deliveries
    best_score = -float('inf')
    best_blend_idx = -1
    best_rate = 0.0
    
    for idx, blend, max_rate in feasible_blends:
        # Start with a base score - the effective processing rate
        score = max_rate
        
        # Adjust score based on rationing factor (how long inventory will last)
        rationing = calculate_rationing_factor(
            blend.ratios, day, projected_inventory, critical_days)
            
        # If we have limited inventory for this blend, reduce the score
        score *= rationing
        
        # Adjust for final days of the planning horizon
        days_remaining = params.get("planning_horizon", 31) - day + 1
        if days_remaining <= params.get("final_days", 10):
            # In final days, prioritize blends with highest processing rates
            # to maximize throughput
            score *= 1.2
        
        # Keep track of the highest scoring blend
        if score > best_score:
            best_score = score
            best_blend_idx = idx
            
            # Determine processing rate based on rationing factor
            best_rate = max_rate * rationing
            
            # Enforce minimum processing rate from parameters
            if best_rate < min_rate:
                best_rate = 0.0  # If we can't meet minimum rate, don't process
    
    # If we found a valid blend with non-zero rate
    if best_blend_idx >= 0 and best_rate > 0:
        # Final adjustment for the end of the planning horizon
        days_remaining = params.get("planning_horizon", 31) - day + 1
        final_days_threshold = params.get("final_days", 10)
        absolute_min = params.get("absolute_min", 15.0)
        
        best_rate = enforce_final_days(
            best_rate, days_remaining, final_days_threshold, absolute_min)
            
        strategy = "standard"
        
        # Add some strategy variations based on conditions
        if rationing < 0.9:
            strategy = "rationed"
        elif days_remaining <= params.get("final_days", 10):
            strategy = "final_days"
            
        return strategy, best_rate, best_blend_idx
    
    # Fallback if no valid blend was found
    return "none", 0.0, None


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
