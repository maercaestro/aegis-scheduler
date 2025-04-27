"""
Optimizer module for improving the scheduler's decisions.
"""
from typing import List, Dict, Any, Tuple, Optional
import logging
import copy

from .models.inventory import CrudeInventory
from .models.blend import BlendRecipe
from .models.tank import Tank

logger = logging.getLogger(__name__)

class EndPeriodOptimizer:
    """
    Optimizer that improves scheduling decisions to ensure
    consistent high throughput throughout the planning horizon.
    """
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the optimizer with parameters.
        
        Args:
            params: Dictionary of optimizer parameters
        """
        self.params = params or {}
        self.min_processing_rate = self.params.get("min_processing_rate", 70.0)
        self.target_utilization = self.params.get("target_utilization", 0.95) 
        self.end_period_window = self.params.get("end_period_window", 5)
        self.target_end_inventory_pct = self.params.get("target_end_inventory_pct", 0.1)
        self.smoothing_window = self.params.get("smoothing_window", 3)
        self.max_iterations = self.params.get("max_iterations", 3)
        
        # Define crude name mappings (e.g., E2 -> E, F2 -> F)
        self.crude_mappings = self.params.get("crude_mappings", {
            "E2": "E",
            "F2": "F"
        })
        
    def _normalize_crude_name(self, crude_name: str) -> str:
        """
        Normalize crude names based on defined mappings.
        For example, convert E2 to E or F2 to F.
        
        Args:
            crude_name: Original crude name
            
        Returns:
            Normalized crude name
        """
        return self.crude_mappings.get(crude_name, crude_name)
    
    def _normalize_blend(self, blend: BlendRecipe) -> BlendRecipe:
        """
        Normalize crude names in a blend recipe.
        
        Args:
            blend: Original blend recipe
            
        Returns:
            Normalized blend recipe
        """
        normalized_ratios = {}
        
        for crude, ratio in blend.ratios.items():
            normalized_crude = self._normalize_crude_name(crude)
            normalized_ratios[normalized_crude] = normalized_ratios.get(normalized_crude, 0.0) + ratio
            
        return BlendRecipe(ratios=normalized_ratios, max_capacity=blend.max_capacity)
    
    def _normalize_crude_usage(self, crude_usage: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize crude names in crude usage dictionary.
        
        Args:
            crude_usage: Original crude usage
            
        Returns:
            Normalized crude usage
        """
        normalized_usage = {}
        
        for crude, amount in crude_usage.items():
            normalized_crude = self._normalize_crude_name(crude)
            normalized_usage[normalized_crude] = normalized_usage.get(normalized_crude, 0.0) + amount
            
        return normalized_usage

    def _simulate_tank_withdrawals(self, 
                                  tanks_data: Dict[str, Dict[str, Any]], 
                                  crude_usage: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Simulate withdrawals from tanks based on crude usage.
        
        Args:
            tanks_data: Dictionary of tank data from schedule record
            crude_usage: Dictionary mapping crude types to usage amounts
            
        Returns:
            Updated tank data after withdrawals
        """
        updated_tanks = copy.deepcopy(tanks_data)
        
        # Process each crude type that needs to be withdrawn
        for crude_type, amount in crude_usage.items():
            remaining = amount
            
            # Find tanks containing this crude
            for tank_id, tank in updated_tanks.items():
                composition = tank["composition"]
                available = composition.get(crude_type, 0.0)
                
                if available > 0:
                    # Calculate how much to take from this tank
                    to_withdraw = min(available, remaining)
                    
                    # Update the tank
                    composition[crude_type] -= to_withdraw
                    if composition[crude_type] <= 0.001:  # Avoid floating point imprecision
                        composition.pop(crude_type)
                    
                    # Update tank level
                    tank["level"] -= to_withdraw
                    if tank["level"] <= 0.001:  # Avoid floating point imprecision
                        tank["level"] = 0.0
                        tank["crude"] = None
                        tank["composition"] = {}
                    
                    # Update remaining amount
                    remaining -= to_withdraw
                    
                    if remaining <= 0.001:  # Avoid floating point imprecision
                        break
            
            # Warning if we couldn't withdraw all needed
            if remaining > 0.001:
                logger.warning(f"Could not withdraw all required {crude_type} from tanks. Short by {remaining:.4f} kb")
                
        return updated_tanks
    
    def optimize_schedule(self, 
                          schedule: List[Dict[str, Any]], 
                          blends: List[BlendRecipe],
                          initial_inventory: CrudeInventory,
                          tanks: List[Tank],
                          planning_horizon: int) -> List[Dict[str, Any]]:
        """
        Optimize the schedule to ensure consistent high throughput throughout
        the planning horizon while respecting all constraints.
        
        Args:
            schedule: Original schedule records
            blends: Available blend recipes
            initial_inventory: Initial crude inventory
            tanks: Available tanks
            planning_horizon: Planning horizon in days
            
        Returns:
            Optimized schedule records
        """
        if not schedule:
            return schedule
        
        # Make a copy of the schedule to avoid modifying the original
        optimized_schedule = copy.deepcopy(schedule)
        
        # Normalize blends to handle E2/E and F2/F correctly
        normalized_blends = [self._normalize_blend(blend) for blend in blends]
        
        # Normalize crude names in inventory
        normalized_inventory = CrudeInventory({})
        for crude, amount in initial_inventory.stock.items():
            normalized_crude = self._normalize_crude_name(crude)
            normalized_inventory.stock[normalized_crude] = normalized_inventory.stock.get(normalized_crude, 0.0) + amount
        
        # Calculate total available inventory over the entire period
        total_initial_inventory = sum(normalized_inventory.stock.values())
        
        # Calculate total deliveries across the planning horizon
        total_deliveries = 0.0
        for record in schedule:
            for delivery in record.get("Deliveries", []):
                crude = self._normalize_crude_name(delivery["crude"])
                total_deliveries += delivery.get("allocated", 0.0)
        
        # Total available volume across the entire period
        total_available_volume = total_initial_inventory + total_deliveries
        
        # Calculate the optimal consistent daily rate across the entire planning horizon
        theoretical_daily_rate = total_available_volume / planning_horizon
        
        # We'll set our target rate as the higher of our minimum rate or the theoretical rate
        target_daily_rate = max(self.min_processing_rate, theoretical_daily_rate * self.target_utilization)
        
        logger.info(f"Optimization - Initial inventory: {total_initial_inventory:.2f} kb")
        logger.info(f"Optimization - Total deliveries: {total_deliveries:.2f} kb")
        logger.info(f"Optimization - Total available volume: {total_available_volume:.2f} kb")
        logger.info(f"Optimization - Theoretical daily rate: {theoretical_daily_rate:.2f} kb/day")
        logger.info(f"Optimization - Target daily rate: {target_daily_rate:.2f} kb/day")
        
        # First pass: Find blend recipes that use accumulated crude types (like E2/E)
        # This step requires understanding which blends can use which crude types
        optimized_schedule = self._select_optimal_blends(
            optimized_schedule, normalized_blends)
        
        # NEW: Iteratively optimize the entire schedule from beginning to end
        optimized_schedule = self._optimize_full_schedule(
            optimized_schedule, normalized_blends, target_daily_rate)
        
        # Second pass: Set low days to at least the minimum rate where possible
        optimized_schedule = self._optimize_low_processing_days(
            optimized_schedule, normalized_blends, target_daily_rate)
        
        # Third pass: Balance end period more effectively
        optimized_schedule = self._optimize_end_period(
            optimized_schedule, normalized_blends, planning_horizon)
        
        # NEW: Final smoothing pass to reduce large variations
        optimized_schedule = self._smooth_processing_rates(
            optimized_schedule, normalized_blends)
        
        return optimized_schedule
        
    def _optimize_full_schedule(self,
                             schedule: List[Dict[str, Any]],
                             normalized_blends: List[BlendRecipe],
                             target_rate: float) -> List[Dict[str, Any]]:
        """
        Optimize the entire schedule from beginning to end to ensure consistent processing.
        
        Args:
            schedule: Schedule records to optimize
            normalized_blends: Available normalized blend recipes
            target_rate: Target processing rate
            
        Returns:
            Optimized schedule records
        """
        result = copy.deepcopy(schedule)
        
        # Sort schedule by day to ensure we process in order
        result.sort(key=lambda r: r["Day"])
        
        # We need to track tank levels as we move through the optimization
        current_tank_levels = None
        
        # Process days in chronological order
        for i, record in enumerate(result):
            # First record: Use existing tank levels
            if i == 0:
                current_tank_levels = copy.deepcopy(record["Tank_Levels"])
                continue
                
            # Skip days with no blend index
            if record["Blend_Index"] is None:
                # Try to assign a blend index if possible
                viable_blend = self._find_viable_blend(current_tank_levels, normalized_blends)
                if viable_blend is not None:
                    record["Blend_Index"] = viable_blend
                    record["Strategy"] = "optimizer_assigned_blend"
                else:
                    # If we can't find a viable blend, continue with next record
                    current_tank_levels = copy.deepcopy(record["Tank_Levels"])
                    continue
                
            # Get the current blend
            blend_idx = record["Blend_Index"]
            
            # Skip if blend index is out of range
            if blend_idx >= len(normalized_blends):
                current_tank_levels = copy.deepcopy(record["Tank_Levels"])
                continue
                
            blend = normalized_blends[blend_idx]
            blend_max = blend.max_capacity
            
            # Check if this is a zero or low processing day
            current_rate = record["Processed"]
            
            if current_rate < self.min_processing_rate:
                # Calculate maximum feasible rate based on tank levels
                max_feasible = self._calculate_max_feasible_rate(
                    current_tank_levels, blend.ratios)
                
                # Set to minimum of target rate, blend max capacity, and what's feasible
                optimal_rate = min(target_rate, blend_max, max_feasible)
                
                # If we can process at least some amount
                if optimal_rate > 5.0:  # Minimum threshold to avoid trace processing
                    record["Processed"] = optimal_rate
                    record["Strategy"] = "optimized_from_beginning"
                    
                    # Update crude usage
                    new_usage = {}
                    for crude, ratio in blend.ratios.items():
                        new_usage[crude] = optimal_rate * ratio
                        
                    record["Crude_Usage"] = new_usage
                    
                    # Update tank levels
                    current_tank_levels = self._simulate_tank_withdrawals(
                        current_tank_levels, new_usage)
                    
                    # Update record tank levels
                    record["Tank_Levels"] = copy.deepcopy(current_tank_levels)
                    
                    logger.info(f"Day {record['Day']}: Set from {current_rate:.2f} to {optimal_rate:.2f} kb/day")
            else:
                # Normal processing day, just update tank levels for next iteration
                current_tank_levels = copy.deepcopy(record["Tank_Levels"])
            
            # Process any deliveries to update tank levels
            for delivery in record.get("Deliveries", []):
                allocated = delivery.get("allocated", 0.0)
                if allocated > 0:
                    crude = self._normalize_crude_name(delivery["crude"])
                    self._allocate_delivery_to_tanks(current_tank_levels, crude, allocated)
                    
            result[i] = record
            
        return result
    
    def _select_optimal_blends(self,
                             schedule: List[Dict[str, Any]],
                             normalized_blends: List[BlendRecipe]) -> List[Dict[str, Any]]:
        """
        Select blend recipes that can utilize accumulated crude types.
        
        Args:
            schedule: Schedule records to optimize
            normalized_blends: Available normalized blend recipes
            
        Returns:
            Optimized schedule records with better blend selections
        """
        result = copy.deepcopy(schedule)
        
        # Find days where we're using Base crude consistently but have other crudes available
        for i, record in enumerate(result):
            # Skip days where no processing occurs
            if record["Blend_Index"] is None:
                continue
                
            # Get current crude usage
            current_usage = self._normalize_crude_usage(record["Crude_Usage"])
            
            # If we're mostly using just one crude (e.g., Base) but have others available
            primary_crude = None
            for crude, amount in current_usage.items():
                if amount > 0.7 * sum(current_usage.values()):  # Using mostly one crude
                    primary_crude = crude
                    break
            
            if primary_crude:
                # Check tank levels for accumulated crudes
                accumulated_crudes = {}
                for tank_id, tank_data in record["Tank_Levels"].items():
                    if tank_data["composition"]:
                        for crude, amount in tank_data["composition"].items():
                            normalized_crude = self._normalize_crude_name(crude)
                            accumulated_crudes[normalized_crude] = accumulated_crudes.get(normalized_crude, 0) + amount
                
                # Remove the primary crude we're already using
                if primary_crude in accumulated_crudes:
                    accumulated_crudes.pop(primary_crude)
                
                # Find blend recipes that could use accumulated crudes
                if accumulated_crudes:
                    for blend_index, blend in enumerate(normalized_blends):
                        blend_can_use_accumulated = False
                        for crude in accumulated_crudes:
                            if crude in blend.ratios and blend.ratios[crude] > 0.05:
                                blend_can_use_accumulated = True
                                break
                                
                        # If this blend can use accumulated crudes and isn't the current blend
                        if blend_can_use_accumulated and blend_index != record["Blend_Index"]:
                            # Evaluate if this blend would be better
                            # For now, simple logic: if it can use accumulated crude, it's better
                            record["Blend_Index"] = blend_index
                            record["Strategy"] = f"optimized_use_accumulated_{record['Strategy']}"
                            
                            # Calculate new crude usage based on this blend
                            processed_amount = record["Processed"]
                            new_usage = {}
                            for crude, ratio in blend.ratios.items():
                                new_usage[crude] = processed_amount * ratio
                                
                            record["Crude_Usage"] = new_usage
                            
                            # Update tank levels based on new usage
                            record["Tank_Levels"] = self._simulate_tank_withdrawals(
                                record["Tank_Levels"], new_usage)
                                
                            logger.info(f"Day {record['Day']}: Changed blend to better utilize accumulated crudes")
                            break
                            
            result[i] = record
                            
        return result
    
    def _optimize_low_processing_days(self, 
                                    schedule: List[Dict[str, Any]],
                                    normalized_blends: List[BlendRecipe],
                                    target_rate: float) -> List[Dict[str, Any]]:
        """
        Optimize days with low processing rates to reach the target rate where possible.
        
        Args:
            schedule: Schedule records to optimize
            normalized_blends: Available normalized blend recipes
            target_rate: Target processing rate
            
        Returns:
            Optimized schedule records
        """
        result = copy.deepcopy(schedule)
        
        # Sort by day for sequential processing
        result.sort(key=lambda r: r["Day"])
        
        # Track tank levels through the simulation
        current_tank_levels = None
        
        # Find days with low or zero processing rates
        for i, record in enumerate(result):
            # First day - use the tank levels as-is
            if i == 0:
                current_tank_levels = copy.deepcopy(record["Tank_Levels"])
                continue
                
            # Skip days where blend is None and we can't assign one
            if record["Blend_Index"] is None:
                viable_blend = self._find_viable_blend(current_tank_levels, normalized_blends)
                if viable_blend is not None:
                    record["Blend_Index"] = viable_blend
                    record["Strategy"] = "optimizer_assigned_blend"
                else:
                    # Update tank levels from record for next iteration
                    current_tank_levels = copy.deepcopy(record["Tank_Levels"])
                    continue
                    
            blend_index = record["Blend_Index"]
            
            # Skip if the blend index is out of range
            if blend_index >= len(normalized_blends):
                current_tank_levels = copy.deepcopy(record["Tank_Levels"])
                continue
                
            blend = normalized_blends[blend_index]
            
            # Get the current processing rate
            current_rate = record["Processed"]
            
            # Critical: Respect the blend's maximum capacity
            blend_max_capacity = blend.max_capacity
            
            # Ensure target rate doesn't exceed blend capacity
            adjusted_target = min(target_rate, blend_max_capacity)
            
            # If rate is low or zero, try to increase it
            if current_rate < self.min_processing_rate:
                # Calculate max feasible rate based on tank levels
                max_feasible = self._calculate_max_feasible_rate(
                    current_tank_levels, blend.ratios)
                
                # Set to minimum of target rate, blend max capacity, and what's feasible
                optimal_rate = min(adjusted_target, max_feasible)
                
                # If we can process at least some amount
                if optimal_rate > 5.0:  # Minimum threshold to avoid trace processing
                    record["Processed"] = optimal_rate
                    record["Strategy"] = "optimized_low_day" 
                    
                    # Calculate new crude usage based on this blend
                    new_usage = {}
                    for crude, ratio in blend.ratios.items():
                        new_usage[crude] = optimal_rate * ratio
                        
                    record["Crude_Usage"] = new_usage
                    
                    # Update tank levels
                    current_tank_levels = self._simulate_tank_withdrawals(
                        current_tank_levels, new_usage)
                    
                    # Update record tank levels
                    record["Tank_Levels"] = copy.deepcopy(current_tank_levels)
                    
                    logger.info(f"Day {record['Day']}: Increased from {current_rate:.2f} to {optimal_rate:.2f} kb/day")
                    
            else:
                # Normal processing day, just update tank levels for next iteration
                current_tank_levels = copy.deepcopy(record["Tank_Levels"])
            
            # Process any deliveries to update tank levels
            for delivery in record.get("Deliveries", []):
                allocated = delivery.get("allocated", 0.0)
                if allocated > 0:
                    crude = self._normalize_crude_name(delivery["crude"])
                    self._allocate_delivery_to_tanks(current_tank_levels, crude, allocated)
                    
            result[i] = record
        
        return result
    
    def _optimize_end_period(self, 
                           schedule: List[Dict[str, Any]],
                           normalized_blends: List[BlendRecipe],
                           planning_horizon: int) -> List[Dict[str, Any]]:
        """
        Optimize processing in the end period for better inventory utilization.
        
        Args:
            schedule: Schedule records to optimize
            normalized_blends: Available normalized blend recipes
            planning_horizon: Planning horizon in days
            
        Returns:
            Optimized schedule records
        """
        if not schedule:
            return schedule
            
        result = copy.deepcopy(schedule)
        
        # Calculate end period
        end_period_start = max(1, planning_horizon - self.end_period_window)
        
        # Get the average processing rate in the first part of the horizon
        early_rates = [r["Processed"] for r in result if r["Day"] < end_period_start]
        if early_rates:
            avg_early_rate = sum(early_rates) / len(early_rates)
            target_end_rate = max(self.min_processing_rate, avg_early_rate * 0.8)
        else:
            target_end_rate = self.min_processing_rate
            
        # Calculate remaining inventory for end period, normalizing crude names
        remaining_inventory = self._calculate_remaining_inventory(
            schedule, end_period_start)
            
        # Calculate total remaining volume
        remaining_volume = sum(remaining_inventory.values())
        remaining_days = planning_horizon - end_period_start + 1
        
        if remaining_days <= 0:
            return result
            
        # Calculate consistent daily rate for end period
        consistent_rate = remaining_volume / remaining_days
        
        # Use the higher of the consistent rate or target end rate
        optimal_rate = max(consistent_rate, target_end_rate)
        
        logger.info(f"End period optimization - Remaining volume: {remaining_volume:.2f}")
        logger.info(f"End period optimization - Consistent daily rate: {consistent_rate:.2f}")
        logger.info(f"End period optimization - Target end rate: {optimal_rate:.2f}")
        
        # We need to track tank levels as we move through the optimization
        # Start with the levels at the beginning of the end period
        current_tank_levels = None
        
        # Sort by day
        result.sort(key=lambda r: r["Day"])
        
        # Update processing rates for the end period
        for day in range(end_period_start, planning_horizon + 1):
            idx = next((i for i, r in enumerate(result) if r["Day"] == day), None)
            
            if idx is None:
                continue
                
            current_record = result[idx]
            
            # If we don't have tank levels yet (first day of end period),
            # find the previous day's record to get those levels
            if current_tank_levels is None:
                prev_idx = next((i for i, r in enumerate(result) if r["Day"] == day - 1), None)
                if prev_idx is not None:
                    current_tank_levels = copy.deepcopy(result[prev_idx]["Tank_Levels"])
                else:
                    current_tank_levels = copy.deepcopy(current_record["Tank_Levels"])
            
            # Choose a blend recipe if one isn't assigned yet
            if current_record["Blend_Index"] is None:
                # Find a viable blend based on available crude in tanks
                viable_blend_index = self._find_viable_blend(
                    current_tank_levels, normalized_blends)
                    
                if viable_blend_index is not None:
                    current_record["Blend_Index"] = viable_blend_index
                    current_record["Strategy"] = "optimized_blend_selection"
                else:
                    # If no viable blend is found, skip this day but update tank levels
                    current_tank_levels = copy.deepcopy(current_record["Tank_Levels"])
                    continue
                    
            blend_index = current_record["Blend_Index"]
            
            # Skip if blend index is out of range
            if blend_index >= len(normalized_blends):
                current_tank_levels = copy.deepcopy(current_record["Tank_Levels"])
                continue
                
            # Get the blend and its maximum capacity
            blend = normalized_blends[blend_index]
            blend_max_capacity = blend.max_capacity
            
            # Ensure optimal rate respects blend capacity
            blend_constrained_rate = min(optimal_rate, blend_max_capacity)
            
            # Further constrain by what's actually available in tanks
            max_rate = self._calculate_max_feasible_rate(
                current_tank_levels, blend.ratios)
            
            feasible_rate = min(blend_constrained_rate, max_rate)
            
            # For the last few days, gradually adjust to use all remaining inventory
            remaining_days = planning_horizon - day + 1
            if remaining_days <= 3:
                adjusted_rate = remaining_volume / remaining_days if remaining_days > 0 else 0
                # Still respect the blend maximum capacity and feasibility
                adjusted_rate = min(adjusted_rate, feasible_rate)
                
                if adjusted_rate > 0:
                    current_record["Processed"] = adjusted_rate
                    logger.info(f"Day {day}: Final adjustment to {adjusted_rate:.2f} kb/day (blend max: {blend_max_capacity:.2f})")
                    remaining_volume -= adjusted_rate
                else:
                    # If no processing is feasible, set to 0
                    current_record["Processed"] = 0.0
                    logger.info(f"Day {day}: No processing feasible")
            else:
                # Otherwise use the optimal rate (constrained by blend capacity and feasibility)
                if feasible_rate > 0:
                    current_record["Processed"] = feasible_rate
                    remaining_volume -= feasible_rate
                    logger.info(f"Day {day}: Set to feasible rate {feasible_rate:.2f} kb/day (blend max: {blend_max_capacity:.2f})")
                else:
                    # If no processing is feasible, set to 0
                    current_record["Processed"] = 0.0
                    logger.info(f"Day {day}: No processing feasible")
            
            # Only update crude usage and tank levels if we're processing
            if current_record["Processed"] > 0:
                # Update crude usage
                new_usage = {}
                for crude, ratio in blend.ratios.items():
                    new_usage[crude] = current_record["Processed"] * ratio
                    
                current_record["Crude_Usage"] = new_usage
                
                # Simulate withdrawals from tanks and update the current tank levels
                current_tank_levels = self._simulate_tank_withdrawals(
                    current_tank_levels, new_usage)
                
                # Update the record's tank levels
                current_record["Tank_Levels"] = copy.deepcopy(current_tank_levels)
            
            # Mark as optimized
            current_record["Strategy"] = f"smooth" if current_record["Strategy"] == "low_inventory" and current_record["Processed"] > 0 else current_record["Strategy"]
            
            # Update the record
            result[idx] = current_record
            
            # Process any deliveries occurring on this day to update tank levels
            for delivery in current_record.get("Deliveries", []):
                # We only care about allocated deliveries
                allocated = delivery.get("allocated", 0.0)
                if allocated <= 0:
                    continue
                    
                crude = delivery["crude"]
                
                # Simplified allocation logic for the optimizer
                self._allocate_delivery_to_tanks(
                    current_tank_levels, crude, allocated)
        
        return result
        
    def _smooth_processing_rates(self,
                               schedule: List[Dict[str, Any]],
                               normalized_blends: List[BlendRecipe]) -> List[Dict[str, Any]]:
        """
        Smooth processing rates to avoid large variations between consecutive days.
        Apply a rolling window approach to balance processing across days.
        
        Args:
            schedule: Schedule records to optimize
            normalized_blends: Available normalized blend recipes
            
        Returns:
            Schedule records with smoothed processing rates
        """
        if not schedule:
            return schedule
            
        result = copy.deepcopy(schedule)
        result.sort(key=lambda r: r["Day"])
        
        # Run multiple iterations of smoothing
        for iteration in range(self.max_iterations):
            # Skip if we only have one day
            if len(result) <= 1:
                break
                
            # Keep track of changes made
            changes_made = False
            
            # Apply smoothing window
            for i in range(1, len(result) - 1):
                prev_record = result[i-1]
                current_record = result[i]
                next_record = result[i+1]
                
                # Skip days without blend indexes
                if (current_record["Blend_Index"] is None or 
                    prev_record["Blend_Index"] is None or 
                    next_record["Blend_Index"] is None):
                    continue
                    
                # Skip if any blend index is invalid
                if (current_record["Blend_Index"] >= len(normalized_blends) or
                    prev_record["Blend_Index"] >= len(normalized_blends) or
                    next_record["Blend_Index"] >= len(normalized_blends)):
                    continue
                
                # Current processing rate and rates of neighboring days
                current_rate = current_record["Processed"]
                prev_rate = prev_record["Processed"]
                next_rate = next_record["Processed"]
                
                # Skip days with zero processing (they might have constraints)
                if current_rate <= 0.001 or prev_rate <= 0.001 or next_rate <= 0.001:
                    continue
                
                # Calculate the average of neighboring rates
                avg_rate = (prev_rate + next_rate) / 2
                
                # If current rate differs significantly from average (more than 30%)
                if abs(current_rate - avg_rate) > 0.3 * avg_rate:
                    # Get current blend
                    blend = normalized_blends[current_record["Blend_Index"]]
                    blend_max = blend.max_capacity
                    
                    # Calculate new rate, limited by blend max capacity
                    new_rate = min(avg_rate, blend_max)
                    
                    # Simulate to check if this new rate is feasible
                    # First, simulate tank levels after previous day
                    simulated_tanks = copy.deepcopy(prev_record["Tank_Levels"])
                    
                    # Subtract previous day's usage
                    for crude, amount in prev_record.get("Crude_Usage", {}).items():
                        if amount > 0:
                            is_withdrawal_ok = self._simulate_withdrawal_feasibility(
                                simulated_tanks, crude, amount)
                    
                    # Process any deliveries from previous day
                    for delivery in prev_record.get("Deliveries", []):
                        allocated = delivery.get("allocated", 0.0)
                        if allocated > 0:
                            crude = self._normalize_crude_name(delivery["crude"])
                            self._allocate_delivery_to_tanks(
                                simulated_tanks, crude, allocated)
                    
                    # Check if new rate is feasible with these tank levels
                    new_usage = {}
                    for crude, ratio in blend.ratios.items():
                        new_usage[crude] = new_rate * ratio
                    
                    # Simulate withdrawal feasibility for each crude type
                    is_feasible = True
                    for crude, amount in new_usage.items():
                        if not self._simulate_withdrawal_feasibility(
                            simulated_tanks, crude, amount):
                            is_feasible = False
                            break
                    
                    # If feasible, update the rate
                    if is_feasible and new_rate > 5.0:  # Avoid trace processing
                        current_record["Processed"] = new_rate
                        current_record["Strategy"] = "smoothed"
                        
                        # Update crude usage
                        current_record["Crude_Usage"] = new_usage
                        
                        # Update tank levels (use previous day's levels as starting point)
                        updated_tanks = self._simulate_tank_withdrawals(
                            copy.deepcopy(prev_record["Tank_Levels"]), new_usage)
                            
                        # Process deliveries
                        for delivery in current_record.get("Deliveries", []):
                            allocated = delivery.get("allocated", 0.0)
                            if allocated > 0:
                                crude = self._normalize_crude_name(delivery["crude"])
                                self._allocate_delivery_to_tanks(
                                    updated_tanks, crude, allocated)
                                
                        # Update tank levels in record
                        current_record["Tank_Levels"] = updated_tanks
                        
                        # Update the record
                        result[i] = current_record
                        
                        changes_made = True
                        logger.info(f"Day {current_record['Day']}: Smoothed from {current_rate:.2f} to {new_rate:.2f} kb/day")
            
            # If no changes were made in this iteration, stop
            if not changes_made:
                break
                
        return result
    
    def _simulate_withdrawal_feasibility(self,
                                       tank_levels: Dict[str, Dict[str, Any]],
                                       crude: str,
                                       amount: float) -> bool:
        """
        Check if a withdrawal of a certain amount of crude is feasible.
        
        Args:
            tank_levels: Current tank levels
            crude: Crude type to withdraw
            amount: Amount to withdraw
            
        Returns:
            True if withdrawal is feasible, False otherwise
        """
        normalized_crude = self._normalize_crude_name(crude)
        available = 0.0
        
        # Count available amount of this crude type
        for tank_id, tank in tank_levels.items():
            for crude_type, tank_amount in tank["composition"].items():
                if self._normalize_crude_name(crude_type) == normalized_crude:
                    available += tank_amount
        
        # Check if we have enough
        return available >= amount
    
    def _find_viable_blend(self, 
                         tank_levels: Dict[str, Dict[str, Any]], 
                         blends: List[BlendRecipe]) -> Optional[int]:
        """
        Find a viable blend recipe based on available crude in tanks.
        
        Args:
            tank_levels: Current tank levels
            blends: Available blend recipes
            
        Returns:
            Index of viable blend or None if none found
        """
        # First, find available crude types in tanks
        available_crudes = {}
        for tank_id, tank in tank_levels.items():
            for crude, amount in tank["composition"].items():
                crude = self._normalize_crude_name(crude)
                available_crudes[crude] = available_crudes.get(crude, 0.0) + amount
                
        # Find a blend that can use the available crudes
        for i, blend in enumerate(blends):
            can_use = True
            for crude in blend.ratios:
                if crude not in available_crudes or available_crudes[crude] < 1.0:
                    can_use = False
                    break
                    
            if can_use:
                return i
                
        # If we haven't found a perfect match, look for blends with
        # majority of required crudes available
        best_blend_idx = None
        best_match_score = 0
        
        for i, blend in enumerate(blends):
            match_score = 0
            total_ratio = sum(blend.ratios.values())
            
            for crude, ratio in blend.ratios.items():
                if crude in available_crudes and available_crudes[crude] >= 1.0:
                    match_score += ratio / total_ratio
                    
            if match_score > best_match_score:
                best_match_score = match_score
                best_blend_idx = i
                
        # Return best match if it uses at least 50% of available crudes
        if best_match_score >= 0.5:
            return best_blend_idx
            
        return None
        
    def _calculate_max_feasible_rate(self, 
                                   tank_levels: Dict[str, Dict[str, Any]], 
                                   blend_ratios: Dict[str, float]) -> float:
        """
        Calculate maximum feasible processing rate based on tank levels.
        
        Args:
            tank_levels: Current tank levels
            blend_ratios: Blend recipe ratios
            
        Returns:
            Maximum feasible processing rate
        """
        # First, find available crude types in tanks
        available_crudes = {}
        for tank_id, tank in tank_levels.items():
            for crude, amount in tank["composition"].items():
                crude = self._normalize_crude_name(crude)
                available_crudes[crude] = available_crudes.get(crude, 0.0) + amount
                
        # Calculate maximum rate based on limiting crude
        max_rate = float('inf')
        
        for crude, ratio in blend_ratios.items():
            if ratio <= 0.001:  # Skip negligible ratios
                continue
                
            available = available_crudes.get(crude, 0.0)
            if available <= 0.001:  # Crude not available
                return 0.0
                
            # Calculate how much processing this crude allows
            rate = available / ratio
            max_rate = min(max_rate, rate)
            
        return max_rate
    
    def _allocate_delivery_to_tanks(self, 
                                  tank_levels: Dict[str, Dict[str, Any]], 
                                  crude: str, 
                                  volume: float) -> None:
        """
        Allocate a delivery to tanks.
        
        Args:
            tank_levels: Current tank levels
            crude: Crude type
            volume: Volume to allocate
        """
        normalized_crude = self._normalize_crude_name(crude)
        remaining = volume
        
        # First pass: allocate to matching tanks
        for tank_id, tank in tank_levels.items():
            # Skip full tanks
            if tank["level"] >= 400:  # Assuming max tank capacity is 400
                continue
                
            # If tank is empty or matches crude type
            if tank["level"] == 0 or tank["crude"] == normalized_crude:
                space = 400 - tank["level"]  # Assuming max tank capacity is 400
                allocated = min(space, remaining)
                
                # Update tank
                if tank["level"] == 0:
                    # Empty tank
                    tank["crude"] = normalized_crude
                    tank["composition"] = {normalized_crude: allocated}
                else:
                    # Tank with matching crude
                    tank["composition"][normalized_crude] = tank["composition"].get(normalized_crude, 0.0) + allocated
                
                tank["level"] += allocated
                remaining -= allocated
                
                if remaining <= 0.001:
                    return
        
        # Second pass: allocate to any tank with space
        for tank_id, tank in tank_levels.items():
            # Skip full tanks
            if tank["level"] >= 400:  # Assuming max tank capacity is 400
                continue
                
            space = 400 - tank["level"]  # Assuming max tank capacity is 400
            allocated = min(space, remaining)
            
            # Update tank
            if tank["level"] == 0:
                # Empty tank
                tank["crude"] = normalized_crude
                tank["composition"] = {normalized_crude: allocated}
            else:
                # Tank with existing crude
                tank["composition"][normalized_crude] = tank["composition"].get(normalized_crude, 0.0) + allocated
            
            tank["level"] += allocated
            remaining -= allocated
            
            if remaining <= 0.001:
                return
                
        # If we couldn't allocate everything, log it
        if remaining > 0.001:
            logger.warning(f"Could not allocate {remaining:.2f} kb of {normalized_crude} due to insufficient tank space")

    def _calculate_remaining_inventory(self, 
                                     schedule: List[Dict[str, Any]],
                                     end_period_start: int) -> Dict[str, float]:
        """
        Calculate remaining crude inventory at the start of the end period.
        
        Args:
            schedule: Schedule records
            end_period_start: First day of the end period
            
        Returns:
            Dictionary mapping crude types to available volumes
        """
        # Initialize inventory tracking with all crude types
        inventory = {}
        
        # Collect all crude types
        for record in schedule:
            # Add crude types from deliveries
            for delivery in record.get("Deliveries", []):
                crude = self._normalize_crude_name(delivery["crude"])
                if crude not in inventory:
                    inventory[crude] = 0.0
            
            # Add crude types from usage
            for crude in record.get("Crude_Usage", {}).keys():
                normalized_crude = self._normalize_crude_name(crude)
                if normalized_crude not in inventory:
                    inventory[normalized_crude] = 0.0
        
        # Process all records up to the end period start
        for record in schedule:
            day = record["Day"]
            
            # Add deliveries
            if day < end_period_start:
                for delivery in record.get("Deliveries", []):
                    crude = self._normalize_crude_name(delivery["crude"])
                    allocated = delivery.get("allocated", 0.0)
                    inventory[crude] += allocated
                
                # Subtract crude usage
                for crude, used in record.get("Crude_Usage", {}).items():
                    normalized_crude = self._normalize_crude_name(crude)
                    inventory[normalized_crude] -= used
                    # Ensure no negative values due to rounding
                    inventory[normalized_crude] = max(0.0, inventory[normalized_crude])
        
        # Process all future deliveries in the end period
        for record in schedule:
            day = record["Day"]
            
            # Only consider deliveries in the end period
            if day >= end_period_start:
                for delivery in record.get("Deliveries", []):
                    crude = self._normalize_crude_name(delivery["crude"])
                    allocated = delivery.get("allocated", 0.0)
                    inventory[crude] += allocated
        
        return inventory