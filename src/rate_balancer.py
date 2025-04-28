import logging
import copy
from typing import List, Dict, Any, Optional, Tuple

from src.models.inventory import CrudeInventory
from src.models.tank import Tank
from src.models.blend import BlendRecipe
from src.simulator import simulate_blend
from src.scheduler import Scheduler  # Import the Scheduler class
import json 

logger = logging.getLogger(__name__)

class RateBalancer:
    """
    Module for balancing processing rates across the planning period
    to avoid inventory shortages in later days.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the RateBalancer module.
        
        Args:
            params: Configuration parameters
        """
        self.params = params or {}
        self.max_iterations = self.params.get("max_iterations", 5)
        self.smoothing_factor = self.params.get("smoothing_factor", 0.8)
        self.early_day_threshold = self.params.get("early_day_threshold", 10)  # Days considered "early"
        self.early_reduction_pct = self.params.get("early_reduction_pct", 0.15)  # Maximum reduction for early days
        self.min_processing_rate = self.params.get("min_processing_rate", 20.0)  # Minimum processing rate
        
    def balance_schedule(self, 
                         schedule: List[Dict[str, Any]],
                         blends: List[BlendRecipe],
                         initial_inventory: CrudeInventory,
                         tanks: List[Tank]) -> List[Dict[str, Any]]:
        """
        Balance the processing rates across the schedule to avoid inventory shortages.
        
        Args:
            schedule: Original schedule records
            blends: Blend recipes
            initial_inventory: Initial crude inventory
            tanks: Tank configurations
            
        Returns:
            Balanced schedule records
        """
        if not schedule:
            logger.warning("No schedule provided to balance.")
            return schedule
            
        logger.info("Starting schedule rate balancing")
        logger.info(f"Original schedule has {len(schedule)} days")
        
        # Identify days with zero processing due to inventory shortages
        zero_days = [record["Day"] for record in schedule if record["Processed"] <= 0.001]
        if not zero_days:
            logger.info("No zero processing days found. Schedule is already balanced.")
            return schedule
            
        logger.info(f"Found {len(zero_days)} days with zero processing: {zero_days}")
        
        # Find the first day with zero processing
        first_zero_day = min(zero_days)
        logger.info(f"First day with zero processing: Day {first_zero_day}")
        
        # Analyze crude usage pattern to identify which crude types ran out
        exhausted_crudes = self._identify_exhausted_crudes(schedule, first_zero_day)
        logger.info(f"Identified exhausted crude types: {exhausted_crudes}")
        
        # Calculate how much to reduce early processing to preserve these crudes
        best_schedule = copy.deepcopy(schedule)
        best_total_processed = sum(record["Processed"] for record in schedule)
        
        # Try different reduction percentages
        for iteration in range(1, self.max_iterations + 1):
            # Try with increasing reduction percentages
            reduction_pct = self.early_reduction_pct * iteration / self.max_iterations
            
            # Create a new balanced schedule
            balanced_schedule = self._create_balanced_schedule(
                schedule=copy.deepcopy(schedule),
                exhausted_crudes=exhausted_crudes,
                first_zero_day=first_zero_day,
                reduction_pct=reduction_pct
            )
            
            # Simulate the new schedule to check its feasibility
            simulated_schedule = self._simulate_schedule(
                balanced_schedule,
                blends,
                copy.deepcopy(initial_inventory),
                copy.deepcopy(tanks)
            )
            
            # Calculate total processed volume
            total_processed = sum(record["Processed"] for record in simulated_schedule)
            
            logger.info(f"Iteration {iteration}: Reduction {reduction_pct:.2%}, "
                        f"Total processed: {total_processed:.2f}")
            
            # Check if this schedule is better than our best so far
            if total_processed > best_total_processed:
                best_schedule = simulated_schedule
                best_total_processed = total_processed
                logger.info(f"New best schedule found: {total_processed:.2f}")
        
        # Count zero processing days after balancing
        final_zero_days = [record["Day"] for record in best_schedule if record["Processed"] <= 0.001]
        
        logger.info(f"Schedule balancing complete. Original total: {sum(record['Processed'] for record in schedule):.2f}")
        logger.info(f"New total: {best_total_processed:.2f}")
        logger.info(f"Zero processing days reduced from {len(zero_days)} to {len(final_zero_days)}")
        
        return best_schedule
    
    def _identify_exhausted_crudes(self, schedule: List[Dict[str, Any]], first_zero_day: int) -> List[str]:
        """
        Identify which crude types ran out by analyzing crude usage before first zero day.
        
        Args:
            schedule: Schedule records
            first_zero_day: Day index when processing first reached zero
            
        Returns:
            List of crude types that were exhausted
        """
        # Get the records before the first zero day
        prior_records = [r for r in schedule if r["Day"] < first_zero_day]
        
        if not prior_records:
            logger.warning("No records found before first zero processing day")
            return []
        
        # Track crude usage by day to identify trends
        crude_usage_by_day = {}
        all_crudes = set()
        
        for record in prior_records:
            day = record["Day"]
            crude_usage = record.get("Crude_Usage", {})
            crude_usage_by_day[day] = crude_usage
            all_crudes.update(crude_usage.keys())
        
        # Look for crudes that had decreasing trends or sudden drops to zero
        critical_crudes = []
        
        for crude in all_crudes:
            # Get usage values for this crude across days
            usage_trend = [crude_usage_by_day.get(day, {}).get(crude, 0) 
                           for day in range(1, first_zero_day)]
            
            # Skip if usage is mostly zero
            if sum(usage_trend) < 0.1:
                continue
            
            # Check if trend is decreasing rapidly or drops to zero near end
            if len(usage_trend) >= 3:
                # Check last few days
                last_three = usage_trend[-3:]
                if last_three[-1] < 0.1 and max(last_three[:-1]) > 1.0:
                    # Usage dropped to near zero at the end
                    critical_crudes.append(crude)
                elif all(last_three[i] < last_three[i-1] * 0.7 for i in range(1, len(last_three))):
                    # Decreasing by more than 30% each day
                    critical_crudes.append(crude)
        
        # If we couldn't identify specific crudes, return the ones used on the day before
        if not critical_crudes and prior_records:
            last_day_record = max(prior_records, key=lambda r: r["Day"])
            critical_crudes = [c for c, v in last_day_record.get("Crude_Usage", {}).items() if v > 0.1]
        
        return critical_crudes
    
    def _create_balanced_schedule(self,
                                 schedule: List[Dict[str, Any]],
                                 exhausted_crudes: List[str],
                                 first_zero_day: int,
                                 reduction_pct: float) -> List[Dict[str, Any]]:
        """
        Create a balanced schedule by reducing processing rates in early days.
        
        Args:
            schedule: Original schedule records
            exhausted_crudes: List of crude types that ran out
            first_zero_day: Day when processing first reached zero
            reduction_pct: Percentage to reduce early day processing
            
        Returns:
            Balanced schedule records
        """
        # Make a copy of the schedule to avoid modifying the original
        balanced_schedule = copy.deepcopy(schedule)
        
        # Calculate total crude usage for exhausted crudes in early days
        early_days = [day for day in range(1, min(first_zero_day, self.early_day_threshold + 1))]
        
        # Calculate saved volume from reduction
        saved_volume = 0
        
        # Reduce processing rates for early days
        for record in balanced_schedule:
            day = record["Day"]
            
            # Only modify early days
            if day in early_days and record["Processed"] > self.min_processing_rate:
                original_rate = record["Processed"]
                
                # Calculate new reduced rate
                reduced_rate = max(
                    self.min_processing_rate,
                    original_rate * (1 - reduction_pct)
                )
                
                # Calculate volume saved
                volume_saved = original_rate - reduced_rate
                saved_volume += volume_saved
                
                # Update record with reduced processing rate
                record["Processed"] = reduced_rate
                record["Strategy"] = f"{record['Strategy']}_balanced"
                
                # Also adjust crude usage proportionally
                for crude, usage in record.get("Crude_Usage", {}).items():
                    if usage > 0:
                        record["Crude_Usage"][crude] = usage * (reduced_rate / original_rate)
                        
                logger.info(f"Day {day}: Reduced processing from {original_rate:.2f} to {reduced_rate:.2f}")
        
        logger.info(f"Total volume saved from early day reductions: {saved_volume:.2f}")
        
        # Now redistribute saved volume to zero-processing days
        zero_days = [r for r in balanced_schedule if r["Day"] >= first_zero_day and r["Processed"] <= 0.001]
        
        if zero_days and saved_volume > 0:
            # Simple approach: distribute saved volume equally among zero days
            rate_per_day = saved_volume / len(zero_days)
            
            for record in zero_days:
                record["Processed"] = rate_per_day
                record["Strategy"] = "redistributed"
                logger.info(f"Day {record['Day']}: Redistributed {rate_per_day:.2f}")
        
        return balanced_schedule
    
    def _simulate_schedule(self,
                          proposed_schedule: List[Dict[str, Any]],
                          blends: List[BlendRecipe],
                          inventory: CrudeInventory,
                          tanks: List[Tank]) -> List[Dict[str, Any]]:
        """
        Simulate a proposed schedule to ensure it's feasible.
        Adjust rates if simulation uncovers inventory issues.
        
        Args:
            proposed_schedule: Proposed schedule records
            blends: Blend recipes
            inventory: Initial crude inventory
            tanks: Tank configurations
            
        Returns:
            Feasible schedule records
        """
        simulated_schedule = copy.deepcopy(proposed_schedule)
        tank_dict = {t.id: t for t in tanks}
        
        # Keep track of available crude inventory
        remaining_inventory = copy.deepcopy(inventory)
        
        # Process each day in the schedule
        for i, record in enumerate(simulated_schedule):
            day = record["Day"]
            blend_idx = record["Blend_Index"]
            
            # Skip if no blend or processing
            if blend_idx is None or record["Processed"] <= 0.001:
                continue
            
            # Get the blend for this day
            try:
                blend = blends[blend_idx]
            except (IndexError, TypeError):
                logger.warning(f"Day {day}: Invalid blend index {blend_idx}")
                # Set processing to zero for this day
                record["Processed"] = 0
                continue
            
            # Check if we have enough inventory for the proposed processing
            processing_rate = record["Processed"]
            feasible = True
            
            # Check each crude required by the blend
            for crude, ratio in blend.ratios.items():
                required = processing_rate * ratio
                
                if not remaining_inventory.available(crude, required):
                    # Not enough inventory - calculate how much we can process
                    available = remaining_inventory.get(crude)
                    if available <= 0:
                        # No inventory at all - can't process
                        feasible = False
                        record["Processed"] = 0
                        logger.info(f"Day {day}: No {crude} available, set processing to 0")
                        break
                    
                    # Adjust processing rate based on available inventory
                    max_rate = available / ratio
                    processing_rate = min(processing_rate, max_rate)
                    feasible = False
                    logger.info(f"Day {day}: Limited by {crude}, "
                               f"reduced processing to {processing_rate:.2f}")
            
            if not feasible:
                # Skip to next day if we can't process anything
                continue
                
            # Update processing rate in record
            record["Processed"] = processing_rate
            
            # Remove used crude from inventory
            for crude, ratio in blend.ratios.items():
                used = processing_rate * ratio
                if used > 0:
                    try:
                        remaining_inventory.remove(crude, used)
                        # Update crude usage in record
                        record["Crude_Usage"][crude] = used
                    except ValueError as e:
                        # This shouldn't happen since we checked availability
                        logger.error(f"Day {day}: Error removing {used} of {crude}: {e}")
                        # Try to recover by removing what's available
                        available = remaining_inventory.get(crude)
                        if available > 0:
                            remaining_inventory.remove(crude, available)
                            record["Crude_Usage"][crude] = available
                        
        return simulated_schedule


# Add a function to the scheduler.py file to use this balancer
def balance_existing_schedule(
    config_path: str,
    input_schedule_path: str = None,
    output_dir: str = "results/balanced"
) -> Dict[str, Any]:
    """
    Balance an existing schedule to avoid inventory shortages.
    
    Args:
        config_path: Path to configuration file
        input_schedule_path: Path to the input schedule JSON file (if None, scheduler must have already run)
        output_dir: Directory to save balanced results
        
    Returns:
        Dictionary with balance results
    """
    try:
        logger.info(f"Starting schedule balancing with config: {config_path}")
        scheduler = Scheduler(config_path)
        
        # Load data before balancing
        scheduler.load_data()
        
        # Load the schedule if provided
        if input_schedule_path:
            logger.info(f"Loading schedule from {input_schedule_path}")
            with open(input_schedule_path, 'r') as f:
                scheduler.schedule_records = json.load(f)
        
        # Ensure we have a schedule to balance
        if not scheduler.schedule_records:
            message = "No schedule records available for balancing"
            logger.error(message)
            return {
                "status": "error",
                "message": message
            }
        
        # Create balancer with parameters
        balancer_params = scheduler.params.get("balancer_params", {})
        if not balancer_params:
            # Set default parameters if not in config
            balancer_params = {
                "max_iterations": 5,
                "smoothing_factor": 0.8,
                "early_day_threshold": 10,
                "early_reduction_pct": 0.15,
                "min_processing_rate": 20.0
            }
            
        # Create rate balancer
        from src.rate_balancer import RateBalancer
        balancer = RateBalancer(params=balancer_params)
        
        # Run balance operation
        logger.info("Running schedule rate balancing")
        original_records = copy.deepcopy(scheduler.schedule_records)
        scheduler.schedule_records = balancer.balance_schedule(
            schedule=scheduler.schedule_records,
            blends=scheduler.blends,
            initial_inventory=copy.deepcopy(scheduler.inventory),
            tanks=scheduler.tanks
        )
        
        # Calculate improvement metrics
        original_processed = sum(r.get("Processed", 0.0) for r in original_records)
        balanced_processed = sum(r.get("Processed", 0.0) for r in scheduler.schedule_records)
        
        # Count zero processing days before and after
        original_zero_days = sum(1 for r in original_records if r.get("Processed", 0.0) <= 0.001)
        balanced_zero_days = sum(1 for r in scheduler.schedule_records if r.get("Processed", 0.0) <= 0.001)
        
        # Save balanced results
        files = scheduler.save_results(output_dir)
        
        # Return results
        return {
            "status": "success",
            "original_processed": original_processed,
            "balanced_processed": balanced_processed,
            "improvement": balanced_processed - original_processed,
            "improvement_percentage": ((balanced_processed / original_processed) - 1) * 100 if original_processed > 0 else 0,
            "original_zero_days": original_zero_days,
            "balanced_zero_days": balanced_zero_days,
            "files": files,
            "days_scheduled": len(scheduler.schedule_records),
            "message": "Schedule balancing completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error during schedule balancing: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }