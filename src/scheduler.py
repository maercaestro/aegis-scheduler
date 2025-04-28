import os
import json
import logging
import csv
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import copy

from src.data_loader import DataLoader, ConfigValidationError
from src.models.tank import Tank
from src.models.inventory import CrudeInventory
from src.models.blend import BlendRecipe
from src.models.delivery import Delivery
from src.allocation import allocate_delivery
from src.simulator import simulate_blend, check_tank_inventory, find_max_feasible_amount
from src.heuristics import choose_strategy, calculate_rationing_factor, enforce_final_days
from src.reporter import to_dataframe, save_reports
from src.models.vessel_optimizer import VesselOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler.log')
    ]
)
logger = logging.getLogger(__name__)

class Scheduler:
    """
    Main scheduler class that orchestrates the entire planning process.
    """
    
    def __init__(self, config_path: str = None, data_loader: DataLoader = None):
        """
        Initialize the scheduler.
        
        Args:
            config_path: Path to configuration file
            data_loader: Optional pre-configured DataLoader instance
        """
        # Initialize data loader
        if data_loader:
            self.loader = data_loader
        elif config_path:
            self.loader = DataLoader(config_path)
        else:
            self.loader = DataLoader()
        
        # Initialize state variables
        self.tanks: List[Tank] = []
        self.inventory: Optional[CrudeInventory] = None
        self.deliveries: List[Delivery] = []
        self.future_deliveries: Dict[int, List[Delivery]] = {}
        self.blends: List[BlendRecipe] = []
        self.params: Dict[str, Any] = {}
        self.scheduler_params: Dict[str, Any] = {}
        self.vessel_params: Dict[str, Any] = {}
        self.optimizer_params: Dict[str, Any] = {}
        self.schedule_records: List[Dict[str, Any]] = []
        self.vessel_results: Optional[Dict[str, Any]] = None
        
        # Output paths
        self.output_dir = "results"
    
    def load_data(self):
        """
        Load all data required for scheduling.
        """
        logger.info("Loading data...")
        
        # Load tanks
        self.tanks = self.loader.get_tanks()
        logger.info(f"Loaded {len(self.tanks)} tanks")
        
        # Load inventory
        self.inventory = self.loader.get_inventory(self.tanks)
        logger.info(f"Loaded inventory with {len(self.inventory.get_all_crudes())} crude types")
        
        # Load deliveries
        self.deliveries, self.future_deliveries = self.loader.get_deliveries()
        logger.info(f"Loaded {len(self.deliveries)} current deliveries and {sum(len(d) for d in self.future_deliveries.values())} future deliveries")
        
        # Load blend recipes
        self.blends = self.loader.load_blend_recipes()
        logger.info(f"Loaded {len(self.blends)} blend recipes")
        
        # Load parameters
        config_data = self.loader.get_config()
        
        # Extract different parameter sections
        self.scheduler_params = config_data.get("scheduler_params", {})
        self.vessel_params = config_data.get("vessel_params", {})
        self.optimizer_params = config_data.get("optimizer_params", {})
        
        # For backward compatibility, copy top-level params to scheduler_params
        for key, value in config_data.items():
            if key not in ["scheduler_params", "vessel_params", "optimizer_params"]:
                if key not in self.scheduler_params:
                    self.scheduler_params[key] = value
        
        # Create merged params dict for backward compatibility
        self.params = {**self.scheduler_params}
        # Add vessel params to the merged params
        for key, value in self.vessel_params.items():
            if key not in self.params:
                self.params[key] = value
        
        logger.info(f"Loaded configuration parameters")
        
        # Validate planning horizon
        if "planning_horizon" not in self.params:
            self.params["planning_horizon"] = 31  # Default value
            self.scheduler_params["planning_horizon"] = 31
            
        logger.debug(f"Planning horizon: {self.params['planning_horizon']} days")
    
    def run_vessel_optimization(self, 
                              data_path: Optional[str] = None, 
                              multiple_solutions: bool = False,
                              num_solutions: int = 3,
                              optimality_gap: float = 0.05) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run vessel optimization to determine delivery schedule.
        
        Args:
            data_path: Path to vessel data file
            multiple_solutions: Whether to generate multiple solutions
            num_solutions: Number of solutions to generate if multiple_solutions=True
            optimality_gap: Maximum gap from optimal (0.05 = 5% worse than optimal)
            
        Returns:
            Vessel optimization results
        """
        # Load vessel data
        vessel_data = self.loader.load_vessel_data(data_path)
        
        # Create vessel optimizer
        optimizer = VesselOptimizer(
            max_free_voyages=self.params.get("max_free_voyages", 6),
            vessel_penalty=self.params.get("vessel_penalty", 1000),
            slack_penalty=self.params.get("slack_penalty", 10),
            lead_time=self.params.get("lead_time", 5),
            vessel_capacity_1_2_grades=self.params.get("vessel_capacity_1_2_grades", 700),
            vessel_capacity_3_grades=self.params.get("vessel_capacity_3_grades", 650)
        )
        
        # Run optimization
        vessel_output_dir = os.path.join(self.output_dir, "vessel_optimization")
        results = optimizer.run(
            data=vessel_data,
            save_output=True,
            output_dir=vessel_output_dir,
            verbose=False,
            multiple_solutions=multiple_solutions,
            num_solutions=num_solutions,
            optimality_gap=optimality_gap
        )
        
        self.vessel_results = results
        
        # If single solution, update deliveries based on vessel schedule
        if not multiple_solutions and 'json_data' in results:
            self._update_deliveries_from_vessel_schedule(results['json_data'])
        
        return results
    
    def _update_deliveries_from_vessel_schedule(self, vessel_data: Dict[str, Any]) -> None:
        """
        Update deliveries based on vessel optimization results.
        
        Args:
            vessel_data: Vessel optimization JSON data
        """
        # Clear current deliveries
        self.deliveries.clear()
        self.future_deliveries.clear()
        
        # Create new deliveries from vessel parcels
        for parcel in vessel_data.get('parcels', []):
            voyage_id = parcel.get('voyage_id')
            
            # Find corresponding voyage to get arrival day
            for voyage in vessel_data.get('voyages', []):
                if voyage.get('voyage_id') == voyage_id:
                    arrival_day = voyage.get('approx_arrival')
                    
                    # Create delivery object
                    delivery = Delivery(
                        day=arrival_day,
                        crude=parcel.get('crude'),
                        volume=parcel.get('volume')
                    )
                    
                    # Add to deliveries
                    self.deliveries.append(delivery)
                    self.future_deliveries.setdefault(arrival_day, []).append(delivery)
                    break
    
    def run_daily_scheduling(self) -> List[Dict[str, Any]]:
        """
        Run the daily scheduling algorithm.
        
        Returns:
            List of daily schedule records
        """
        # Initialize parameters
        planning_horizon = self.params.get("planning_horizon", 40)
        processed_key = "Processed"
        
        # Create a copy of the initial inventory
        inventory = copy.deepcopy(self.inventory)
        
        # Initialize daily records
        records = []
        
        logger.info("===== Starting daily scheduling process =====")
        logger.info(f"Planning horizon: {planning_horizon} days")
        logger.info(f"Initial inventory: {inventory.to_dict()}")
        
        # For each day in the planning period
        for current_day in range(1, planning_horizon + 1):
            logger.info(f"==== Processing Day {current_day} ====")
            
            # Initialize today's record
            record = {
                "Day": current_day,
                "Strategy": None,
                "Blend_Index": None,
                processed_key: 0.0,
                "Crude_Usage": {},
                "Remaining": {}
            }
            
            # Check upcoming deliveries
            upcoming = self._get_deliveries_in_window(current_day, 7)
            if upcoming:
                logger.info(f"Day {current_day}: Upcoming deliveries in next 7 days: {len(upcoming)}")
                for delivery in upcoming:
                    logger.info(f"  - Day {delivery.day}: {delivery.crude} = {delivery.volume:.2f} kb")
            else:
                logger.info(f"Day {current_day}: No upcoming deliveries in next 7 days")
            
            # Log current inventory state
            logger.info(f"Day {current_day}: Current inventory levels:")
            for crude in inventory.get_all_crudes():
                logger.info(f"  - {crude}: {inventory.get(crude):.2f} kb")
            
            # Get the processing strategy for today
            logger.info(f"Day {current_day}: Selecting processing strategy...")
            strategy, processing_rate, blend_idx = choose_strategy(
                day=current_day,
                inventory=inventory,
                upcoming_deliveries=upcoming,
                blend_recipes=self.blends,
                params=self.params
            )
            
            # Try to follow the chosen strategy
            record["Strategy"] = strategy
            record["Blend_Index"] = blend_idx
            
            # If no feasible strategy, record zero processing and continue
            if strategy == "none" or blend_idx is None:
                logger.info(f"Day {current_day}: No feasible processing strategy found, skipping day")
                records.append(record)
                continue
                
            # Get the blend recipe
            blend = self.blends[blend_idx]
            logger.info(f"Day {current_day}: Selected strategy '{strategy}' with blend recipe {blend_idx} at rate {processing_rate:.2f} kb")
            logger.info(f"Day {current_day}: Blend recipe ratios: {blend.ratios}")
            
            # Record the planned processing rate
            record[processed_key] = processing_rate
            
            # Process each crude according to blend ratio
            try:
                # First check if we have enough inventory for each crude
                enough_inventory = True
                limited_crude = None
                available_fraction = 1.0
                
                # Check each crude required by the blend
                for crude, ratio in blend.ratios.items():
                    qty = processing_rate * ratio
                    logger.info(f"Day {current_day}: Checking availability of {crude}: Need {qty:.2f} kb (ratio {ratio:.2f})")
                    if not inventory.available(crude, qty):
                        enough_inventory = False
                        available = inventory.get(crude)
                        logger.warning(f"Day {current_day}: Not enough {crude}. Have {available:.2f} kb, need {qty:.2f} kb")
                        # Calculate what fraction of the desired rate we can actually process
                        if ratio > 0:
                            fraction = available / (processing_rate * ratio)
                            if fraction < available_fraction:
                                available_fraction = fraction
                                limited_crude = crude
                        
                # If we don't have enough inventory, adjust the processing rate
                if not enough_inventory:
                    original_rate = processing_rate
                    # Adjust processing rate to use what's available
                    processing_rate = processing_rate * available_fraction
                    logger.warning(f"Day {current_day}: Not enough {limited_crude}. Adjusting processing from {original_rate:.2f} to {processing_rate:.2f}")
                    # Update the record with adjusted rate
                    record[processed_key] = processing_rate
                    
                # Now process each crude with adjusted rate
                logger.info(f"Day {current_day}: Processing at rate {processing_rate:.2f} kb")
                for crude, ratio in blend.ratios.items():
                    qty = processing_rate * ratio
                    if qty > 0:
                        # Try to remove the crude, but handle potential errors
                        try:
                            if inventory.available(crude, qty):
                                logger.info(f"Day {current_day}: Using {qty:.2f} kb of {crude}")
                                inventory.remove(crude, qty)
                                # Record usage
                                record["Crude_Usage"][crude] = qty
                            else:
                                # Use what's available
                                available = inventory.get(crude)
                                if available > 0:
                                    inventory.remove(crude, available)
                                    record["Crude_Usage"][crude] = available
                                    logger.warning(f"Day {current_day}: Used remaining {available:.2f} of {crude}")
                                else:
                                    logger.warning(f"Day {current_day}: No {crude} available")
                        except ValueError as e:
                            # This shouldn't happen since we checked availability, but just in case
                            logger.error(f"Day {current_day}: Error processing {crude}: {str(e)}")
                            # Set processing to zero for this crude
                            record["Crude_Usage"][crude] = 0
                    else:
                        logger.info(f"Day {current_day}: {crude} not used (ratio: {ratio:.2f})")
                    
                # Record remaining inventory
                logger.info(f"Day {current_day}: Updated inventory levels after processing:")
                for crude in inventory.get_all_crudes():
                    remaining = inventory.get(crude)
                    record["Remaining"][crude] = remaining
                    logger.info(f"  - {crude}: {remaining:.2f} kb")
                    
            except Exception as e:
                logger.error(f"Error on day {current_day}: {str(e)}", exc_info=True)
                # Continue with the next day instead of failing
                record[processed_key] = 0.0
                record["Strategy"] = "error"
                record["Error"] = str(e)
            
            # Check for daily deliveries and continue processing active deliveries
            active_deliveries = getattr(self, 'active_deliveries', [])
            delayed_deliveries = getattr(self, 'delayed_deliveries', {})
            day_deliveries = self.future_deliveries.get(current_day, [])
            
            # Track tank objects for loading operations
            tanks_dict = {tank.id: tank for tank in self.tanks}
            
            # First check if we have capacity for today's deliveries
            total_tank_space_available = 0
            for tank in tanks_dict.values():
                if tank.available_space() > 0:
                    total_tank_space_available += tank.available_space()
                    
            # Calculate total volume to be delivered today
            day_delivery_volume = sum(delivery.volume for delivery in day_deliveries)
            
            # Handle active deliveries that are currently being unloaded
            if active_deliveries:
                logger.info(f"Day {current_day}: Continuing to process {len(active_deliveries)} active deliveries")
                still_active = []
                
                for delivery in active_deliveries:
                    # Try to unload more from this delivery
                    transferred = delivery.continue_loading(tanks_dict)
                    
                    if transferred > 0:
                        logger.info(f"Day {current_day}: Unloaded {transferred:.2f} kb of {delivery.crude} from ongoing delivery")
                        # Add to inventory after successful transfer
                        inventory.add(delivery.crude, transferred)
                    
                    # Check if delivery is still active
                    if delivery.is_loading:
                        still_active.append(delivery)
                        logger.info(f"Day {current_day}: Still unloading delivery of {delivery.crude}, "
                                   f"{delivery.unloaded_volume:.2f} kb remaining, {delivery.loading_days_left} days left")
                    else:
                        if delivery.unloaded_volume > 0:
                            logger.warning(f"Day {current_day}: Delivery of {delivery.crude} completed but {delivery.unloaded_volume:.2f} kb "
                                          f"could not be unloaded due to lack of tank space")
                        else:
                            logger.info(f"Day {current_day}: Completed unloading delivery of {delivery.crude}")
                
                # Update active deliveries
                active_deliveries = still_active
            
            # Check if any delayed deliveries should be processed today
            if current_day in delayed_deliveries:
                rescheduled_deliveries = delayed_deliveries.pop(current_day)
                logger.info(f"Day {current_day}: Processing {len(rescheduled_deliveries)} rescheduled deliveries that were delayed")
                day_deliveries.extend(rescheduled_deliveries)
            
            # Process new deliveries for today
            if day_deliveries:
                logger.info(f"Day {current_day}: Processing {len(day_deliveries)} deliveries:")
                
                # Calculate if we need to delay any deliveries
                total_delivery_volume = sum(delivery.volume for delivery in day_deliveries)
                if total_delivery_volume > total_tank_space_available * 1.5:  # Allow some flexibility with 2-day loading
                    logger.warning(f"Day {current_day}: Not enough tank space for all deliveries. "
                                  f"Need space for {total_delivery_volume:.2f} kb, only have {total_tank_space_available:.2f} kb available.")
                    
                    # Sort deliveries by volume (largest first) to prioritize larger deliveries
                    day_deliveries.sort(key=lambda d: d.volume, reverse=True)
                    
                    # Process deliveries until we hit capacity constraints
                    processed_deliveries = []
                    delayed_deliveries_today = []
                    
                    accumulated_volume = 0
                    for delivery in day_deliveries:
                        # If adding this delivery would exceed our capacity threshold, delay it
                        if accumulated_volume + delivery.volume > total_tank_space_available * 1.5:
                            # Delay this delivery by 3 days
                            delay_days = 3
                            original_day = delivery.day
                            delivery.delay(delay_days)
                            
                            # Add to delayed deliveries dictionary
                            if delivery.day not in delayed_deliveries:
                                delayed_deliveries[delivery.day] = []
                            delayed_deliveries[delivery.day].append(delivery)
                            
                            logger.warning(f"Day {current_day}: Delaying delivery of {delivery.volume:.2f} kb of {delivery.crude} "
                                         f"by {delay_days} days (from day {original_day} to day {delivery.day})")
                            
                            # Add to tracking list
                            delayed_deliveries_today.append(delivery)
                        else:
                            accumulated_volume += delivery.volume
                            processed_deliveries.append(delivery)
                    
                    # Update day_deliveries to only include ones we're processing today
                    day_deliveries = processed_deliveries
                    
                    # Add record of delayed deliveries to today's daily record
                    if delayed_deliveries_today:
                        record["Delayed_Deliveries"] = [
                            {
                                "crude": d.crude,
                                "volume": d.volume,
                                "original_day": d.original_day,
                                "delayed_to": d.day,
                                "delay_days": d.delay_days
                            }
                            for d in delayed_deliveries_today
                        ]
                
                # Process the deliveries for today
                for delivery in day_deliveries:
                    logger.info(f"Day {current_day}: Starting to receive {delivery.volume:.2f} kb of {delivery.crude}")
                    
                    # Start the loading process for this delivery
                    delivery.start_loading(loading_days=2)  # 2-day loading process
                    
                    # Try initial unloading (day 1 of 2)
                    transferred = delivery.continue_loading(tanks_dict)
                    
                    if transferred > 0:
                        logger.info(f"Day {current_day}: Unloaded {transferred:.2f} kb of {delivery.crude} on first day")
                        # Add the transferred amount to inventory
                        inventory.add(delivery.crude, transferred)
                    
                    # Check if loading is still in progress
                    if delivery.is_loading:
                        # Add to active deliveries for next day
                        active_deliveries.append(delivery)
                        logger.info(f"Day {current_day}: Will continue unloading {delivery.unloaded_volume:.2f} kb "
                                   f"of {delivery.crude} on day {current_day + 1}")
            
            # Store updated active deliveries list for next iteration
            self.active_deliveries = active_deliveries
            self.delayed_deliveries = delayed_deliveries
            
            # Log tank levels after deliveries
            tanks_dict = {tank.id: tank for tank in self.tanks}
            logger.info(f"Day {current_day}: Tank levels after deliveries:")
            for tank_id, tank in tanks_dict.items():
                crude_info = f"({tank.crude})" if tank.crude else "(empty)"
                logger.info(f"  - Tank {tank_id} {crude_info}: {tank.level:.2f}/{tank.capacity:.2f} kb")
            
            # Record remaining inventory and tank allocation in daily record
            record["Tank_Levels"] = {
                tank_id: {
                    "crude": tank.crude,
                    "level": tank.level,
                    "capacity": tank.capacity,
                    "available_space": tank.available_space(),
                    "composition": tank.composition
                }
                for tank_id, tank in tanks_dict.items()
            }
            
            # Record remaining inventory
            logger.info(f"Day {current_day}: Updated inventory levels after all operations:")
            for crude in inventory.get_all_crudes():
                remaining = inventory.get(crude)
                record["Remaining"][crude] = remaining
                logger.info(f"  - {crude}: {remaining:.2f} kb")
            
            # Save the record for today
            records.append(record)
            logger.info(f"Day {current_day}: Processing complete")
        
        logger.info("===== Completed daily scheduling process =====")
        logger.info(f"Total days scheduled: {len(records)}")
        logger.info(f"Total volume processed: {sum(r.get(processed_key, 0.0) for r in records):.2f} kb")
        
        # Store the records for later use
        self.schedule_records = records
        
        return records
    
    def _get_deliveries_in_window(self, start_day: int, window: int) -> List[Delivery]:
        """
        Get all deliveries within a certain window starting from start_day.
        
        Args:
            start_day: The day to start looking from
            window: The number of days to look ahead
            
        Returns:
            List of deliveries within the window
        """
        deliveries_in_window = []
        for day in range(start_day, min(start_day + window, 365)):  # Assuming 365 days in a year
            deliveries_in_window.extend(self.future_deliveries.get(day, []))
        return deliveries_in_window
    
    def run(self, 
        optimize_vessels: bool = True, 
        multiple_vessel_solutions: bool = False,
        output_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        Run the scheduling process.
        
        Args:
            optimize_vessels: Whether to run vessel optimization
            multiple_vessel_solutions: Whether to generate multiple vessel solutions
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with results
        """
        try:
            # Make sure we have the output directory set
            self.output_dir = output_dir
            
            # Ensure data is loaded
            if not self.tanks or self.inventory is None:
                logger.info("Loading data before running scheduler")
                self.load_data()
                
            # Step 1: Optimize vessel deliveries
            vessel_result = None
            if optimize_vessels:
                vessel_result = self._run_vessel_optimization(
                    multiple_solutions=multiple_vessel_solutions,
                    output_dir=output_dir
                )
            
            # Step 2: Run the main scheduling algorithm
            result = self._run_scheduling_algorithm()
            
            # Save results
            files = self.save_results(output_dir)
            
            # Return combined results
            combined_result = {
                "status": "success",
                "schedule": result,
                "files": files,
                "days_scheduled": len(result),
                "total_processed": sum(r.get("Processed", 0.0) for r in result),
                "optimizations": "none"
            }
            
            if vessel_result:
                combined_result["vessel_optimization"] = vessel_result
                
            return combined_result
            
        except Exception as e:
            logger.error(f"Error during scheduling: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

    
    def _run_vessel_optimization(self, multiple_solutions: bool = False, output_dir: str = "results") -> Dict[str, Any]:
        """
        Run vessel optimization as part of the scheduling process.
        
        Args:
            multiple_solutions: Whether to generate multiple solutions
            output_dir: Directory to save output files
            
        Returns:
            Vessel optimization results
        """
        # Set the output directory first (used by run_vessel_optimization internally)
        self.output_dir = output_dir
        
        # Call run_vessel_optimization without output_dir parameter
        return self.run_vessel_optimization(
            multiple_solutions=multiple_solutions
        )
    
    def _run_scheduling_algorithm(self) -> List[Dict[str, Any]]:
        """
        Run the main scheduling algorithm.
        
        Returns:
            Schedule records
        """
        # Run daily scheduling
        return self.run_daily_scheduling()
    
    def save_results(self, output_dir: str = "results") -> Dict[str, str]:
        """
        Save scheduling results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping file names to file paths
        """
        logger.info(f"Saving results to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        files = {}
        
        # Save schedule to JSON
        schedule_path = os.path.join(output_dir, "daily_schedule.json")
        with open(schedule_path, 'w') as f:
            json.dump(self.schedule_records, f, indent=2)
        files["Schedule JSON"] = schedule_path
        
        # Save schedule to CSV
        csv_path = os.path.join(output_dir, "daily_schedule.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.schedule_records[0].keys() if self.schedule_records else [])
            writer.writeheader()
            writer.writerows(self.schedule_records)
        files["Schedule CSV"] = csv_path
        
        # Save final inventory
        if hasattr(self, 'inventory') and self.inventory is not None:
            inventory_path = os.path.join(output_dir, "final_inventory.json")
            with open(inventory_path, 'w') as f:
                json.dump(self.inventory.to_dict(), f, indent=2)
            files["Final Inventory"] = inventory_path
        
        # Generate additional reports using reporter.py functions if available
        try:
            from src.reporter import save_reports
            additional_files = save_reports(
                self.schedule_records,  # Pass as positional argument instead of keyword
                output_dir=output_dir,
                inventory=self.inventory if hasattr(self, 'inventory') else None,
                blends=self.blends if hasattr(self, 'blends') else None,
            )
            files.update(additional_files)
        except ImportError:
            logger.warning("Reporter module not available, skipping additional reports")
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
        
        logger.info(f"Saved {len(files)} result files")
        return files

def schedule_plant_operation(
    config_path: str,
    output_dir: str = "results",
    optimize_vessels: bool = True,
    multiple_vessel_solutions: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for scheduling plant operation.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save output files
        optimize_vessels: Whether to run vessel optimization
        multiple_vessel_solutions: Whether to generate multiple vessel solutions
        
    Returns:
        Dictionary with results summary
    """
    try:
        logger.info(f"Starting scheduler with config: {config_path}")
        scheduler = Scheduler(config_path)
        
        result = scheduler.run(
            optimize_vessels=optimize_vessels,
            multiple_vessel_solutions=multiple_vessel_solutions,
            output_dir=output_dir
        )
        logger.info(f"Scheduling completed with status: {result['status']}")
        return result
    except ConfigValidationError as e:
        logger.error(f"Configuration validation error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


def optimize_existing_schedule(
    config_path: str,
    input_schedule_path: str = None,
    output_dir: str = "results/optimized"
) -> Dict[str, Any]:
    """
    Optimize an existing schedule.
    
    Args:
        config_path: Path to configuration file
        input_schedule_path: Path to the input schedule JSON file (if None, scheduler must have already run)
        output_dir: Directory to save optimized results
        
    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info(f"Starting schedule optimization with config: {config_path}")
        scheduler = Scheduler(config_path)
        
        # Load data before optimization
        scheduler.load_data()
        
        result = scheduler.optimize_schedule(
            input_schedule_path=input_schedule_path,
            output_dir=output_dir
        )
        
        logger.info(f"Schedule optimization completed with status: {result['status']}")
        if result['status'] == 'success':
            logger.info(f"Original total processed: {result['original_processed']:.2f} kb")
            logger.info(f"Optimized total processed: {result['optimized_processed']:.2f} kb")
            logger.info(f"Improvement: {result['improvement']:.2f} kb ({result['improvement_percentage']:.2f}%)")
            logger.info(f"Zero processing days reduced from {result['original_zero_days']} to {result['optimized_zero_days']}")
            
        return result
    except ConfigValidationError as e:
        logger.error(f"Configuration validation error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


if __name__ == "__main__":
    result = schedule_plant_operation("configs/scheduler_config.json")
    print(f"Scheduling completed with status: {result['status']}")
    if result['status'] == 'success':
        print(f"Total days scheduled: {result['days_scheduled']}")
        print(f"Total volume processed: {result['total_processed']:.2f} kb")
        print(f"Results saved to: {result['files']['daily_schedule_csv']}")
