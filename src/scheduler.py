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
from src.optimizer import EndPeriodOptimizer

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
        Run the main scheduling loop for all days in planning horizon.
        
        Returns:
            List of daily schedule records
        """
        planning_horizon = self.params.get("planning_horizon", 40)
        lookahead = self.params.get("lookahead", 7)
        smoothing = self.params.get("smoothing", 0.85)
        critical = self.params.get("critical", 8)
        final_days = self.params.get("final_days", 10)
        absolute_min = self.params.get("absolute_min", 15.0)
        min_daily = self.params.get("min_daily", 50.0)
        increment = self.params.get("increment", 5.0)
        daily_capacity = self.params.get("daily_capacity", 100.0)
        
        self.schedule_records = []
        self.delayed_deliveries = {}  # Store delayed deliveries by day
        
        logger.info(f"Starting daily scheduling with planning horizon: {planning_horizon} days")
        logger.info(f"Initial inventory: {self.inventory.stock}")
        
        for day in range(1, planning_horizon + 1):
            logger.info(f"\n--- Day {day} ---")
            logger.info(f"Current inventory: {self.inventory.stock}")
            
            # Create record for this day
            record = {
                "Day": day,
                "Blend_Index": None,
                "Strategy": "none",
                "Processed": 0.0,
                "Tank_Levels": {},
                "Deliveries": [],
                "Delayed_Deliveries": [],
                "Crude_Usage": {}  # Track usage of each crude type
            }
            
            # Log and record tank levels
            logger.info("Tank levels:")
            for tank in self.tanks:
                logger.info(f"  Tank {tank.id}: {tank.level:.2f} kb of {tank.crude if tank.crude else 'Empty'}")
                if tank.composition:
                    for crude, vol in tank.composition.items():
                        logger.info(f"    - {crude}: {vol:.2f} kb")
                
                # Record tank level in daily record
                record["Tank_Levels"][tank.id] = {
                    "level": tank.level,
                    "crude": tank.crude,
                    "composition": tank.composition.copy() if tank.composition else {}
                }
            
            # 1. Check for delayed deliveries for today
            delayed_today = self.delayed_deliveries.get(day, [])
            if delayed_today:
                logger.info(f"Processing {len(delayed_today)} delayed deliveries for day {day}:")
                for d in delayed_today:
                    logger.info(f"  Delayed delivery: {d.volume:.2f} kb of {d.crude}")
                    # Try to allocate to tanks
                    alloc, rem = allocate_delivery(self.tanks, d)
                    logger.info(f"  Allocated {d.volume - rem:.2f} kb, Remainder: {rem:.2f} kb")
                    logger.info(f"  Allocation: {alloc}")
                    
                    # Record this delivery in the daily record
                    delivery_record = {
                        "crude": d.crude,
                        "volume": d.volume,
                        "allocated": d.volume - rem,
                        "remainder": rem,
                        "delayed": True,
                        "original_day": d.original_day if d.original_day is not None else d.day
                    }
                    record["Deliveries"].append(delivery_record)
                    
                    # If there's still remainder, delay it further
                    if rem > 0:
                        delay_delivery = Delivery(
                            day=day + 1,
                            crude=d.crude,
                            volume=rem,
                            original_day=d.original_day if d.original_day is not None else d.day
                        )
                        if day + 1 not in self.delayed_deliveries:
                            self.delayed_deliveries[day + 1] = []
                        self.delayed_deliveries[day + 1].append(delay_delivery)
                        
                        # Record the delay
                        delay_record = {
                            "crude": d.crude,
                            "volume": rem,
                            "original_day": d.original_day if d.original_day is not None else d.day,
                            "new_day": day + 1
                        }
                        record["Delayed_Deliveries"].append(delay_record)
                
                # Update inventory from tanks
                self.inventory.sync_with_tanks({t.id: t for t in self.tanks})
            
            # 2. Handle deliveries arriving today
            todays = self.future_deliveries.get(day, [])
            if todays:
                logger.info(f"Deliveries arriving on day {day}:")
                for d in todays:
                    logger.info(f"  {d.volume:.2f} kb of {d.crude}")
                    # Allocate to tanks
                    alloc, rem = allocate_delivery(self.tanks, d)
                    logger.info(f"  Allocated {d.volume - rem:.2f} kb, Remainder: {rem:.2f} kb")
                    logger.info(f"  Allocation: {alloc}")
                    
                    # Record this delivery in the daily record
                    delivery_record = {
                        "crude": d.crude,
                        "volume": d.volume,
                        "allocated": d.volume - rem,
                        "remainder": rem,
                        "delayed": False
                    }
                    record["Deliveries"].append(delivery_record)
                    
                    # If there's remainder, schedule for tomorrow
                    if rem > 0:
                        delay_delivery = Delivery(
                            day=day + 1,
                            crude=d.crude,
                            volume=rem,
                            original_day=d.day
                        )
                        if day + 1 not in self.delayed_deliveries:
                            self.delayed_deliveries[day + 1] = []
                        self.delayed_deliveries[day + 1].append(delay_delivery)
                        
                        # Record the delay
                        delay_record = {
                            "crude": d.crude,
                            "volume": rem,
                            "original_day": d.day,
                            "new_day": day + 1
                        }
                        record["Delayed_Deliveries"].append(delay_record)
                    
                    # Update inventory from tanks
                    self.inventory.sync_with_tanks({t.id: t for t in self.tanks})
                logger.info(f"Updated inventory after deliveries: {self.inventory.stock}")
    
            # Rest of the scheduling logic
            window = min(lookahead, planning_horizon - day + 1)
            subset = {d: self.future_deliveries.get(d, []) for d in range(day, day + window)}
            logger.info(f"Lookahead window: {window} days")
            
            # Log future deliveries in window
            future_del_count = sum(len(deliv) for deliv in subset.values())
            logger.info(f"Future deliveries in window: {future_del_count}")
            for d, delivs in subset.items():
                for deliv in delivs:
                    logger.info(f"  Day {d}: {deliv.volume:.2f} kb of {deliv.crude}")
    
            # 3. Evaluate blends
            candidates = []  # (blend_idx, blend, rate, total, strat)
            # Project inventory for rationing
            inv_proj = {d: self.inventory.stock.copy() for d in range(day, planning_horizon + 1)}  # simple proj
            logger.info(f"Evaluating {len(self.blends)} blend recipes")
    
            for idx, blend in enumerate(self.blends):
                logger.info(f"  Blend {idx}: {blend.ratios}")
                max_per_day = simulate_blend(blend, self.inventory, self.tanks, subset, window, increment)
                logger.info(f"  Max per day: {[f'{m:.2f}' for m in max_per_day]}")
                rate, total, strat = choose_strategy(max_per_day, smoothing)
                logger.info(f"  Strategy: {strat}, Rate: {rate:.2f}, Total: {total:.2f}")
                
                # Apply rationing
                f = calculate_rationing_factor(blend.ratios, day, inv_proj, critical)
                logger.info(f"  Rationing factor: {f:.4f}")
                rate *= f
                total *= f
                
                # Final days enforcement
                original_rate = rate
                rate = enforce_final_days(rate, planning_horizon - day + 1, final_days, absolute_min)
                if rate != original_rate:
                    logger.info(f"  Final days enforcement adjusted rate: {rate:.2f}")
                
                candidates.append((idx, blend, rate, total, strat))
    
            # 4. Select best candidate
            # Prefer highest total, tie-breaker on rate
            if not candidates:
                logger.info("No viable blend candidates found")
                best = (None, None, 0.0, 0.0, "none")
            else:
                best = max(candidates, key=lambda x: x[3])
                logger.info(f"Selected best candidate: Blend {best[0]}, Rate: {best[2]:.2f}, Total: {best[3]:.2f}")
            
            idx, blend, rate, total, strat = best
    
            # 5. Execute processing and track crude usage
            processed = rate
            if processed > 1e-6:
                logger.info(f"Processing {processed:.2f} kb")
                # Withdraw from inventory & tanks
                for crude, ratio in blend.ratios.items():
                    qty = processed * ratio
                    logger.info(f"  Need to withdraw {qty:.4f} kb of {crude} (available: {self.inventory.get(crude):.4f} kb)")
                    self.inventory.remove(crude, qty)
                    logger.info(f"  Inventory of {crude} after removal: {self.inventory.get(crude):.4f} kb")
                    
                    # Track crude usage in the daily record
                    record["Crude_Usage"][crude] = qty
                    
                    remaining = qty
                    logger.info(f"  Withdrawing {qty:.4f} kb of {crude} from tanks")
                    for tank in self.tanks:
                        avail = tank.composition.get(crude, 0.0)
                        take = min(avail, remaining)
                        if take > 1e-6:
                            logger.info(f"    From tank {tank.id}: {take:.4f} kb (had {avail:.4f} kb)")
                            try:
                                tank.withdraw(crude, take)
                                remaining -= take
                                if remaining <= 1e-6:
                                    break
                            except Exception as e:
                                logger.error(f"    Error withdrawing from tank {tank.id}: {str(e)}")
                                logger.error(f"    Tank state: level={tank.level}, crude={tank.crude}, composition={tank.composition}")
                                raise
                    
                    if remaining > 1e-6:
                        logger.warning(f"  Could not withdraw {remaining:.4f} kb of {crude} from tanks")
            else:
                logger.info("No processing today (rate <= 0)")
            
            # 6. Record day
            record["Blend_Index"] = idx
            record["Strategy"] = strat
            record["Processed"] = processed
            
            self.schedule_records.append(record)
            logger.info(f"Day {day} complete. Processed: {processed:.2f} kb")
    
            # Early exit
            if all(t.is_empty() for t in self.tanks):
                logger.info("All tanks empty - ending scheduling early")
                break
                
        logger.info("Daily scheduling complete")
        return self.schedule_records
    
    def save_results(self, output_dir: str = None) -> Dict[str, str]:
        """
        Save scheduling results to output files.
        
        Args:
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with file paths
        """
        if output_dir:
            self.output_dir = output_dir
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert records to DataFrame with simplified structure for daily_schedule.csv
        simplified_records = []
        for record in self.schedule_records:
            simplified = {
                "Day": record["Day"],
                "Blend_Index": record["Blend_Index"],
                "Strategy": record["Strategy"],
                "Processed": record["Processed"]
            }
            simplified_records.append(simplified)
        
        df = to_dataframe(simplified_records)
        
        # Save simplified CSV and JSON for backward compatibility
        csv_path = os.path.join(self.output_dir, "daily_schedule.csv")
        json_path = os.path.join(self.output_dir, "daily_schedule.json")
        save_reports(df, csv_path, json_path)
        
        # Save detailed schedule with tank levels and deliveries
        detailed_json_path = os.path.join(self.output_dir, "detailed_schedule.json")
        with open(detailed_json_path, 'w') as f:
            # Process records to make them JSON serializable
            serializable_records = []
            for record in self.schedule_records:
                serializable_record = copy.deepcopy(record)
                # Remove any non-serializable objects or convert them
                serializable_records.append(serializable_record)
            
            json.dump(serializable_records, f, indent=2)
        
        # Save detailed schedule as CSV
        detailed_csv_path = os.path.join(self.output_dir, "detailed_schedule.csv")
        self._save_detailed_schedule_csv(detailed_csv_path)
        
        # Return file paths
        file_paths = {
            "daily_schedule_csv": csv_path,
            "daily_schedule_json": json_path,
            "detailed_schedule_json": detailed_json_path,
            "detailed_schedule_csv": detailed_csv_path
        }
        
        return file_paths
    
    def _save_detailed_schedule_csv(self, csv_path: str) -> None:
        """
        Save the detailed schedule as a CSV file.
        
        Args:
            csv_path: Path to save the CSV file
        """
        # Collect all possible crude types and tank IDs
        all_crude_types = set()
        all_tank_ids = set()
        
        for record in self.schedule_records:
            all_tank_ids.update(record["Tank_Levels"].keys())
            all_crude_types.update(record["Crude_Usage"].keys())
            
            # Also check for crude types in tank compositions
            for tank_data in record["Tank_Levels"].values():
                if "composition" in tank_data:
                    all_crude_types.update(tank_data["composition"].keys())
        
        # Sort for consistent columns
        all_crude_types = sorted(all_crude_types)
        all_tank_ids = sorted(all_tank_ids)
        
        # Create CSV header
        header = ["Day", "Blend_Index", "Strategy", "Processed"]
        
        # Add columns for tank levels
        for tank_id in all_tank_ids:
            header.append(f"Tank_{tank_id}_Level")
            header.append(f"Tank_{tank_id}_Crude")
            
        # Add columns for crude usage
        for crude in all_crude_types:
            header.append(f"Used_{crude}")
            
        # Add columns for deliveries
        header.extend(["Delivery_Count", "Delayed_Count"])
            
        # Write CSV file
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(header)
            
            # Write data rows
            for record in self.schedule_records:
                row = [
                    record["Day"],
                    record["Blend_Index"] if record["Blend_Index"] is not None else "",
                    record["Strategy"],
                    record["Processed"]
                ]
                
                # Add tank levels
                for tank_id in all_tank_ids:
                    if tank_id in record["Tank_Levels"]:
                        tank_data = record["Tank_Levels"][tank_id]
                        row.append(tank_data["level"])
                        row.append(tank_data["crude"] if tank_data["crude"] else "")
                    else:
                        row.append("")
                        row.append("")
                
                # Add crude usage
                for crude in all_crude_types:
                    row.append(record["Crude_Usage"].get(crude, ""))
                
                # Add delivery counts
                row.append(len(record["Deliveries"]))
                row.append(len(record["Delayed_Deliveries"]))
                
                writer.writerow(row)
                
        logger.info(f"Detailed schedule saved to {csv_path}")
    
    def run(self, 
        optimize_vessels: bool = True, 
        multiple_vessel_solutions: bool = False,
        optimize_end_period: bool = False,  # Changed default to False
        output_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        Run the scheduling process.
        
        Args:
            optimize_vessels: Whether to run vessel optimization
            multiple_vessel_solutions: Whether to generate multiple vessel solutions
            optimize_end_period: Whether to optimize end-period utilization
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
            
            # Step 3: End-period optimization if enabled
            if optimize_end_period:
                result = self._optimize_end_period(result)
            
            # Save results
            files = self.save_results(output_dir)
            
            # Return combined results
            combined_result = {
                "status": "success",
                "schedule": result,
                "files": files,
                "days_scheduled": len(result),
                "total_processed": sum(r.get("Processed", 0.0) for r in result),
                "optimizations": "end-period" if optimize_end_period else "none"
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

    def optimize_schedule(self, input_schedule_path: str = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Optimize an existing schedule.
        
        Args:
            input_schedule_path: Path to the input schedule JSON file (if None, uses the current schedule_records)
            output_dir: Directory to save optimized results
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Set output directory
            if output_dir:
                self.output_dir = output_dir
            
            # Load the schedule if provided
            if input_schedule_path:
                logger.info(f"Loading schedule from {input_schedule_path}")
                with open(input_schedule_path, 'r') as f:
                    self.schedule_records = json.load(f)
            
            # Ensure we have a schedule to optimize
            if not self.schedule_records:
                message = "No schedule records available for optimization"
                logger.error(message)
                return {
                    "status": "error",
                    "message": message
                }
                
            # Load data if needed
            if not self.tanks or self.inventory is None:
                logger.info("Loading data before optimizing")
                self.load_data()
                
            # Define optimizer parameters
            optimizer_params = self.params.get("optimizer_params", {})
            
            # Add crude name mappings
            optimizer_params["crude_mappings"] = {
                "E2": "E",
                "F2": "F"
            }
            
            # Create optimizer
            optimizer = EndPeriodOptimizer(params=optimizer_params)
            
            # Run optimization
            logger.info("Running schedule optimization")
            original_records = copy.deepcopy(self.schedule_records)
            self.schedule_records = optimizer.optimize_schedule(
                schedule=self.schedule_records,
                blends=self.blends,
                initial_inventory=self.inventory,
                tanks=self.tanks,
                planning_horizon=self.params.get("planning_horizon", 40)
            )
            
            # Calculate improvement metrics
            original_processed = sum(r.get("Processed", 0.0) for r in original_records)
            optimized_processed = sum(r.get("Processed", 0.0) for r in self.schedule_records)
            
            # Count zero processing days before and after
            original_zero_days = sum(1 for r in original_records if r.get("Processed", 0.0) <= 0.001)
            optimized_zero_days = sum(1 for r in self.schedule_records if r.get("Processed", 0.0) <= 0.001)
            
            # Save optimized results
            files = self.save_results(self.output_dir)
            
            # Return results
            return {
                "status": "success",
                "original_processed": original_processed,
                "optimized_processed": optimized_processed,
                "improvement": optimized_processed - original_processed,
                "improvement_percentage": ((optimized_processed / original_processed) - 1) * 100 if original_processed > 0 else 0,
                "original_zero_days": original_zero_days,
                "optimized_zero_days": optimized_zero_days,
                "files": files,
                "days_scheduled": len(self.schedule_records),
                "message": "Schedule optimization completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error during schedule optimization: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

    def _optimize_end_period(self, schedule_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize end-period scheduling to ensure consistent high throughput.
        
        Args:
            schedule_records: Daily schedule records
            
        Returns:
            Optimized schedule records
        """
        if not schedule_records:
            return schedule_records
            
        # Define optimizer parameters
        optimizer_params = self.params.get("optimizer_params", {})
        
        # Add crude name mappings
        optimizer_params["crude_mappings"] = {
            "E2": "E",
            "F2": "F"
        }
        
        # Create optimizer
        optimizer = EndPeriodOptimizer(params=optimizer_params)
        
        # Run optimization
        optimized_records = optimizer.optimize_schedule(
            schedule=schedule_records,
            blends=self.blends,
            initial_inventory=self.inventory,
            tanks=self.tanks,
            planning_horizon=self.params.get("planning_horizon", 40)
        )
        
        return optimized_records
    
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

def schedule_plant_operation(
    config_path: str,
    output_dir: str = "results",
    optimize_vessels: bool = True,
    multiple_vessel_solutions: bool = False,
    optimize_end_period: bool = False  # Changed default to False
) -> Dict[str, Any]:
    """
    Main entry point for scheduling plant operation.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save output files
        optimize_vessels: Whether to run vessel optimization
        multiple_vessel_solutions: Whether to generate multiple vessel solutions
        optimize_end_period: Whether to optimize end-period utilization
        
    Returns:
        Dictionary with results summary
    """
    try:
        logger.info(f"Starting scheduler with config: {config_path}")
        scheduler = Scheduler(config_path)
        
        result = scheduler.run(
            optimize_vessels=optimize_vessels,
            multiple_vessel_solutions=multiple_vessel_solutions,
            optimize_end_period=optimize_end_period,
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
