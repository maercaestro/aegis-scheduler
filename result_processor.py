"""
Result processing module for optimization results.
Handles extraction, formatting, and saving of optimization results.
"""

import pandas as pd
import os
from typing import Dict, List, Any, Optional, Tuple
from pyomo.environ import value
import pickle


class ResultProcessor:
    """Processes and formats optimization results."""
    
    def __init__(self, model, data: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize result processor.
        
        Args:
            model: Solved Pyomo model
            data: Data dictionary used in optimization
            config: Configuration dictionary
        """
        self.model = model
        self.data = data
        self.config = config
        self.start_date = data['start_date']
        self.crudes = data['crudes']
        self.products_info = data['products_info']
    
    def extract_blending_results(self) -> pd.DataFrame:
        """
        Extract crude blending results from the solved model.
        
        Returns:
            DataFrame containing blending results
        """
        days = []
        final_products = []
        quantities_produced = []
        profit_each_slot = []
        slots = []
        inventory = []
        ullage = []
        
        # Initialize crude tracking dictionaries
        crude_blended = {c: [] for c in self.crudes}
        crude_available = {c: [] for c in self.crudes}
        
        for slot in self.model.SLOTS:
            slots.append(slot)
            
            # Calculate day from slot
            day = int((slot + 1) / 2) if (slot + 1) % 2 == 0 else int((slot + 2) / 2)
            days.append(day)
            
            total_profit = 0
            
            for blend in self.model.BLENDS:
                if value(self.model.IsBlendConsumed[blend, slot]) > 0.5:
                    final_products.append(blend)
                    produced = value(self.model.BlendFraction[blend, slot]) * value(self.model.BCb[blend])
                    quantities_produced.append(produced)
                    
                    inventory_total = 0
                    
                    for crude in self.crudes:
                        blended_amount = (value(self.model.BCb[blend]) * 
                                        value(self.model.BRcb[blend, crude]) * 
                                        value(self.model.BlendFraction[blend, slot]))
                        profit = self.model.MRc[crude] * blended_amount
                        crude_blended[crude].append(blended_amount)
                        
                        inv = value(self.model.Inventory[crude, day])
                        crude_available[crude].append(inv)
                        inventory_total += inv
                        total_profit += profit
                    
                    inventory.append(inventory_total)
                    break
            else:
                # No blend consumed in this slot
                final_products.append("None")
                quantities_produced.append(0.0)
                inventory_total = 0
                
                for crude in self.crudes:
                    crude_blended[crude].append(0.0)
                    inv = value(self.model.Inventory[crude, day])
                    crude_available[crude].append(inv)
                    inventory_total += inv
                
                inventory.append(inventory_total)
            
            ullage.append(value(self.model.Ullage[day]))
            profit_each_slot.append(total_profit)
        
        # Create records
        records = []
        for i in range(len(slots)):
            record = {
                "Date": pd.to_datetime(self.start_date) + pd.Timedelta(days=days[i] - 1),
                "Slot": slots[i],
                "Final Product": final_products[i],
                "Quantity Produced": round(quantities_produced[i] / 1000, 1),
                "Inventory Available": round(inventory[i] / 1000, 1),
                "Ullage": round(ullage[i] / 1000, 1),
                "Profit": profit_each_slot[i],
                "Flag": "Optimization"
            }
            
            # Add crude blended and available columns
            for c in self.crudes:
                record[f"Crude {c} Available"] = round(crude_available[c][i] / 1000, 1)
                record[f"Crude {c} Blended"] = round(crude_blended[c][i] / 1000, 1)
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert slots to 1 or 2
        df['Slot'] = df['Slot'].apply(lambda x: 2 if x % 2 == 0 else 1)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Reduce rows where one slot has zero production
        def reduce_rows(group):
            if (group["Quantity Produced"] == 0).sum() == 1:
                # Keep the non-zero row, force slot = 1
                row = group[group["Quantity Produced"] != 0].copy()
                row.loc[:, "Slot"] = 1
                return row
            else:
                return group
        
        combined_df_reduced = df.groupby(["Date", "Flag"], group_keys=False).apply(reduce_rows).reset_index(drop=True)
        
        return combined_df_reduced
    
    def extract_vessel_routing_results(self) -> pd.DataFrame:
        """
        Extract vessel routing results from the solved model.
        
        Returns:
            DataFrame containing vessel routing results
        """
        records = []
        
        # Create parcel size lookup
        parcel_size = {}
        for window, loc_data in self.data['crude_availability'].items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    key = (location, crude_type, window)
                    parcel_size[key] = info["parcel_size"]
        
        for v in self.model.VESSELS:
            is_vessel_started = False
            is_vessel_terminated = False
            is_at_melaka = 0
            last_port_location = None
            pending_sailing_records = []
            crude_loaded = {}
            
            for d in self.model.DAYS:
                at_location = False
                activity_name_list = []
                location_visited = None
                is_loading = 0
                is_unloading = 0
                
                for l in self.model.LOCATIONS:
                    if value(self.model.AtLocation[v, l, d]) > 0.5:
                        at_location = True
                        location_visited = l
                        last_port_location = l
                        
                        if not is_vessel_started:
                            activity_name_list.append("Arrival T")
                            is_vessel_started = True
                        
                        # Check for pickup activities
                        for p in self.model.PARCELS:
                            if value(self.model.Pickup[v, p, d]) > 0.5:
                                crude_type = p[1]
                                crude_volume_carried = parcel_size[p]
                                crude_loaded[f"{crude_type} Volume"] = crude_volume_carried
                                activity_name_list.append("Loading")
                                is_loading = 1
                                break
                        
                        # Check for Melaka arrival
                        if l == "Melaka" and is_at_melaka == 0:
                            activity_name_list.append("Arrival M")
                            is_at_melaka = 1
                        
                        # Check for discharge
                        if value(self.model.Discharge[v, d]) > 0.5:
                            activity_name_list.append("Discharge")
                            is_unloading = 1
                        
                        # Check for discharge continuation
                        if (d > 1) and value(self.model.Discharge[v, d-1]) > 0.5:
                            activity_name_list.append("Discharge")
                            is_vessel_terminated = True
                            is_unloading = 1
                        
                        # If no specific activity, it's demurrage
                        if 'Loading' not in activity_name_list and "Discharge" not in activity_name_list:
                            activity_name_list.append("Demurrage")
                
                # Handle sailing
                if is_vessel_started and not is_vessel_terminated and not at_location:
                    activity_name_list.append("Sailing")
                
                # Find next port when sailing
                next_port_location = None
                if not at_location:
                    for future_d in range(d + 1, max(self.model.DAYS) + 1):
                        for l_future in self.model.LOCATIONS:
                            if value(self.model.AtLocation[v, l_future, future_d]) > 0.5:
                                next_port_location = l_future
                                break
                        if next_port_location:
                            break
                
                # Determine last port display
                if at_location:
                    last_port_display = location_visited
                    # Update pending sailing records
                    for rec in pending_sailing_records:
                        rec["Last Port"] = f"{rec['Last Port'].split('--')[0]}--{location_visited}"
                        records.append(rec)
                    pending_sailing_records.clear()
                elif not at_location and last_port_location and next_port_location:
                    last_port_display = f"{last_port_location}--{next_port_location}"
                else:
                    last_port_display = "Unknown"
                
                # Create activity records
                for activity_name in activity_name_list:
                    demurrage_activity = 1 if activity_name == "Demurrage" else 0
                    
                    record = {
                        "Activity Date": pd.to_datetime(self.start_date) + pd.Timedelta(days=d - 1),
                        "Activity Name": activity_name,
                        "Activity End Date": pd.to_datetime(self.start_date) + pd.Timedelta(days=d),
                        "Vessel ID": v,
                        "Last Port": last_port_display,
                        "is_at_Melaka": is_at_melaka,
                        "is Demurrage Day": demurrage_activity,
                        "is_crude_unloading_day": is_unloading,
                        "is_loading": is_loading,
                        "Scenario Id": f"Scenario {self.config.get('scenario', 'Unknown')}"
                    }
                    
                    # Add crude loaded information
                    record.update(crude_loaded)
                    
                    if activity_name == "Sailing":
                        pending_sailing_records.append(record)
                    else:
                        records.append(record)
        
        return pd.DataFrame(records)
    
    def calculate_summary_metrics(self, blending_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary metrics from blending results.
        
        Args:
            blending_df: DataFrame containing blending results
            
        Returns:
            Dictionary containing summary metrics
        """
        total_throughput = blending_df['Quantity Produced'].sum()
        total_margin = blending_df['Profit'].sum()
        
        days_count = self.config["DAYS"]["end"] - self.config["DAYS"]["start"] + 1
        average_throughput = total_throughput / days_count
        average_margin = total_margin / days_count
        
        metrics = {
            "total_throughput": total_throughput,
            "total_margin": total_margin,
            "average_throughput": average_throughput,
            "average_margin": average_margin,
        }
        
        # Add demurrage if available
        try:
            metrics["total_demurrage_at_melaka"] = value(self.model.DemurrageAtMelaka)
            metrics["total_demurrage_at_source"] = value(self.model.DemurrageAtSource)
            metrics["total_demurrage"] = metrics["total_demurrage_at_melaka"] + metrics["total_demurrage_at_source"]
        except Exception as e:
            print(f"Warning: Could not calculate demurrage metrics: {e}")
        
        return metrics
    
    def save_results(self, output_dir: str, vessel_count: int, optimization_type: str, 
                    max_demurrage_limit: Optional[int] = None, 
                    max_transitions: Optional[int] = None) -> Dict[str, str]:
        """
        Save all results to files.
        
        Args:
            output_dir: Directory to save results
            vessel_count: Number of vessels used
            optimization_type: Type of optimization performed
            max_demurrage_limit: Maximum demurrage limit (for throughput optimization)
            max_transitions: Maximum transitions allowed
            
        Returns:
            Dictionary containing file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract results
        blending_df = self.extract_blending_results()
        vessel_df = self.extract_vessel_routing_results()
        
        # Generate filenames
        days_count = self.config["DAYS"]["end"]
        base_name = f'{optimization_type}_optimization_{vessel_count}_vessels_{days_count}_days'
        
        if max_transitions is not None:
            base_name += f'_{max_transitions}_transitions'
        
        if optimization_type == 'throughput' and max_demurrage_limit is not None:
            base_name += f'_{max_demurrage_limit}_demurrages'
        
        crude_blending_filename = f'crude_blending_{base_name}.csv'
        vessel_routing_filename = f'vessel_routing_{base_name}.csv'
        model_filename = f'{base_name}.pkl'
        metrics_filename = f'metrics_{base_name}.json'
        
        # Save files
        file_paths = {}
        
        # Save CSV files
        blending_path = os.path.join(output_dir, crude_blending_filename)
        vessel_path = os.path.join(output_dir, vessel_routing_filename)
        
        blending_df.to_csv(blending_path, index=False)
        vessel_df.to_csv(vessel_path, index=False)
        
        file_paths['crude_blending'] = blending_path
        file_paths['vessel_routing'] = vessel_path
        
        # Save model (pickle)
        model_path = os.path.join(output_dir, model_filename)
        try:
            with open(model_path, 'wb') as fp:
                pickle.dump(self.model, fp)
            file_paths['model'] = model_path
        except Exception as e:
            print(f"Warning: Could not save model to pickle: {e}")
        
        # Save metrics
        metrics = self.calculate_summary_metrics(blending_df)
        metrics_path = os.path.join(output_dir, metrics_filename)
        
        try:
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            file_paths['metrics'] = metrics_path
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
        
        return file_paths
    
    def print_summary(self, blending_df: Optional[pd.DataFrame] = None) -> None:
        """
        Print optimization summary to console.
        
        Args:
            blending_df: Blending DataFrame (will be extracted if not provided)
        """
        if blending_df is None:
            blending_df = self.extract_blending_results()
        
        metrics = self.calculate_summary_metrics(blending_df)
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        print(f"Total Throughput: {metrics['total_throughput']:.1f} (thousand units)")
        print(f"Total Margin: ${metrics['total_margin']:,.2f}")
        print(f"Average Daily Throughput: {metrics['average_throughput']:.1f} (thousand units)")
        print(f"Average Daily Margin: ${metrics['average_margin']:,.2f}")
        
        if 'total_demurrage' in metrics:
            print(f"Total Demurrage Cost: ${metrics['total_demurrage']:,.2f}")
            print(f"  - At Melaka: ${metrics['total_demurrage_at_melaka']:,.2f}")
            print(f"  - At Source: ${metrics['total_demurrage_at_source']:,.2f}")
        
        print("="*60)


def save_model_only(model, output_path: str) -> bool:
    """
    Save only the model to a pickle file.
    
    Args:
        model: Pyomo model to save
        output_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as fp:
            pickle.dump(model, fp)
        print(f"Model saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False