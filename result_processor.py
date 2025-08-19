"""
Result processor for optimization results.
Handles extraction and formatting of optimization results into DataFrames.
"""

import pandas as pd
from pyomo.environ import value
from tqdm import tqdm


class ResultProcessor:
    """Processes optimization results and creates output DataFrames."""
    
    def __init__(self, model, config: dict, crudes: list, start_date: pd.Timestamp):
        """
        Initialize result processor.
        
        Args:
            model: Solved Pyomo model
            config: Configuration dictionary
            crudes: List of crude types
            start_date: Start date for the schedule
        """
        self.model = model
        self.config = config
        self.crudes = crudes
        self.start_date = start_date
        
    def extract_crude_blending_results(self) -> pd.DataFrame:
        """Extract crude blending optimization results."""
        model = self.model
        
        days = []
        Final_Product = []
        Quantity_produced = []
        profit_each_slot = []
        slots = []
        inventory = []
        ullage = []
        crude_blended = {c: [] for c in self.crudes}
        crude_available = {c: [] for c in self.crudes}

        print("📊 Extracting crude blending results...")
        
        # Process each slot with progress bar
        for slot in tqdm(model.SLOTS, desc="Processing slots", unit="slot"):
            slots.append(slot)

            if (slot + 1) % 2 == 0:
                day = int((slot + 1) / 2)
            days.append(day)
            total_profit = 0

            for blend in model.BLENDS:
                if value(model.IsBlendConsumed[blend, slot]) > 0.5:
                    Final_Product.append(blend)
                    produced = value(model.BlendFraction[blend, slot]) * value(model.BCb[blend])
                    Quantity_produced.append(produced)
                    inventory_total = 0
                   
                    for crude in model.CRUDES:
                        blended_amount = value(model.BCb[blend]) * value(model.BRcb[blend, crude]) * value(model.BlendFraction[blend, slot])
                        profit = model.MRc[crude] * blended_amount
                        crude_blended[crude].append(blended_amount)
                        inv = value(model.Inventory[crude, day])
                        crude_available[crude].append(inv)
                        inventory_total += inv
                        total_profit += profit
                        
                    inventory.append(inventory_total)
            ullage.append(value(model.Ullage[day]))
            profit_each_slot.append(total_profit)   

        records = []
        for i in range(len(slots)):
            record = {
                "Date": pd.to_datetime(self.start_date) + pd.Timedelta(days=days[i] - 1), 
                "Slot": slots[i],
                "Final Product": Final_Product[i],
                "Quantity Produced": round(Quantity_produced[i] / 1000, 1),
                **{f"Crude {c} Available": round(crude_available[c][i] / 1000, 1) for c in self.crudes},
                **{f"Crude {c} Blended": round(crude_blended[c][i] / 1000, 1) for c in self.crudes},
                "Inventory Available": round(inventory[i] / 1000, 1),
                "Ullage": round(ullage[i] / 1000, 1),
                "Profit": profit_each_slot[i],
                "Flag": "Optimization"
            }
            records.append(record)
        
        df = pd.DataFrame(records)

        # Adjust slot numbering
        slot = []
        for i in df['Slot']:
            if i % 2 == 0:
                slot.append(2)
            else:
                slot.append(1)
        df['Slot'] = slot
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
        
    def extract_vessel_routing_results(self, crude_availability: dict) -> pd.DataFrame:
        """Extract vessel routing optimization results."""
        model = self.model
        
        print("🚢 Extracting vessel routing results...")
        
        # Create parcel size lookup
        parcel_size = {}
        for window, loc_data in crude_availability.items():
            for location, crude_dict in loc_data.items():
                for crude_type, info in crude_dict.items():
                    key = (location, crude_type, window)
                    parcel_size[key] = info["parcel_size"]
        
        records = []

        # Process each vessel with progress bar
        for v in tqdm(model.VESSELS, desc="Processing vessels", unit="vessel"):
            is_vessel_started = False
            is_vessel_terminated = False
            is_at_melaka = 0
            last_port_location = None
            pending_sailing_records = []
            crude_loaded = {}

            for d in model.DAYS:
                at_location = False
                activity_name_list = []
                location_visited = None
                is_loading = 0
                is_unloading = 0

                for l in model.LOCATIONS:
                    if value(model.AtLocation[v, l, d]) > 0.5:
                        at_location = True
                        location_visited = l

                        # Update last_port_location when vessel is at port
                        last_port_location = l

                        if not is_vessel_started:
                            activity_name_list.append("Arrival T")
                            is_vessel_started = True

                        for p in model.PARCELS:
                            if value(model.Pickup[v, p, d]) > 0.5:
                                crude_type = p[1]
                                crude_volume_carried = parcel_size[p]
                                crude_loaded[f"{crude_type} Volume"] = crude_volume_carried
                                activity_name_list.append("Loading")
                                is_loading = 1
                                break

                        if l == "Melaka" and is_at_melaka == 0:
                            activity_name_list.append("Arrival M")
                            is_at_melaka = 1

                        if value(model.Discharge[v, d]) > 0.5:
                            activity_name_list.append("Discharge")
                            is_unloading = 1
                    
                        if (d > 1) and value(model.Discharge[v, d - 1]) > 0.5:
                            activity_name_list.append("Discharge")
                            is_vessel_terminated = True
                            is_unloading = 1

                        if 'Loading' not in activity_name_list and "Discharge" not in activity_name_list:
                            activity_name_list.append("Demurrage")

                if is_vessel_started and not is_vessel_terminated and not at_location:
                    activity_name_list.append("Sailing")

                # Find next port when sailing
                next_port_location = None
                if not at_location:
                    # Look ahead to find first future location
                    for future_d in range(d + 1, max(model.DAYS) + 1):
                        for l_future in model.LOCATIONS:
                            if value(model.AtLocation[v, l_future, future_d]) > 0.5:
                                next_port_location = l_future
                                break
                        if next_port_location:
                            break

                # Decide Last Port display
                if at_location:
                    last_port_display = location_visited
                    # Update any pending sailing records now that we know next port
                    for rec in pending_sailing_records:
                        rec["Last Port"] = f"{rec['Last Port'].split('--')[0]}--{location_visited}"
                        records.append(rec)
                    pending_sailing_records.clear()
                elif not at_location and last_port_location and next_port_location:
                    last_port_display = f"{last_port_location}--{next_port_location}"
                else:
                    last_port_display = "Unknown"

                for activity_name in activity_name_list:
                    if activity_name == "Demurrage":
                        demurrage_activity = 1
                    else:
                        demurrage_activity = 0
                    record = {
                        "Activity Date": pd.to_datetime(self.start_date) + pd.Timedelta(days=d - 1),
                        "Activity Name": activity_name,
                        "Activity End Date": pd.to_datetime(self.start_date) + pd.Timedelta(days=d),
                        "Vessel ID": v,
                        "Last Port": last_port_display,
                        **crude_loaded,
                        "is_at_Melaka": is_at_melaka,
                        "is Demurrage Day": demurrage_activity,
                        "is_crude_unloading_day": is_unloading,
                        "is_loading": is_loading,
                        "Scenario Id": f"Test Scenario"
                    }

                    if activity_name == "Sailing":
                        # Store temporarily to update once we know next port
                        pending_sailing_records.append(record)
                    else:
                        records.append(record)
                        
        vessel_df = pd.DataFrame(records)
        return vessel_df
        
    def calculate_summary_metrics(self, crude_blending_df: pd.DataFrame) -> dict:
        """Calculate summary metrics from optimization results."""
        total_throughput = crude_blending_df['Quantity Produced'].sum()
        total_margin = crude_blending_df['Profit'].sum()
        average_throughput = total_throughput / self.config["DAYS"]["end"]
        average_margin = total_margin / self.config["DAYS"]["end"]
        
        demurrage_at_melaka = value(self.model.DemurrageAtMelaka)
        demurrage_at_source = value(self.model.DeumrrageAtSource)
        
        return {
            "total_throughput": total_throughput,
            "total_margin": total_margin,
            "average_throughput": average_throughput,
            "average_margin": average_margin,
            "demurrage_at_melaka": demurrage_at_melaka,
            "demurrage_at_source": demurrage_at_source,
            "total_demurrage": demurrage_at_melaka + demurrage_at_source
        }
