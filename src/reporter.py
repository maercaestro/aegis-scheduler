import os
import pandas as pd
from typing import List, Dict, Any


def to_dataframe(schedule_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of schedule record dicts into a pandas DataFrame.
    Reorders columns for clarity: Day, Blend_Index, Strategy, Processed.
    """
    df = pd.DataFrame(schedule_records)
    # Ensure expected columns exist and order them
    cols = [col for col in ["Day", "Blend_Index", "Strategy", "Processed"] if col in df.columns]
    # Append any other columns at the end
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]
    return df


def save_reports(
    df: pd.DataFrame,
    output_csv: str = None,
    output_json: str = None,
    output_dir: str = "results",
    inventory = None,
    blends = None
) -> None:
    """
    Save the schedule DataFrame to CSV and JSON files.
    
    Args:
        df: DataFrame containing schedule data
        output_csv: CSV output filename (relative to output_dir)
        output_json: JSON output filename (relative to output_dir)
        output_dir: Directory to save output files
        inventory: Optional inventory object for additional reports
        blends: Optional blend recipes for additional reports
    """
    # Set default filenames if not provided
    if output_csv is None:
        output_csv = os.path.join(output_dir, "detailed_schedule.csv")
    elif not os.path.isabs(output_csv):
        output_csv = os.path.join(output_dir, output_csv)
        
    if output_json is None:
        output_json = os.path.join(output_dir, "detailed_schedule.json")
    elif not os.path.isabs(output_json):
        output_json = os.path.join(output_dir, output_json)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Save files
    df.to_csv(output_csv, index=False)
    df.to_json(output_json, orient='records', indent=2)
    
    return {
        "detailed_schedule_csv": output_csv,
        "detailed_schedule_json": output_json
    }
