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
    output_csv: str,
    output_json: str
) -> None:
    """
    Save the schedule DataFrame to CSV and JSON files.
    """
    df.to_csv(output_csv, index=False)
    df.to_json(output_json, orient='records', indent=2)
