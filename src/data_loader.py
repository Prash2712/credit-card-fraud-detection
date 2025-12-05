import pandas as pd
from pathlib import Path

def load_data(path: str):
    """
    Load the credit card fraud dataset.

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(file_path)
    return df
