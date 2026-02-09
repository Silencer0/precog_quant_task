import pandas as pd
import os
from glob import glob


def load_asset_data(data_dir, asset_id=None):
    """
    Loads stock data for a specific asset or all assets.
    Returns a dictionary of DataFrames.
    """
    if asset_id:
        file_path = os.path.join(data_dir, f"{asset_id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            return {asset_id: df}
        else:
            raise FileNotFoundError(f"Asset file {file_path} not found.")

    files = glob(os.path.join(data_dir, "Asset_*.csv"))
    data = {}
    for f in sorted(files):
        aid = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        data[aid] = df
    return data


def get_data_dir():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../dataset/archive/anonymized_data/"
        )
    )
