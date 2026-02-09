import sys
import os
import pandas as pd

# Allow running as a script: `python src/export_data.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.loader import load_asset_data, get_data_dir
from src.cleaning.imputation import impute_ffill


def export_cleaned_data():
    """Export the cleaned dataset with causal imputation only (no interpolation or lookahead)."""
    print("Exporting cleaned data...")
    data_dir = get_data_dir()
    data = load_asset_data(data_dir)

    output_dir = "dataset/cleaned"
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []
    for aid, df in data.items():
        # Use causal forward-fill to eliminate lookahead bias
        df_clean = impute_ffill(df)

        # Save CSV
        df_clean.to_csv(os.path.join(output_dir, f"{aid}.csv"))

        # Prepare for Parquet (add Asset ID column)
        df_parquet = df_clean.copy()
        df_parquet["Asset_ID"] = aid
        all_dfs.append(df_parquet)

    # Combine and save Parquet
    final_df = pd.concat(all_dfs)

    parquet_path = os.path.join(output_dir, "cleaned_stock_data.parquet")
    try:
        final_df.to_parquet(parquet_path)
    except ImportError:
        csv_path = os.path.join(output_dir, "cleaned_stock_data.csv")
        final_df.to_csv(csv_path, index=True)
        print(
            "Parquet engine missing (pyarrow/fastparquet). "
            f"Wrote combined CSV instead: {csv_path}"
        )
    print(f"Cleaned data saved to {output_dir}")


if __name__ == "__main__":
    export_cleaned_data()
