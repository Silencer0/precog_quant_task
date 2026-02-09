
import traceback
import sys

# Redirect stdout/stderr
sys.stdout = open("test_output.txt", "w")
sys.stderr = sys.stdout

try:
    print("Importing...", flush=True)
    import pandas as pd
    import numpy as np
    from src.backtester.engine import run_backtest, BacktestConfig
    print("Imports successful.", flush=True)

    dates = pd.date_range('2023-01-01', periods=100)
    close = pd.DataFrame(np.random.randn(100, 2) + 100, index=dates, columns=['A', 'B'])
    weights = pd.DataFrame(0.5, index=dates, columns=['A', 'B']) # 50/50

    print("Running Event Driven...", flush=True)
    cfg = BacktestConfig(mode='event_driven', rebalance='D')
    res = run_backtest(close_prices=close, weights=weights, config=cfg)
    print(f"Event Driven End Equity: {res.equity.iloc[-1]:.2f}", flush=True)

    print("Running Vectorized...", flush=True)
    cfg_vec = BacktestConfig(mode='vectorized')
    res_vec = run_backtest(close_prices=close, weights=weights, config=cfg_vec)
    print(f"Vectorized End Equity: {res_vec.equity.iloc[-1]:.2f}", flush=True)

except Exception:
    traceback.print_exc()

sys.stdout.close()
