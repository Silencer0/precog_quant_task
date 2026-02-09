
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the engine to path
engine_path = "/home/anivarth/college/quant-task/another_testing_engine/trade-engine/trade-engine"
if engine_path not in sys.path:
    sys.path.append(engine_path)

from tradeengine.actors.memory import MemPortfolioActor
from tradeengine.actors.sql import SQLOrderbookActor
from tradeengine.backtest import BacktestStrategy
from tradeengine.dto import Asset
from sqlalchemy import create_engine, StaticPool
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_demo():
    print("Script started")
    # 1. Load some assets
    assets = ["Asset_001", "Asset_002", "Asset_003", "Asset_004", "Asset_005"]
    data_dir = "/home/anivarth/college/quant-task/dataset/cleaned"
    
    dfs = {}
    for a in assets:
        path = os.path.join(data_dir, f"{a}.csv")
        df = pd.read_csv(path, parse_dates=True, index_col="Date")
        dfs[a] = df.sort_index().head(1000)

    # 2. Calculate Technial Indicators & Signals
    # SMA Crossover strategy
    signals_raw = {}
    for a, df in dfs.items():
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        # Simple Entry/Exit
        df['Buy_Signal'] = (df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1))
        df['Sell_Signal'] = (df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1))
        signals_raw[a] = df

    # 3. Portfolio Management (Slots Logic)
    # We want max 20 positions, 1/20th allocation each.
    # Since we only have 5 assets here for the demo, they'll all fit, 
    # but we'll implement the logic properly.
    
    max_slots = 20
    active_positions = set()
    
    # We need to process signals time-synchronously
    all_dates = sorted(pd.concat([df.index.to_series() for df in dfs.values()]).unique())
    
    final_signals = {a: {} for a in assets}
    
    for t in all_dates:
        for a in assets:
            df = signals_raw[a]
            if t not in df.index:
                continue
            
            row = df.loc[t]
            
            # EXIT LOGIC
            if row['Sell_Signal'] and a in active_positions:
                active_positions.remove(a)
                final_signals[a][t] = {'CloseOrder': {}}
            
            # ENTRY LOGIC
            elif row['Buy_Signal'] and a not in active_positions:
                if len(active_positions) < max_slots:
                    active_positions.add(a)
                    # Allocation: 1/20th (5%)
                    final_signals[a][t] = {'TargetWeightOrder': {'size': 0.05}}

    # Convert final_signals to required format (Dict[Hashable, pd.Series])
    formatted_signals = {}
    for a, sig_dict in final_signals.items():
        if sig_dict:
            s = pd.Series(sig_dict)
            s.index = pd.to_datetime(s.index)
            formatted_signals[a] = s
        else:
            formatted_signals[a] = pd.Series(dtype=object)

    # 4. Run Backtest
    strategy_id = str(uuid.uuid4())
    portfolio_actor = MemPortfolioActor.start(funding=1000000.0) # 1 Million funding
    
    # We need a SQL Orderbook
    db_engine = create_engine('sqlite://', echo=False, connect_args={'check_same_thread': False}, poolclass=StaticPool)
    orderbook_actor = SQLOrderbookActor.start(
        portfolio_actor,
        db_engine,
        strategy_id=strategy_id
    )

    # Market data format for BacktestStrategy: Dict[Hashable, pd.DataFrame]
    quote_frames = {a: dfs[a][['Open', 'High', 'Low', 'Close']] for a in assets}
    
    print("Starting backtest...")
    bt_strategy = BacktestStrategy(orderbook_actor, portfolio_actor, quote_frames)
    backtest_result = bt_strategy.run_backtest(formatted_signals)
    
    print("Backtest Complete.")
    print("Portfolio Performance (last 5 rows):")
    print(backtest_result.porfolio_performance.tail())

if __name__ == "__main__":
    run_demo()
