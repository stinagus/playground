from datetime import datetime
from utils import *
import pandas as pd
import yfinance as yf

if __name__ == "__main__":

    pattern_types = ['engulfing', 'rejection']
    ticker_symbol = "^GSPC" # SP500
    start_date = datetime(2022, 1, 1)
    end_date = datetime.now()  # Adjust the end date as per the current date
    daily_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
    daily_data.index = pd.to_datetime(daily_data.index)
    print("Daily Data:")
    print(daily_data.head())

    df = clean_df(daily_data)

    for pattern_type in pattern_types:
        df = detect_candlestick_patterns(df, pattern_type=pattern_type)
        df = add_pointpos_column(df)

        # Simulate trading strategy
        trade_log, final_balance = simulate_trading_strategy(df)
        print("Trade Log:")
        for trade in trade_log:
            print(trade)
        print("Final Balance:", final_balance)

        # Plot candlestick chart with signals and balance
        plot_candlestick_with_signals_and_balance(df, start_index=0, num_rows=len(df), trade_log=trade_log, title=pattern_type)
