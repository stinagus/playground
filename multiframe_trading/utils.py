import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Clean DataFrame function
def clean_df(df):
    df = df.copy()
    df.dropna(inplace=True)
    df = df.astype(float)
    return df


def is_bullish_engulfing(open_price, close_price, prev_open, prev_close):
    return close_price > open_price and open_price < prev_close and close_price > prev_open and prev_close > prev_open


def is_bearish_engulfing(open_price, close_price, prev_open, prev_close):
    return close_price < open_price and open_price > prev_close and close_price < prev_open and prev_close < prev_open


def is_bullish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit=0.5e-4):
    body = abs(close_price - open_price)
    lower_wick = min(open_price, close_price) - low_price
    upper_wick = high_price - max(close_price, open_price)
    currentbody = abs(open_price - close_price)
    c1 = currentbody > body_perc_limit * close_price
    return c1 and lower_wick > body * tail_body_ratio and upper_wick < body / tail_body_ratio


def is_bearish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit=0.5e-4):
    body = abs(close_price - open_price)
    upper_wick = high_price - max(close_price, open_price)
    lower_wick = min(open_price, close_price) - low_price
    currentbody = abs(open_price - close_price)
    c1 = currentbody > body_perc_limit * close_price
    return c1 and upper_wick > body * tail_body_ratio and lower_wick < body / tail_body_ratio


# General signal detection function
def detect_candlestick_patterns(df, pattern_type, tail_body_ratio=2, body_perc_limit=0.5e-4):
    df['candlesignal'] = 0
    for i in range(1, len(df)):
        open_price = df.iloc[i]['Open']
        close_price = df.iloc[i]['Close']
        high_price = df.iloc[i]['High']
        low_price = df.iloc[i]['Low']
        prev_open = df.iloc[i - 1]['Open']
        prev_close = df.iloc[i - 1]['Close']

        if pattern_type == 'engulfing':
            if is_bullish_engulfing(open_price, close_price, prev_open, prev_close):
                df.at[df.index[i], 'candlesignal'] = 2  # Buy signal
            elif is_bearish_engulfing(open_price, close_price, prev_open, prev_close):
                df.at[df.index[i], 'candlesignal'] = 1  # Sell signal
        elif pattern_type == 'rejection':
            if is_bullish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit):
                df.at[df.index[i], 'candlesignal'] = 2  # Buy signal
            elif is_bearish_rejection(open_price, close_price, high_price, low_price, tail_body_ratio, body_perc_limit):
                df.at[df.index[i], 'candlesignal'] = 1  # Sell signal

    return df


# Add point position column for plotting
def add_pointpos_column(df):
    def pointpos(row):
        if row['candlesignal'] == 2:
            return row['Low'] - 1e-4
        elif row['candlesignal'] == 1:
            return row['High'] + 1e-4
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    return df


# Simulate trading strategy
def simulate_trading_strategy(df, initial_balance=10000, stop_loss=0.02, take_profit=0.04):
    balance = initial_balance
    position = None
    entry_price = 0
    trade_log = []

    for i in range(len(df)):
        if position is None:
            if df.iloc[i]['candlesignal'] == 2:  # Bullish signal (Buy)
                position = 'long'
                entry_price = df.iloc[i]['Close']
                trade_log.append((i, 'buy', entry_price, balance))
            elif df.iloc[i]['candlesignal'] == 1:  # Bearish signal (Sell)
                position = 'short'
                entry_price = df.iloc[i]['Close']
                trade_log.append((i, 'sell', entry_price, balance))
        elif position == 'long':
            if df.iloc[i]['Low'] < entry_price * (1 - stop_loss):
                balance -= (entry_price - entry_price * (1 - stop_loss)) * (initial_balance / entry_price)
                trade_log.append((i, 'stop_loss', df.iloc[i]['Low'], balance))
                position = None
            elif df.iloc[i]['High'] > entry_price * (1 + take_profit):
                balance += (entry_price * (1 + take_profit) - entry_price) * (initial_balance / entry_price)
                trade_log.append((i, 'take_profit', df.iloc[i]['High'], balance))
                position = None
        elif position == 'short':
            if df.iloc[i]['High'] > entry_price * (1 + stop_loss):
                balance -= (entry_price * (1 + stop_loss) - entry_price) * (initial_balance / entry_price)
                trade_log.append((i, 'stop_loss', df.iloc[i]['High'], balance))
                position = None
            elif df.iloc[i]['Low'] < entry_price * (1 - take_profit):
                balance += (entry_price - entry_price * (1 - take_profit)) * (initial_balance / entry_price)
                trade_log.append((i, 'take_profit', df.iloc[i]['Low'], balance))
                position = None

    return trade_log, balance


# Plot candlestick chart with signals and balance
def plot_candlestick_with_signals_and_balance(df, start_index, num_rows, trade_log, title='Trading Strategy'):
    df_subset = df[start_index:start_index + num_rows]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    fig.add_trace(go.Candlestick(x=df_subset.index,
                                 open=df_subset['Open'],
                                 high=df_subset['High'],
                                 low=df_subset['Low'],
                                 close=df_subset['Close'],
                                 name='Candlesticks'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df_subset.index, y=df_subset['pointpos'], mode="markers",
                             marker=dict(size=10, color="MediumPurple", symbol='circle'),
                             name="Entry Points"),
                  row=1, col=1)

    balance_changes = [balance for _, _, _, balance in trade_log]
    balance_dates = [df.index[i] for i, _, _, _ in trade_log]

    fig.add_trace(go.Scatter(x=balance_dates, y=balance_changes, mode='lines+markers',
                             name='Balance', line=dict(color='blue', width=2)),
                  row=2, col=1)

    for trade in trade_log:
        index, action, price, _ = trade
        if action == 'buy':
            fig.add_trace(go.Scatter(x=[df.index[index]], y=[df.iloc[index]['Low'] - 0.02],
                                     mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
                                     showlegend=False),
                          row=1, col=1)
        elif action == 'sell':
            fig.add_trace(go.Scatter(x=[df.index[index]], y=[df.iloc[index]['High'] + 0.02],
                                     mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
                                     showlegend=False),
                          row=1, col=1)

    fig.update_layout(
        title=title,
        width=1200,
        height=800,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="white"
            ),
            bgcolor="black",
            bordercolor="gray",
            borderwidth=2
        )
    )

    fig.show()

