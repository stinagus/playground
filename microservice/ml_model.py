from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import yfinance as yf
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import ta
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

def create_features(df):
    if df.empty or len(df) < 26:
        return pd.DataFrame()

    df = df.copy()

    """ Garman-Klass volatility: extends Parkinson's volatility by taking into account the opening and closing price.
        As markets are most active during the opening and closing of a trading session, it makes volatility estimation 
        more accurate. Takes into account intraday price extremums as well. Assumes continous diffusion process
     
        RSI: measures the speed and magnitude of a security's recent price changes to evaluate overvalued or undervalued 
        conditions in the price of that security. 
        
        Bollinger Bands: Helps gauge the volatility of stocks and other securities to determine if they are over- or undervalued.
        The center line is the stock price's 20-day simple moving average (SMA). The upper and lower bands are set at a 
        certain number of standard deviations, usually two, above and below the middle line.
     """
    df['garman_klass_vol'] = np.sqrt(0.5 * (np.log(df['High'] / df['Low']) ** 2) -
                                     (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2))
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Adj Close'], window=14).rsi()
    df['bb_low'] = ta.volatility.BollingerBands(close=df['Adj Close'], window=20).bollinger_lband()
    df['bb_mid'] = ta.volatility.BollingerBands(close=df['Adj Close'], window=20).bollinger_mavg()
    df['bb_high'] = ta.volatility.BollingerBands(close=df['Adj Close'], window=20).bollinger_hband()
    df['macd'] = ta.trend.MACD(close=df['Adj Close']).macd()

    for lag in range(1, 4):
        df[f'lag_{lag}'] = df['Adj Close'].shift(lag)

    df.dropna(inplace=True)

    return df

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_ml.html", {"request": request})

@app.post("/train", response_class=HTMLResponse)
async def train_model(request: Request, ticker: str = Form(...)):
    ticker = ticker.upper()

    df = yf.download(ticker, start="2020-01-01")

    df = create_features(df)

    df['target'] = df['Adj Close'].shift(-1)
    df.dropna(inplace=True)

    if df.empty:
        return templates.TemplateResponse("error.html", {"request": request, "ticker": ticker,
                                                         "error": "Not enough data to train the model."})

    X = df.drop(columns=['target'])
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_predictions)

    # Decide best model for return of final prediction
    if rf_mse < xgb_mse:
        best_model = rf_model.fit(X_scaled, y)
        best_prediction = rf_model.predict(X_scaled[-1].reshape(1, -1))
        chosen_model = "Random Forest"
    else:
        best_model = xgb_model.fit(X_scaled, y)
        best_prediction = xgb_model.predict(X_scaled[-1].reshape(1, -1))
        chosen_model = "XGBoost"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "ticker": ticker,
            "prediction": best_prediction[0],
            "model_used": chosen_model,
            "closing_price": df['Adj Close'].values[-1]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
