from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
import yfinance as yf
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def comma_format(value):
    return f"{value:,}"

templates = Jinja2Templates(directory="templates")
templates.env.filters['comma'] = comma_format

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def get_ticker(request: Request, ticker: str = Form(...)):
    ticker = ticker.upper()
    return RedirectResponse(url=f"/stock/{ticker}", status_code=303)

def get_stock_data(ticker="AAPL"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            latest_data = hist.iloc[-1]
            closing_price = latest_data["Close"]
            volume = latest_data["Volume"]
            return closing_price, volume, ticker
        else:
            print("Error: No data returned for the ticker")
            return None, None, ticker
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None, None, ticker

@app.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock(request: Request, ticker: str):
    closing_price, volume, ticker = get_stock_data(ticker)
    if closing_price and volume:
        return templates.TemplateResponse("stock_index.html", {"request": request, "ticker": ticker, "closing_price": closing_price, "volume": volume})
    else:
        return templates.TemplateResponse("stock_index.html", {"request": request, "ticker": ticker, "error": f"No data found for ticker {ticker}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
