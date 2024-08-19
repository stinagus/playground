from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf

app = Flask(__name__)

@app.template_filter('comma')
def comma_format(value):
    return f"{value:,}"

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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker").upper()
        print(f"Form submitted with ticker: {ticker}")
        return redirect(url_for("stock", ticker=ticker))
    return render_template("index.html")


@app.route("/stock/<ticker>", methods=["GET"])
def stock(ticker):
    print(f"Fetching data for ticker: {ticker}")  # Debugging statement
    closing_price, volume, ticker = get_stock_data(ticker)
    if closing_price and volume:
        return f"Ticker: {ticker}, Closing Price: {closing_price}, Volume: {volume}"
    else:
        return f"No data found for ticker {ticker}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
