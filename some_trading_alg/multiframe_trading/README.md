# Multiframe Trading

## Rejection and Engulfing Signal Strategy

### Candlestick Analysis

Candlestick charts are essential tools in technical analysis, depicting price movements over a chosen time frame. Each candlestick comprises a body (the rectangular part) and wicks (lines extending above and below the body). Here’s what each component signifies:

- **Body**: Represents the opening and closing prices of the period. A filled (red or black) body means the closing price was lower than the opening price, indicating bearish sentiment. An empty (green or white) body shows the closing price was higher, suggesting bullish sentiment.

- **Wicks**: Also known as shadows, these lines extend from the top and bottom of the body, indicating the highest and lowest prices reached during the period.

### Rejection Candlestick Pattern

The rejection candlestick pattern, also referred to as a pin bar or hammer, signals potential reversals in price direction at key levels of support or resistance. Here are the characteristics of a rejection candlestick:

- **Small Body**: The body of the candle is relatively small compared to its wicks.

- **Long Wicks**: The candle has long upper and/or lower wicks (shadows), indicating price rejection from higher or lower levels.

- **Closing Price**: For a bullish rejection, the closing price is significantly higher than the opening price, suggesting bulls regained control after testing lower levels. Conversely, for a bearish rejection, the closing price is notably lower than the opening price, indicating bears regained dominance after testing higher levels.

#### Implementation in Code

The `is_bullish_rejection` and `is_bearish_rejection` functions in the Python code detect these patterns based on specific criteria:

- **Tail-to-Body Ratio**: Determines the significance of the wicks relative to the body size.

- **Body Percentage Limit**: Sets a threshold for how small the body can be relative to the closing price.

### Engulfing Candlestick Pattern

The engulfing candlestick pattern is a two-candle reversal pattern where the body of the second candle completely engulfs the body of the first candle. It signals a shift in market sentiment:

- **Bullish Engulfing**: The second candle (bullish) opens below the close of the previous candle (bearish) and closes above its open, engulfing the entire body of the previous candle. This suggests a reversal from bearish to bullish sentiment.

- **Bearish Engulfing**: The second candle (bearish) opens above the close of the previous candle (bullish) and closes below its open, engulfing the entire body of the previous candle. This indicates a reversal from bullish to bearish sentiment.

#### Implementation in Code

The `is_bullish_engulfing` and `is_bearish_engulfing` functions in the code detect these patterns by comparing the open, close, and previous candlestick’s open and close prices.
