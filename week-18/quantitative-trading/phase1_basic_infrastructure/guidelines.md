# Phase 1: Basic Trading Infrastructure - Implementation Guidelines

## Overview
Phase 1 focuses on building the foundational components of a trading system. You'll learn how to fetch and process market data, manage orders and positions, and implement basic trading strategies.

## Learning Objectives
- Understand market data formats and processing
- Implement order management and position tracking
- Build simple trading strategies
- Calculate performance metrics
- Run basic backtests

## Components

### 1. Market Data Handler (template_market_data.py)
**Time estimate**: 2-3 hours

**What you'll build**:
- Data fetcher for Yahoo Finance or Alpha Vantage
- Data normalization and cleaning
- Return calculations (log and simple)
- Data persistence (save/load)

**Step-by-step approach**:
1. Start with `fetch_historical_data()`
   - Install yfinance: `pip install yfinance`
   - Use `yf.download()` to fetch data
   - Return DataFrame with OHLCV columns
   
2. Implement `normalize_data()`
   - Forward fill missing values: `df.fillna(method='ffill')`
   - Ensure datetime index: `df.index = pd.to_datetime(df.index)`
   - Drop any remaining NaNs at start: `df.dropna(inplace=True)`

3. Add `calculate_returns()`
   - Log returns: `np.log(df['Close'] / df['Close'].shift(1))`
   - Simple returns: `df['Close'].pct_change()`

4. Implement persistence
   - Save: `df.to_csv(filepath)` or `df.to_parquet(filepath)`
   - Load: `pd.read_csv(filepath, index_col=0, parse_dates=True)`

**Testing**:
```python
handler = MarketDataHandler()
data = handler.fetch_historical_data('AAPL', '2020-01-01', '2023-01-01')
print(data.head())
print(data.describe())
```

### 2. Order Management System (template_order_management.py)
**Time estimate**: 2-3 hours

**What you'll build**:
- Order class with validation
- Position tracking with P&L
- Portfolio management
- Execution simulator with slippage and commissions

**Step-by-step approach**:
1. Complete the `Order` class
   - Add validation in `__init__`: quantity > 0, price > 0 for limits
   - Use enums for type safety

2. Implement `Position.update()`
   ```python
   if quantity_change > 0:  # Buying
       # Update average price weighted by quantities
       total_cost = (self.quantity * self.average_price) + (quantity_change * price)
       self.quantity += quantity_change
       self.average_price = total_cost / self.quantity
   else:  # Selling
       # Calculate realized P&L
       self.realized_pnl += (-quantity_change) * (price - self.average_price)
       self.quantity += quantity_change
   ```

3. Implement `Portfolio.execute_order()`
   - Check cash available for buys
   - Update position via `position.update()`
   - Adjust cash: `self.cash -= (quantity * price + commission)`
   - Record trade in history

4. Implement `ExecutionSimulator.simulate_execution()`
   - Market order: execute immediately at current price + slippage
   - Limit order: check if limit price reached
   - Calculate commission as % of trade value

**Testing**:
```python
portfolio = Portfolio(initial_cash=100000)
order = Order('AAPL', OrderType.MARKET, OrderSide.BUY, 100)
simulator = ExecutionSimulator()
portfolio.execute_order(order, 150.0, commission=15.0)
print(portfolio.get_position('AAPL'))
```

### 3. Basic Trading Strategy (template_basic_strategy.py)
**Time estimate**: 2-3 hours

**What you'll build**:
- Moving average crossover strategy
- Mean reversion (Bollinger Bands)
- Momentum strategy
- Performance metrics (Sharpe ratio, max drawdown)
- Simple backtester

**Step-by-step approach**:
1. Implement `MovingAverageCrossover.generate_signals()`
   ```python
   signals['fast_ma'] = data['Close'].rolling(window=self.fast_period).mean()
   signals['slow_ma'] = data['Close'].rolling(window=self.slow_period).mean()
   
   # Generate signals
   signals['signal'] = 0
   signals.loc[signals['fast_ma'] > signals['slow_ma'], 'signal'] = 1  # Buy
   signals.loc[signals['fast_ma'] < signals['slow_ma'], 'signal'] = -1  # Sell
   ```

2. Implement `MeanReversion.generate_signals()`
   ```python
   signals['middle_band'] = data['Close'].rolling(window=self.period).mean()
   signals['std'] = data['Close'].rolling(window=self.period).std()
   signals['upper_band'] = signals['middle_band'] + (self.num_std * signals['std'])
   signals['lower_band'] = signals['middle_band'] - (self.num_std * signals['std'])
   
   signals['signal'] = 0
   signals.loc[data['Close'] < signals['lower_band'], 'signal'] = 1  # Buy
   signals.loc[data['Close'] > signals['upper_band'], 'signal'] = -1  # Sell
   ```

3. Implement `calculate_sharpe_ratio()`
   ```python
   excess_returns = returns.mean() - risk_free_rate/252
   std_returns = returns.std()
   sharpe = (excess_returns / std_returns) * np.sqrt(252) if std_returns > 0 else 0
   return sharpe
   ```

4. Implement `calculate_max_drawdown()`
   ```python
   cummax = equity_curve.cummax()
   drawdown = (equity_curve - cummax) / cummax
   return drawdown.min()
   ```

5. Implement `StrategyBacktester.run_backtest()`
   ```python
   signals = strategy.generate_signals(data)
   portfolio_value = [self.initial_capital]
   cash = self.initial_capital
   position = 0
   
   for i in range(1, len(signals)):
       # Get signal and price
       signal = signals.iloc[i]['signal']
       price = signals.iloc[i]['price']
       
       # Execute trades based on signal changes
       if signal == 1 and position == 0:  # Buy
           shares = int(cash / price)
           cost = shares * price * (1 + self.commission)
           if cost <= cash:
               position = shares
               cash -= cost
       elif signal == -1 and position > 0:  # Sell
           proceeds = position * price * (1 - self.commission)
           cash += proceeds
               position = 0
       
       # Calculate portfolio value
       value = cash + position * price
       portfolio_value.append(value)
   ```

**Testing**:
```python
# Fetch data
handler = MarketDataHandler()
data = handler.fetch_historical_data('AAPL', '2020-01-01', '2023-01-01')

# Test MA crossover
strategy = MovingAverageCrossover(fast_period=50, slow_period=200)
backtester = StrategyBacktester(initial_capital=100000)
results = backtester.run_backtest(strategy, data)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Total Return: {results['total_return']:.2%}")
```

## Common Issues and Solutions

### Issue 1: Missing data in API calls
**Solution**: Use forward fill for prices, drop leading NaNs
```python
df = df.fillna(method='ffill').dropna()
```

### Issue 2: Look-ahead bias in signals
**Solution**: Always use `.shift(1)` when backtesting to ensure you're using yesterday's signal for today's trade
```python
signals['position'] = signals['signal'].shift(1)
```

### Issue 3: Insufficient cash for orders
**Solution**: Check cash before executing
```python
if quantity * price * (1 + commission_pct) <= cash:
    # Execute order
```

### Issue 4: Division by zero in metrics
**Solution**: Check for zero denominator
```python
sharpe = (mean / std) * np.sqrt(252) if std > 0 else 0
```

## Validation Checklist

Before moving to Phase 2, ensure you can:
- [ ] Fetch historical data for any symbol
- [ ] Calculate returns correctly (log and simple)
- [ ] Create and execute orders
- [ ] Track positions with correct average price
- [ ] Calculate realized and unrealized P&L
- [ ] Generate signals from all three strategies
- [ ] Run a backtest with realistic costs
- [ ] Calculate Sharpe ratio and max drawdown
- [ ] Get sensible results (positive Sharpe for good strategies)

## Expected Results

With a good implementation, you should see:
- MA Crossover on AAPL (2020-2023): Sharpe ~0.5-1.0
- Mean Reversion: Works better in ranging markets
- Momentum: Works better in trending markets
- Max drawdown typically 10-30% for simple strategies

## Next Steps

Once Phase 1 is complete, move to Phase 2 to learn:
- Advanced statistical analysis
- Proper walk-forward testing
- Risk management techniques
- More sophisticated backtesting

## Resources

- **yfinance docs**: https://pypi.org/project/yfinance/
- **Pandas time series**: https://pandas.pydata.org/docs/user_guide/timeseries.html
- **Backtesting basics**: "Algorithmic Trading" by Ernie Chan
- **Performance metrics**: "Quantitative Trading" by Ernie Chan
