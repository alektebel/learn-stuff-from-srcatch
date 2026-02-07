# Phase 2: Statistical Techniques & Backtesting - Implementation Guidelines

## Overview
Phase 2 builds on Phase 1 by introducing advanced statistical analysis, professional-grade backtesting, and comprehensive risk management. You'll learn to analyze time series properties, build robust backtesting frameworks, and implement sophisticated risk controls.

## Learning Objectives
- Master time series statistical analysis
- Understand stationarity and cointegration
- Build event-driven backtesting systems
- Implement professional risk management
- Perform walk-forward optimization
- Use Monte Carlo simulation for strategy validation

## Prerequisites
- Completed Phase 1 (market data, basic strategies, simple backtesting)
- Understanding of statistics (mean, variance, correlation)
- Familiarity with hypothesis testing
- Basic knowledge of linear regression

## Required Packages
```bash
pip install pandas numpy scipy statsmodels arch scikit-learn matplotlib seaborn
```

## Components

### 1. Statistical Analysis (template_statistical_analysis.py)
**Time estimate**: 4-5 hours

**What you'll build**:
- Stationarity testing (ADF, KPSS)
- Autocorrelation analysis (ACF, PACF)
- Cointegration testing for pairs trading
- GARCH volatility modeling
- Statistical arbitrage strategies

**Step-by-step approach**:

#### Part 1: Stationarity Testing (60 minutes)
1. **Implement ADF Test**
   ```python
   from statsmodels.tsa.stattools import adfuller
   
   def adf_test(self, series, regression='c'):
       result = adfuller(series, regression=regression)
       return {
           'test_statistic': result[0],
           'p_value': result[1],
           'critical_values': result[4],
           'is_stationary': result[1] < 0.05
       }
   ```
   - Null hypothesis: series has unit root (non-stationary)
   - p-value < 0.05: reject null (series is stationary)
   - Most price series are non-stationary
   - Most return series are stationary

2. **Implement KPSS Test**
   ```python
   from statsmodels.tsa.stattools import kpss
   
   def kpss_test(self, series, regression='c'):
       result = kpss(series, regression=regression)
       return {
           'test_statistic': result[0],
           'p_value': result[1],
           'critical_values': result[3],
           'is_stationary': result[1] > 0.05
       }
   ```
   - Null hypothesis: series is stationary (opposite of ADF)
   - p-value > 0.05: fail to reject null (stationary)
   - Use both tests for confirmation

3. **Test on Real Data**
   ```python
   # Prices should be non-stationary
   price_adf = tester.adf_test(prices)
   print(f"Prices stationary: {price_adf['is_stationary']}")  # Should be False
   
   # Returns should be stationary
   returns = prices.pct_change().dropna()
   returns_adf = tester.adf_test(returns)
   print(f"Returns stationary: {returns_adf['is_stationary']}")  # Should be True
   ```

#### Part 2: Autocorrelation Analysis (45 minutes)
1. **Calculate ACF/PACF**
   ```python
   from statsmodels.tsa.stattools import acf, pacf
   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   
   def calculate_acf(self, series, nlags=40):
       return acf(series, nlags=nlags)
   
   def plot_acf_pacf(self, series, nlags=40):
       fig, axes = plt.subplots(2, 1, figsize=(12, 8))
       plot_acf(series, lags=nlags, ax=axes[0])
       plot_pacf(series, lags=nlags, ax=axes[1])
       plt.show()
   ```
   - ACF shows direct and indirect correlations
   - PACF shows only direct correlations
   - Use for identifying AR/MA orders

2. **Ljung-Box Test**
   ```python
   from statsmodels.stats.diagnostic import acorr_ljungbox
   
   def ljung_box_test(self, series, lags=10):
       result = acorr_ljungbox(series, lags=lags)
       return {
           'lb_stat': result['lb_stat'],
           'p_value': result['lb_pvalue']
       }
   ```
   - Tests for autocorrelation in residuals
   - p-value < 0.05: significant autocorrelation

#### Part 3: Cointegration Testing (90 minutes)
1. **Engle-Granger Test**
   ```python
   from statsmodels.regression.linear_model import OLS
   import statsmodels.api as sm
   
   def engle_granger_test(self, y, x):
       # Step 1: Run regression
       x_with_const = sm.add_constant(x)
       model = OLS(y, x_with_const)
       results = model.fit()
       hedge_ratio = results.params[1]
       
       # Step 2: Test residuals for stationarity
       residuals = results.resid
       adf_result = adfuller(residuals)
       
       return {
           'hedge_ratio': hedge_ratio,
           'adf_statistic': adf_result[0],
           'p_value': adf_result[1],
           'is_cointegrated': adf_result[1] < 0.05
       }
   ```

2. **Find Pairs**
   ```python
   def find_cointegrated_pairs(self, data, significance=0.05):
       pairs = []
       symbols = data.columns
       
       for i in range(len(symbols)):
           for j in range(i+1, len(symbols)):
               result = self.engle_granger_test(
                   data[symbols[i]], 
                   data[symbols[j]]
               )
               if result['is_cointegrated']:
                   pairs.append((
                       symbols[i], 
                       symbols[j], 
                       result['hedge_ratio']
                   ))
       
       return pairs
   ```

3. **Calculate Spread**
   ```python
   def calculate_spread(self, y, x, hedge_ratio):
       return y - hedge_ratio * x
   ```

#### Part 4: GARCH Modeling (60 minutes)
1. **Fit GARCH Model**
   ```python
   from arch import arch_model
   
   def fit(self, returns):
       # Scale returns to percentages for numerical stability
       returns_pct = returns * 100
       
       # Fit GARCH(1,1)
       model = arch_model(returns_pct, vol='Garch', p=self.p, q=self.q)
       self.results = model.fit(disp='off')
       
       return {
           'omega': self.results.params['omega'],
           'alpha': self.results.params['alpha[1]'],
           'beta': self.results.params['beta[1]']
       }
   ```

2. **Forecast Volatility**
   ```python
   def forecast_volatility(self, horizon=1):
       forecast = self.results.forecast(horizon=horizon)
       # Convert back from percentage terms
       vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100
       return vol_forecast
   ```

#### Part 5: Pairs Trading Strategy (75 minutes)
1. **Z-Score Signals**
   ```python
   def calculate_zscore(self, series, window=20):
       rolling_mean = series.rolling(window=window).mean()
       rolling_std = series.rolling(window=window).std()
       zscore = (series - rolling_mean) / rolling_std
       return zscore
   
   def generate_pairs_signals(self, spread, window=20):
       signals = pd.DataFrame(index=spread.index)
       signals['spread'] = spread
       signals['zscore'] = self.calculate_zscore(spread, window)
       
       # Entry signals
       signals['long'] = signals['zscore'] < -self.entry_threshold
       signals['short'] = signals['zscore'] > self.entry_threshold
       
       # Exit signals
       signals['exit'] = np.abs(signals['zscore']) < self.exit_threshold
       
       return signals
   ```

2. **Calculate Half-Life**
   ```python
   from statsmodels.tsa.ar_model import AutoReg
   
   def calculate_half_life(self, spread):
       # Fit AR(1): spread_t = phi * spread_{t-1} + epsilon
       model = AutoReg(spread.dropna(), lags=1)
       results = model.fit()
       phi = results.params[1]
       
       # Half-life = -log(2) / log(phi)
       half_life = -np.log(2) / np.log(phi)
       return half_life
   ```

**Testing**:
```python
# Load data for potential pairs
import yfinance as yf
data = yf.download(['KO', 'PEP'], start='2020-01-01')['Adj Close']

# Test for cointegration
analyzer = CointegrationAnalyzer()
result = analyzer.engle_granger_test(data['KO'], data['PEP'])
print(f"Cointegrated: {result['is_cointegrated']}")
print(f"Hedge Ratio: {result['hedge_ratio']:.4f}")

# Calculate and plot spread
spread = analyzer.calculate_spread(data['KO'], data['PEP'], result['hedge_ratio'])
spread.plot(title='KO-PEP Spread')

# Calculate half-life
strategy = StatisticalArbitrageStrategy()
half_life = strategy.calculate_half_life(spread)
print(f"Half-life: {half_life:.2f} days")
```

### 2. Advanced Backtesting (template_backtesting.py)
**Time estimate**: 5-6 hours

**What you'll build**:
- Event-driven backtesting engine
- Realistic execution simulation
- Walk-forward optimization
- Monte Carlo analysis
- Performance attribution

**Step-by-step approach**:

#### Part 1: Event-Driven Architecture (90 minutes)
1. **Implement DataHandler**
   ```python
   class HistoricDataHandler(DataHandler):
       def update_bars(self):
           if self.bar_index >= len(self.timestamps):
               return False
           
           current_time = self.timestamps[self.bar_index]
           
           for symbol in self.symbols:
               if current_time in self.data[symbol].index:
                   bar = self.data[symbol].loc[current_time]
                   self.latest_bars[symbol].append(bar)
           
           self.bar_index += 1
           return True
       
       def get_latest_bars(self, symbol, n=1):
           if symbol not in self.latest_bars:
               return pd.DataFrame()
           
           bars = self.latest_bars[symbol][-n:]
           return pd.DataFrame(bars)
   ```

2. **Implement ExecutionHandler**
   ```python
   def execute_order(self, order):
       # Get current price
       bars = self.data_handler.get_latest_bars(order.symbol, 1)
       if bars.empty:
           return None
       
       current_price = bars['Close'].iloc[0]
       
       # Calculate slippage
       slippage = self.slippage_model(
           order.symbol, 
           order.quantity, 
           order.order_type
       )
       
       # Determine fill price
       if order.direction == 'BUY':
           fill_price = current_price + slippage
       else:
           fill_price = current_price - slippage
       
       # Calculate commission
       commission = self.commission_model(order.quantity, fill_price)
       
       # Create fill event
       fill = FillEvent(
           timestamp=self.data_handler.get_latest_bar_datetime(),
           symbol=order.symbol,
           quantity=order.quantity,
           direction=order.direction,
           fill_price=fill_price,
           commission=commission,
           slippage=slippage
       )
       
       return fill
   ```

3. **Implement Main Loop**
   ```python
   def run(self):
       while self.data_handler.update_bars():
           # Create market event
           timestamp = self.data_handler.get_latest_bar_datetime()
           market_event = MarketEvent(timestamp, self.data_handler.symbols)
           
           # Strategy generates signals
           signals = self.strategy.generate_signals(market_event)
           
           for signal in signals:
               # Portfolio converts to orders
               order = self.portfolio.update_signal(signal)
               
               if order:
                   # Execute order
                   fill = self.execution_handler.execute_order(order)
                   
                   if fill:
                       # Update portfolio
                       self.portfolio.update_fill(fill)
           
           # Update portfolio value
           self.portfolio.update_portfolio_value()
   ```

#### Part 2: Walk-Forward Optimization (90 minutes)
```python
def run_walk_forward(self, train_period=252, test_period=63, step_size=21):
    results = []
    
    # Get all dates
    all_dates = sorted(set().union(*[set(df.index) for df in self.data.values()]))
    
    current_idx = 0
    while current_idx + train_period + test_period <= len(all_dates):
        # Split data
        train_start = all_dates[current_idx]
        train_end = all_dates[current_idx + train_period - 1]
        test_start = all_dates[current_idx + train_period]
        test_end = all_dates[current_idx + train_period + test_period - 1]
        
        # Optimize on training data
        train_data = {
            symbol: df[train_start:train_end] 
            for symbol, df in self.data.items()
        }
        best_params = self.optimize_window(train_data)
        
        # Test on out-of-sample data
        test_data = {
            symbol: df[test_start:test_end] 
            for symbol, df in self.data.items()
        }
        engine = BacktestEngine(
            test_data, 
            self.strategy_class, 
            **best_params
        )
        engine.run()
        results.append({
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end),
            'params': best_params,
            'performance': engine.calculate_performance()
        })
        
        # Roll forward
        current_idx += step_size
    
    return results
```

#### Part 3: Monte Carlo Simulation (60 minutes)
```python
def simulate(self, n_simulations=1000):
    results = []
    
    for i in range(n_simulations):
        # Randomly resample trades with replacement
        resampled_trades = np.random.choice(
            [t['pnl'] for t in self.trades],
            size=len(self.trades),
            replace=True
        )
        
        # Calculate cumulative return
        cumulative_pnl = np.cumsum(resampled_trades)
        final_pnl = cumulative_pnl[-1]
        max_drawdown = np.min(cumulative_pnl - np.maximum.accumulate(cumulative_pnl))
        
        results.append({
            'final_pnl': final_pnl,
            'max_drawdown': max_drawdown,
            'equity_curve': cumulative_pnl
        })
    
    # Calculate statistics
    final_pnls = [r['final_pnl'] for r in results]
    return {
        'mean_pnl': np.mean(final_pnls),
        'std_pnl': np.std(final_pnls),
        'percentile_5': np.percentile(final_pnls, 5),
        'percentile_95': np.percentile(final_pnls, 95),
        'prob_profit': np.mean(np.array(final_pnls) > 0)
    }
```

### 3. Risk Management (template_risk_management.py)
**Time estimate**: 4-5 hours

**Key implementations**:

#### VaR Calculation
```python
# Historical VaR
def historical_var(self, returns, horizon=1):
    var_percentile = (1 - self.confidence_level) * 100
    var = -np.percentile(returns, var_percentile)
    return var * np.sqrt(horizon)

# Parametric VaR
def parametric_var(self, returns, horizon=1):
    from scipy import stats
    z_score = stats.norm.ppf(1 - self.confidence_level)
    mean = returns.mean()
    std = returns.std()
    var = -(mean + z_score * std) * np.sqrt(horizon)
    return var

# CVaR
def calculate_cvar(self, returns, horizon=1):
    var_threshold = np.percentile(returns, (1 - self.confidence_level) * 100)
    cvar = -returns[returns <= var_threshold].mean()
    return cvar * np.sqrt(horizon)
```

#### Position Sizing
```python
# Kelly Criterion
def kelly_criterion(self, win_rate, avg_win, avg_loss):
    win_loss_ratio = avg_win / avg_loss
    kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
    # Use half-Kelly for safety
    return max(0, kelly_pct * 0.5)

# Fixed Fractional
def fixed_fractional(self, risk_per_trade, entry_price, stop_loss):
    capital_at_risk = self.total_capital * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss)
    shares = int(capital_at_risk / risk_per_share)
    return shares
```

#### Portfolio Optimization
```python
def maximize_sharpe(self, mean_returns, cov_matrix):
    n_assets = len(mean_returns)
    
    def neg_sharpe(weights):
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - self.risk_free_rate) / portfolio_std
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]
    initial_guess = np.ones(n_assets) / n_assets
    
    result = optimize.minimize(
        neg_sharpe,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x
```

## Common Issues and Solutions

### Issue 1: Non-stationary series in models
**Solution**: Always test for stationarity first, difference if needed
```python
if not is_stationary(prices):
    returns = prices.pct_change().dropna()
    # Use returns instead of prices
```

### Issue 2: Cointegration breaks down
**Solution**: Monitor rolling cointegration tests
```python
def rolling_cointegration(y, x, window=252):
    results = []
    for i in range(window, len(y)):
        result = engle_granger_test(y[i-window:i], x[i-window:i])
        results.append(result['p_value'])
    return results
```

### Issue 3: Walk-forward performance degradation
**Solution**: This is normal! Out-of-sample always worse
- Expect 30-50% Sharpe ratio degradation
- If more, likely overfitting
- Use simple strategies with few parameters

### Issue 4: Monte Carlo shows high risk of ruin
**Solution**: Reduce position sizes, improve risk management
```python
if risk_of_ruin > 0.01:  # More than 1% chance of 20% loss
    # Reduce position sizes by 50%
    position_size *= 0.5
```

## Validation Checklist

Before moving to Phase 3, ensure you can:
- [ ] Test any series for stationarity
- [ ] Find cointegrated pairs in stock universe
- [ ] Calculate optimal hedge ratios
- [ ] Fit GARCH model and forecast volatility
- [ ] Backtest pairs trading strategy profitably
- [ ] Run event-driven backtest without look-ahead bias
- [ ] Implement realistic slippage and commissions
- [ ] Perform walk-forward optimization
- [ ] Run Monte Carlo simulations
- [ ] Calculate VaR and CVaR
- [ ] Size positions using Kelly and fixed fractional
- [ ] Optimize portfolio weights (Markowitz)
- [ ] Calculate hedging requirements
- [ ] Monitor risk limits in real-time

## Expected Results

**Stationarity Tests**:
- Prices: Non-stationary (p-value > 0.05 in ADF)
- Returns: Stationary (p-value < 0.05 in ADF)

**Cointegration**:
- Good pairs: p-value < 0.05
- Common pairs: KO-PEP, XOM-CVX, JPM-BAC
- Hedge ratio typically 0.5-2.0

**GARCH**:
- Alpha + Beta ≈ 0.98 (high persistence)
- Omega small (long-run variance)
- Volatility clustering captured

**Pairs Trading**:
- Sharpe ratio: 1.0-2.0 for good pairs
- Win rate: 50-60%
- Avg trade: 1-3 days holding period

**Walk-Forward**:
- Out-of-sample Sharpe: 50-70% of in-sample
- Consistent profitability across windows = robust

**Monte Carlo**:
- Wide distribution of outcomes
- Risk of ruin should be < 5%

**Risk Management**:
- Kelly often suggests 5-20% per trade (too much!)
- Use 1-2% fixed fractional instead
- VaR and CVaR should align reasonably

## Next Steps

Once Phase 2 is complete, move to Phase 3 to learn:
- Feature engineering for ML
- Machine learning models for trading
- Time series cross-validation
- ML-based strategy development

## Resources

**Books**:
- "Analysis of Financial Time Series" by Ruey Tsay
- "Algorithmic Trading" by Ernie Chan
- "Advances in Financial Machine Learning" by Marcos López de Prado

**Papers**:
- Engle, R. (1982). "Autoregressive Conditional Heteroscedasticity"
- Engle, R. & Granger, C. (1987). "Co-integration and Error Correction"

**Online**:
- Statsmodels documentation
- ARCH package documentation
- QuantStart blog
- Robot Wealth blog

**Practice Datasets**:
- S&P 500 stocks (pairs trading)
- Forex pairs (cointegration)
- Crypto markets (high volatility for GARCH)
