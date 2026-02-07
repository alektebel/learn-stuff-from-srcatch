"""
Basic Trading Strategy - Template
==================================
This module implements simple trading strategies for learning purposes.

TODO: Implement the following components:
1. Moving average crossover strategy
2. Mean reversion strategy
3. Momentum strategy
4. Strategy performance metrics (Sharpe ratio, max drawdown, etc.)
5. Strategy backtesting framework

Learning objectives:
- Understand basic trading strategy logic
- Implement signal generation
- Calculate performance metrics
- Backtest strategies properly
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class TradingStrategy:
    """
    Base class for trading strategies.
    
    TODO 1: Complete the base strategy class
    - Define interface for all strategies
    - Implement common utility methods
    - Add performance tracking
    """
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.signals = pd.DataFrame()
        self.trades = []
        self.performance = {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with signals (-1, 0, 1 for sell, hold, buy)
        
        TODO 2: Define signal generation interface
        - This should be overridden by specific strategies
        - Return DataFrame with 'signal' column
        """
        raise NotImplementedError("Must implement generate_signals()")
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            returns: Series of strategy returns
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Dictionary with performance metrics
        
        TODO 3: Implement performance metrics
        - Sharpe ratio: (mean_return - rf) / std_return * sqrt(252)
        - Sortino ratio: like Sharpe but only downside deviation
        - Max drawdown: maximum peak-to-trough decline
        - Win rate: percentage of winning trades
        - Profit factor: gross profit / gross loss
        """
        # TODO: Implement metric calculations
        pass
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        TODO 4: Implement Sharpe ratio
        - Annualized: (mean_return - rf) / std_return * sqrt(252)
        - Handle zero std case
        """
        # TODO: Implement Sharpe ratio
        pass
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of portfolio values
        
        Returns:
            Maximum drawdown as percentage
        
        TODO 5: Implement max drawdown calculation
        - Calculate cumulative maximum
        - Find maximum percentage decline from peak
        """
        # TODO: Implement max drawdown
        pass


class MovingAverageCrossover(TradingStrategy):
    """
    Simple moving average crossover strategy.
    
    TODO 6: Implement MA crossover strategy
    - Generate buy signal when fast MA crosses above slow MA
    - Generate sell signal when fast MA crosses below slow MA
    """
    
    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        """
        Initialize MA crossover strategy.
        
        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
        """
        super().__init__(f"MA_Crossover_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on MA crossover.
        
        Args:
            data: DataFrame with 'Close' prices
        
        Returns:
            DataFrame with signals
        
        TODO 7: Implement MA crossover logic
        - Calculate fast and slow moving averages
        - Generate signal when fast crosses slow
        - 1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        
        # TODO: Calculate moving averages
        # signals['fast_ma'] = ...
        # signals['slow_ma'] = ...
        
        # TODO: Generate signals
        # signals['signal'] = ...
        
        # TODO: Return signals
        pass


class MeanReversion(TradingStrategy):
    """
    Mean reversion strategy using Bollinger Bands.
    
    TODO 8: Implement mean reversion strategy
    - Buy when price touches lower band
    - Sell when price touches upper band
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize mean reversion strategy.
        
        Args:
            period: Period for moving average and std calculation
            num_std: Number of standard deviations for bands
        """
        super().__init__(f"MeanReversion_{period}_{num_std}")
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on Bollinger Bands.
        
        Args:
            data: DataFrame with 'Close' prices
        
        Returns:
            DataFrame with signals
        
        TODO 9: Implement Bollinger Bands strategy
        - Calculate middle band (SMA)
        - Calculate upper and lower bands (mean +/- num_std * std)
        - Buy when price < lower band
        - Sell when price > upper band
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        
        # TODO: Calculate Bollinger Bands
        # signals['middle_band'] = ...
        # signals['std'] = ...
        # signals['upper_band'] = ...
        # signals['lower_band'] = ...
        
        # TODO: Generate signals
        # signals['signal'] = ...
        
        # TODO: Return signals
        pass


class MomentumStrategy(TradingStrategy):
    """
    Momentum strategy based on rate of change.
    
    TODO 10: Implement momentum strategy
    - Buy when momentum is positive and strong
    - Sell when momentum is negative and strong
    """
    
    def __init__(self, period: int = 20, threshold: float = 0.02):
        """
        Initialize momentum strategy.
        
        Args:
            period: Lookback period for momentum calculation
            threshold: Threshold for signal generation (2% default)
        """
        super().__init__(f"Momentum_{period}_{threshold}")
        self.period = period
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on momentum.
        
        Args:
            data: DataFrame with 'Close' prices
        
        Returns:
            DataFrame with signals
        
        TODO 11: Implement momentum strategy
        - Calculate price momentum (ROC)
        - Generate buy signal when momentum > threshold
        - Generate sell signal when momentum < -threshold
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        
        # TODO: Calculate momentum (rate of change)
        # momentum = (price - price.shift(period)) / price.shift(period)
        # signals['momentum'] = ...
        
        # TODO: Generate signals based on threshold
        # signals['signal'] = ...
        
        # TODO: Return signals
        pass


class StrategyBacktester:
    """
    Backtests trading strategies.
    
    TODO 12: Implement backtesting framework
    - Execute trades based on signals
    - Track portfolio value over time
    - Calculate returns and metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
    
    def run_backtest(
        self,
        strategy: TradingStrategy,
        data: pd.DataFrame
    ) -> Dict:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Strategy to backtest
            data: Historical market data
        
        Returns:
            Dictionary with backtest results
        
        TODO 13: Implement backtesting logic
        - Generate signals using strategy
        - Simulate trades based on signals
        - Track positions and cash
        - Calculate returns
        - Return performance metrics and equity curve
        """
        # TODO: Generate signals
        # signals = strategy.generate_signals(data)
        
        # TODO: Initialize portfolio tracking
        # portfolio_value = []
        # positions = 0
        # cash = self.initial_capital
        
        # TODO: Simulate trading
        # for i, row in signals.iterrows():
        #     # Execute trades based on signals
        #     # Update positions and cash
        #     # Track portfolio value
        
        # TODO: Calculate performance metrics
        # returns = ...
        # metrics = strategy.calculate_metrics(returns)
        
        # TODO: Return results
        pass
    
    def plot_results(self, results: Dict):
        """
        Plot backtest results.
        
        Args:
            results: Backtest results dictionary
        
        TODO 14: Implement result visualization
        - Plot equity curve
        - Plot drawdown
        - Plot signals on price chart
        - Show performance metrics
        """
        # TODO: Implement plotting
        pass


# Testing and example usage
if __name__ == "__main__":
    """
    TODO 15: Test your implementation
    
    Test cases to implement:
    1. Generate signals for each strategy
    2. Verify signal logic is correct
    3. Run backtest for each strategy
    4. Calculate and compare performance metrics
    5. Plot results
    """
    
    print("Test 1: MA Crossover Strategy...")
    # TODO: Create strategy and test signals
    
    print("\nTest 2: Mean Reversion Strategy...")
    # TODO: Create strategy and test signals
    
    print("\nTest 3: Momentum Strategy...")
    # TODO: Create strategy and test signals
    
    print("\nTest 4: Backtesting...")
    # TODO: Run backtests and compare strategies
    
    print("\nAll tests completed!")


"""
Implementation Guidelines:
==========================

Phase 1: Base Classes (20 minutes)
- Implement TradingStrategy base class
- Define signal generation interface
- Add basic metric calculations (Sharpe, max drawdown)
- Test with dummy data

Phase 2: Simple Strategies (40 minutes)
- Implement MovingAverageCrossover
  * Calculate fast and slow MAs
  * Detect crossovers
  * Generate signals
- Test with historical data
- Verify signals make sense

Phase 3: More Strategies (40 minutes)
- Implement MeanReversion (Bollinger Bands)
- Implement MomentumStrategy
- Test each strategy independently
- Compare signal patterns

Phase 4: Backtesting (40 minutes)
- Implement StrategyBacktester
- Simulate trades based on signals
- Track portfolio value
- Calculate returns and metrics
- Compare strategy performance

Tips:
- Use vectorized pandas operations for efficiency
- Be careful with look-ahead bias (don't use future data)
- Test strategies on different time periods
- Consider transaction costs realistically
- Start with simple implementations, add complexity later

Common Pitfalls:
- Look-ahead bias (using future information)
- Ignoring transaction costs
- Overfitting to historical data
- Not handling missing data properly
- Signal calculation errors at boundaries
- Not accounting for slippage
- Curve-fitting with too many parameters

Key Metrics:
- Sharpe Ratio: Risk-adjusted returns
- Max Drawdown: Worst peak-to-trough decline
- Win Rate: % of profitable trades
- Profit Factor: Gross profit / Gross loss
- Sortino Ratio: Sharpe using only downside risk
"""
