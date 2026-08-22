"""
Statistical Analysis for Trading - Template
============================================
This module implements statistical techniques for analyzing financial time series.

TODO: Implement the following components:
1. Stationarity tests (ADF, KPSS)
2. Autocorrelation analysis (ACF, PACF)
3. Cointegration tests for pairs trading
4. GARCH volatility modeling
5. Statistical arbitrage strategies

Learning objectives:
- Understand time series properties
- Test for stationarity and cointegration
- Model volatility with GARCH
- Implement pairs trading strategies
- Perform statistical arbitrage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StationarityTester:
    """
    Tests time series for stationarity using various statistical tests.
    
    TODO 1: Implement stationarity testing
    - Augmented Dickey-Fuller (ADF) test
    - Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    - Interpretation of results
    """
    
    def __init__(self):
        """
        Initialize stationarity tester.
        
        TODO: Set up test parameters
        - Default significance level (0.05)
        - Test configurations
        """
        self.significance_level = 0.05
        pass
    
    def adf_test(
        self, 
        series: pd.Series,
        regression: str = 'c'
    ) -> Dict:
        """
        Perform Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            regression: Type of regression ('c' for constant, 'ct' for constant+trend)
        
        Returns:
            Dictionary with test results
        
        TODO 2: Implement ADF test
        - Use statsmodels.tsa.stattools.adfuller
        - Test null hypothesis: series has unit root (non-stationary)
        - Return test statistic, p-value, critical values
        - Interpret: p-value < 0.05 means reject null (stationary)
        
        Example:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, regression=regression)
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < self.significance_level
            }
        """
        # TODO: Implement ADF test
        pass
    
    def kpss_test(
        self,
        series: pd.Series,
        regression: str = 'c'
    ) -> Dict:
        """
        Perform Kwiatkowski-Phillips-Schmidt-Shin test.
        
        Args:
            series: Time series to test
            regression: Type of regression ('c' or 'ct')
        
        Returns:
            Dictionary with test results
        
        TODO 3: Implement KPSS test
        - Use statsmodels.tsa.stattools.kpss
        - Test null hypothesis: series is stationary
        - Return test statistic, p-value, critical values
        - Interpret: p-value < 0.05 means reject null (non-stationary)
        - KPSS is complementary to ADF (reversed null hypothesis)
        """
        # TODO: Implement KPSS test
        pass
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Comprehensive stationarity check using multiple tests.
        
        Args:
            series: Time series to test
        
        Returns:
            Dictionary with all test results and recommendation
        
        TODO 4: Implement comprehensive stationarity check
        - Run both ADF and KPSS tests
        - Compare results
        - Provide recommendation (stationary, non-stationary, uncertain)
        - Suggest transformations if non-stationary (differencing, log)
        """
        # TODO: Run both tests and compare
        pass


class AutocorrelationAnalyzer:
    """
    Analyzes autocorrelation and partial autocorrelation.
    
    TODO 5: Implement autocorrelation analysis
    - ACF calculation and plotting
    - PACF calculation and plotting
    - Ljung-Box test for autocorrelation
    """
    
    def __init__(self):
        """Initialize autocorrelation analyzer."""
        pass
    
    def calculate_acf(
        self,
        series: pd.Series,
        nlags: int = 40
    ) -> np.ndarray:
        """
        Calculate autocorrelation function.
        
        Args:
            series: Time series
            nlags: Number of lags to calculate
        
        Returns:
            Array of autocorrelation values
        
        TODO 6: Implement ACF calculation
        - Use statsmodels.tsa.stattools.acf
        - Calculate for specified number of lags
        - Return array of correlation values
        
        Example:
            from statsmodels.tsa.stattools import acf
            return acf(series, nlags=nlags)
        """
        # TODO: Calculate ACF
        pass
    
    def calculate_pacf(
        self,
        series: pd.Series,
        nlags: int = 40
    ) -> np.ndarray:
        """
        Calculate partial autocorrelation function.
        
        Args:
            series: Time series
            nlags: Number of lags to calculate
        
        Returns:
            Array of partial autocorrelation values
        
        TODO 7: Implement PACF calculation
        - Use statsmodels.tsa.stattools.pacf
        - PACF removes indirect correlation effects
        - Useful for identifying AR order
        """
        # TODO: Calculate PACF
        pass
    
    def ljung_box_test(
        self,
        series: pd.Series,
        lags: int = 10
    ) -> Dict:
        """
        Perform Ljung-Box test for autocorrelation.
        
        Args:
            series: Time series (typically residuals)
            lags: Number of lags to test
        
        Returns:
            Dictionary with test results
        
        TODO 8: Implement Ljung-Box test
        - Use statsmodels.stats.diagnostic.acorr_ljungbox
        - Tests null hypothesis: no autocorrelation
        - Return test statistic and p-values for each lag
        - Significant p-values indicate autocorrelation
        """
        # TODO: Implement Ljung-Box test
        pass
    
    def plot_acf_pacf(
        self,
        series: pd.Series,
        nlags: int = 40
    ):
        """
        Plot ACF and PACF together.
        
        Args:
            series: Time series
            nlags: Number of lags to plot
        
        TODO 9: Implement ACF/PACF plotting
        - Create 2-subplot figure
        - Plot ACF in first subplot
        - Plot PACF in second subplot
        - Add confidence intervals
        - Use statsmodels.graphics.tsaplots
        """
        # TODO: Implement plotting
        pass


class CointegrationAnalyzer:
    """
    Analyzes cointegration between multiple time series for pairs trading.
    
    TODO 10: Implement cointegration analysis
    - Engle-Granger test
    - Johansen test
    - Hedge ratio calculation
    """
    
    def __init__(self):
        """Initialize cointegration analyzer."""
        self.significance_level = 0.05
        pass
    
    def engle_granger_test(
        self,
        y: pd.Series,
        x: pd.Series
    ) -> Dict:
        """
        Perform Engle-Granger cointegration test.
        
        Args:
            y: First time series (dependent)
            x: Second time series (independent)
        
        Returns:
            Dictionary with test results and hedge ratio
        
        TODO 11: Implement Engle-Granger test
        - Run OLS regression: y = beta * x + alpha
        - Get residuals from regression
        - Test residuals for stationarity using ADF
        - If residuals are stationary, series are cointegrated
        - Hedge ratio is the regression coefficient (beta)
        
        Example:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tsa.stattools import adfuller
            
            # Regression
            model = OLS(y, sm.add_constant(x))
            results = model.fit()
            hedge_ratio = results.params[1]
            
            # Test residuals
            residuals = results.resid
            adf_result = adfuller(residuals)
        """
        # TODO: Implement Engle-Granger test
        pass
    
    def johansen_test(
        self,
        data: pd.DataFrame,
        det_order: int = 0
    ) -> Dict:
        """
        Perform Johansen cointegration test.
        
        Args:
            data: DataFrame with multiple time series
            det_order: Deterministic term order
        
        Returns:
            Dictionary with test results
        
        TODO 12: Implement Johansen test
        - Use statsmodels.tsa.vector_ar.vecm.coint_johansen
        - Tests for multiple cointegrating relationships
        - More powerful than Engle-Granger for >2 series
        - Return trace statistics and eigenvalues
        """
        # TODO: Implement Johansen test
        pass
    
    def find_cointegrated_pairs(
        self,
        data: pd.DataFrame,
        significance: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        """
        Find all cointegrated pairs in a dataset.
        
        Args:
            data: DataFrame with multiple time series (columns are symbols)
            significance: Significance level for test
        
        Returns:
            List of tuples: (symbol1, symbol2, hedge_ratio)
        
        TODO 13: Implement pair finding
        - Test all pairs for cointegration
        - Use Engle-Granger test for each pair
        - Return only significant pairs
        - Include hedge ratio for each pair
        """
        # TODO: Find cointegrated pairs
        pass
    
    def calculate_spread(
        self,
        y: pd.Series,
        x: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """
        Calculate the spread between two cointegrated series.
        
        Args:
            y: First series
            x: Second series
            hedge_ratio: Hedge ratio from cointegration test
        
        Returns:
            Spread series
        
        TODO 14: Calculate spread
        - Spread = y - hedge_ratio * x
        - This spread should be mean-reverting if cointegrated
        - Can be used for pairs trading signals
        """
        # TODO: Calculate spread
        pass


class GARCHModel:
    """
    GARCH model for volatility forecasting.
    
    TODO 15: Implement GARCH modeling
    - Fit GARCH(1,1) model
    - Forecast volatility
    - Calculate Value at Risk (VaR)
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model.
        
        Args:
            p: AR order (GARCH component)
            q: MA order (ARCH component)
        
        TODO: Set up model parameters
        """
        self.p = p
        self.q = q
        self.model = None
        self.results = None
        pass
    
    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit GARCH model to returns.
        
        Args:
            returns: Return series (should be mean-zero)
        
        Returns:
            Dictionary with model parameters
        
        TODO 16: Implement GARCH fitting
        - Use arch package: from arch import arch_model
        - Fit GARCH(p, q) model
        - Store results for forecasting
        - Return omega, alpha, beta parameters
        
        Example:
            from arch import arch_model
            model = arch_model(returns * 100, vol='Garch', p=self.p, q=self.q)
            results = model.fit(disp='off')
            return results.params
        """
        # TODO: Fit GARCH model
        pass
    
    def forecast_volatility(
        self,
        horizon: int = 1
    ) -> pd.Series:
        """
        Forecast volatility for given horizon.
        
        Args:
            horizon: Forecast horizon in periods
        
        Returns:
            Series of volatility forecasts
        
        TODO 17: Implement volatility forecasting
        - Use fitted model to forecast
        - Return variance or standard deviation
        - Convert to annualized if needed
        """
        # TODO: Forecast volatility
        pass
    
    def calculate_var(
        self,
        confidence: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk using GARCH forecasts.
        
        Args:
            confidence: Confidence level (0.95 for 95% VaR)
            horizon: Time horizon
        
        Returns:
            VaR estimate
        
        TODO 18: Calculate VaR from GARCH
        - Forecast volatility for horizon
        - Calculate VaR: VaR = -z_score * volatility * sqrt(horizon)
        - z_score from normal distribution (1.645 for 95%)
        """
        # TODO: Calculate VaR
        pass


class StatisticalArbitrageStrategy:
    """
    Statistical arbitrage strategy using mean reversion.
    
    TODO 19: Implement statistical arbitrage
    - Z-score based trading signals
    - Pairs trading strategy
    - Dynamic position sizing
    """
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ):
        """
        Initialize statistical arbitrage strategy.
        
        Args:
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        
        TODO: Set up strategy parameters
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        pass
    
    def calculate_zscore(
        self,
        series: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling z-score.
        
        Args:
            series: Price or spread series
            window: Rolling window for mean and std
        
        Returns:
            Z-score series
        
        TODO 20: Calculate z-score
        - Rolling mean: series.rolling(window).mean()
        - Rolling std: series.rolling(window).std()
        - Z-score: (series - rolling_mean) / rolling_std
        """
        # TODO: Calculate z-score
        pass
    
    def generate_pairs_signals(
        self,
        spread: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Generate pairs trading signals from spread.
        
        Args:
            spread: Spread between two cointegrated assets
            window: Window for z-score calculation
        
        Returns:
            DataFrame with signals
        
        TODO 21: Generate pairs trading signals
        - Calculate z-score of spread
        - Long when z-score < -entry_threshold (spread too low)
        - Short when z-score > entry_threshold (spread too high)
        - Exit when abs(z-score) < exit_threshold
        - Return signal for both assets (opposite directions)
        """
        # TODO: Generate signals
        pass
    
    def backtest_pairs_strategy(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        hedge_ratio: float,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Backtest pairs trading strategy.
        
        Args:
            asset1: First asset prices
            asset2: Second asset prices
            hedge_ratio: Hedge ratio from cointegration
            initial_capital: Starting capital
        
        Returns:
            Dictionary with backtest results
        
        TODO 22: Backtest pairs strategy
        - Calculate spread using hedge ratio
        - Generate signals from spread
        - Simulate trades on both assets
        - Track portfolio value
        - Calculate performance metrics
        - Account for transaction costs
        """
        # TODO: Implement pairs backtesting
        pass
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.
        
        Args:
            spread: Spread series
        
        Returns:
            Half-life in periods
        
        TODO 23: Calculate half-life
        - Fit AR(1) model to spread
        - Half-life = -log(2) / log(phi)
        - Where phi is the AR(1) coefficient
        - Indicates how long spread takes to revert
        - Useful for position holding period
        
        Example:
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(spread, lags=1)
            results = model.fit()
            phi = results.params[1]
            half_life = -np.log(2) / np.log(phi)
        """
        # TODO: Calculate half-life
        pass


# Testing and example usage
if __name__ == "__main__":
    """
    TODO 24: Test your implementation
    
    Test cases to implement:
    1. Test stationarity on price vs returns
    2. Analyze autocorrelation in returns
    3. Find cointegrated pairs in stock universe
    4. Fit GARCH model and forecast volatility
    5. Backtest pairs trading strategy
    """
    
    print("Test 1: Stationarity Testing...")
    # TODO: Create sample data and test
    # - Load two related stocks (e.g., Coca-Cola and Pepsi)
    # - Test prices for stationarity (should be non-stationary)
    # - Test returns for stationarity (should be stationary)
    
    print("\nTest 2: Autocorrelation Analysis...")
    # TODO: Analyze autocorrelation
    # - Calculate ACF/PACF of returns
    # - Plot results
    # - Interpret patterns
    
    print("\nTest 3: Cointegration Testing...")
    # TODO: Test for cointegration
    # - Load multiple related stocks
    # - Find cointegrated pairs
    # - Calculate hedge ratios
    # - Plot spreads
    
    print("\nTest 4: GARCH Modeling...")
    # TODO: Fit GARCH model
    # - Calculate returns
    # - Fit GARCH(1,1)
    # - Forecast volatility
    # - Compare to realized volatility
    
    print("\nTest 5: Pairs Trading Backtest...")
    # TODO: Backtest pairs strategy
    # - Use cointegrated pair
    # - Generate signals
    # - Run backtest
    # - Analyze results
    
    print("\nAll tests completed!")


"""
Implementation Guidelines:
==========================

Phase 1: Stationarity Testing (30 minutes)
- Install required packages: pip install statsmodels arch
- Implement ADF and KPSS tests
- Test on price series (non-stationary) vs returns (stationary)
- Understand p-value interpretation
- Key insight: Most price series are non-stationary, returns are stationary

Phase 2: Autocorrelation Analysis (30 minutes)
- Implement ACF and PACF calculations
- Plot results to visualize patterns
- Ljung-Box test for residual diagnostics
- ACF useful for MA order, PACF for AR order
- Most return series show little autocorrelation (efficient markets)

Phase 3: Cointegration Testing (45 minutes)
- Implement Engle-Granger two-step procedure
- Find pairs in stock universe
- Calculate optimal hedge ratios
- Visualize spreads
- Look for pairs in same industry (KO-PEP, XOM-CVX)

Phase 4: GARCH Modeling (45 minutes)
- Fit GARCH(1,1) to return series
- Understand omega (long-run variance), alpha (ARCH), beta (GARCH)
- Forecast volatility
- Compare to historical volatility
- Use for risk management and option pricing

Phase 5: Statistical Arbitrage (60 minutes)
- Implement z-score based signals
- Build pairs trading strategy
- Calculate half-life for holding periods
- Backtest with realistic costs
- Monitor spread stationarity over time

Tips:
- Always check for stationarity before modeling
- Differencing makes most series stationary
- Cointegration is about long-run relationships
- GARCH captures volatility clustering
- Transaction costs are crucial for stat arb
- Pairs can lose cointegration over time

Common Pitfalls:
- Using non-stationary series in models
- Ignoring structural breaks
- Not accounting for transaction costs in pairs trading
- Over-fitting cointegration relationships
- Assuming cointegration is stable
- Look-ahead bias in z-score calculations
- Not testing for stability of hedge ratio

Key Concepts:
- Stationarity: Mean and variance constant over time
- Cointegration: Two non-stationary series with stationary linear combination
- GARCH: Generalized AutoRegressive Conditional Heteroskedasticity
- Half-life: Time for spread to revert halfway to mean
- Z-score: Standard deviations from mean

Performance Expectations:
- Good pairs trading: Sharpe ratio 1.0-2.0
- High frequency pairs: Even higher Sharpe
- Key is finding stable cointegration
- Monitor correlation breakdown
- Rebalance hedge ratios periodically

Resources:
- "Analysis of Financial Time Series" by Tsay
- "Algorithmic Trading" by Ernie Chan
- statsmodels documentation
- ARCH package documentation
"""
