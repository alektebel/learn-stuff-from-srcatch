"""
Risk Management System - Template
==================================
This module implements comprehensive risk management for trading strategies.

TODO: Implement the following components:
1. Value at Risk (VaR) calculation (historical, parametric, Monte Carlo)
2. Conditional Value at Risk (CVaR/Expected Shortfall)
3. Position sizing (Kelly criterion, risk parity, volatility targeting)
4. Portfolio optimization (Markowitz, Black-Litterman)
5. Dynamic hedging strategies

Learning objectives:
- Understand risk measurement techniques
- Implement optimal position sizing
- Build portfolio optimization
- Design hedging strategies
- Manage portfolio risk dynamically
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import optimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class VaRCalculator:
    """
    Calculates Value at Risk using multiple methods.
    
    TODO 1: Implement VaR calculations
    - Historical VaR
    - Parametric VaR (variance-covariance)
    - Monte Carlo VaR
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level (0.95 for 95% VaR)
        
        TODO: Set up calculator
        """
        self.confidence_level = confidence_level
        pass
    
    def historical_var(
        self,
        returns: pd.Series,
        horizon: int = 1
    ) -> float:
        """
        Calculate historical VaR.
        
        Args:
            returns: Return series
            horizon: Time horizon in days
        
        Returns:
            VaR estimate (positive number represents loss)
        
        TODO 2: Implement historical VaR
        - Sort returns from worst to best
        - Find percentile at (1 - confidence_level)
        - Scale by sqrt(horizon) for multi-day VaR
        - No distributional assumptions needed
        
        Example:
            # 95% VaR means 5% worst case
            var_percentile = (1 - self.confidence_level)
            var = -np.percentile(returns, var_percentile * 100)
            return var * np.sqrt(horizon)
        """
        # TODO: Calculate historical VaR
        pass
    
    def parametric_var(
        self,
        returns: pd.Series,
        horizon: int = 1
    ) -> float:
        """
        Calculate parametric VaR assuming normal distribution.
        
        Args:
            returns: Return series
            horizon: Time horizon in days
        
        Returns:
            VaR estimate
        
        TODO 3: Implement parametric VaR
        - Calculate mean and std of returns
        - Use normal distribution z-score
        - VaR = -(mean + z * std) * sqrt(horizon)
        - z = 1.645 for 95%, 2.326 for 99%
        
        Example:
            from scipy import stats
            z_score = stats.norm.ppf(1 - self.confidence_level)
            mean = returns.mean()
            std = returns.std()
            var = -(mean + z_score * std) * np.sqrt(horizon)
        """
        # TODO: Calculate parametric VaR
        pass
    
    def monte_carlo_var(
        self,
        returns: pd.Series,
        horizon: int = 1,
        n_simulations: int = 10000
    ) -> float:
        """
        Calculate Monte Carlo VaR.
        
        Args:
            returns: Return series
            horizon: Time horizon in days
            n_simulations: Number of simulations
        
        Returns:
            VaR estimate
        
        TODO 4: Implement Monte Carlo VaR
        - Estimate distribution parameters from returns
        - Simulate n_simulations paths over horizon
        - Calculate percentile of simulated returns
        - More flexible than parametric (can use any distribution)
        
        Steps:
        1. Fit distribution to returns (normal, t-distribution, etc.)
        2. Generate random returns from distribution
        3. Calculate cumulative return over horizon
        4. Find percentile
        """
        # TODO: Calculate Monte Carlo VaR
        pass
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        horizon: int = 1
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Return series
            horizon: Time horizon in days
        
        Returns:
            CVaR estimate
        
        TODO 5: Implement CVaR
        - CVaR = expected loss given loss exceeds VaR
        - Average of all returns below VaR threshold
        - More conservative than VaR
        - Coherent risk measure
        
        Example:
            var_threshold = np.percentile(returns, (1 - self.confidence_level) * 100)
            cvar = -returns[returns <= var_threshold].mean()
            return cvar * np.sqrt(horizon)
        """
        # TODO: Calculate CVaR
        pass
    
    def portfolio_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        horizon: int = 1
    ) -> float:
        """
        Calculate portfolio VaR with correlations.
        
        Args:
            returns: DataFrame with returns for multiple assets
            weights: Portfolio weights
            horizon: Time horizon
        
        Returns:
            Portfolio VaR
        
        TODO 6: Implement portfolio VaR
        - Calculate portfolio returns: returns @ weights
        - Account for correlations via covariance matrix
        - More accurate than individual VaRs
        - Diversification reduces portfolio VaR
        
        Steps:
        1. Calculate portfolio return series
        2. Calculate VaR on portfolio returns
        OR:
        1. Calculate covariance matrix
        2. Portfolio variance = weights.T @ cov @ weights
        3. VaR = z_score * sqrt(portfolio_variance) * sqrt(horizon)
        """
        # TODO: Calculate portfolio VaR
        pass


class PositionSizer:
    """
    Calculates optimal position sizes using various methods.
    
    TODO 7: Implement position sizing
    - Kelly criterion
    - Fixed fractional
    - Risk parity
    - Volatility targeting
    """
    
    def __init__(self, total_capital: float):
        """
        Initialize position sizer.
        
        Args:
            total_capital: Total available capital
        
        TODO: Set up position sizer
        """
        self.total_capital = total_capital
        pass
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion position size.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
        
        Returns:
            Optimal fraction of capital to risk
        
        TODO 8: Implement Kelly criterion
        - Kelly% = W - (1-W)/(R)
        - W = win rate
        - R = avg_win / avg_loss (win/loss ratio)
        - Result is fraction of capital to bet
        - Usually use half-Kelly or less (less aggressive)
        
        Example:
            win_loss_ratio = avg_win / avg_loss
            kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
            # Use half-Kelly for safety
            return max(0, kelly_pct * 0.5)
        """
        # TODO: Calculate Kelly criterion
        pass
    
    def fixed_fractional(
        self,
        risk_per_trade: float = 0.01,
        entry_price: float = 100,
        stop_loss: float = 95
    ) -> int:
        """
        Calculate position size using fixed fractional method.
        
        Args:
            risk_per_trade: Fraction of capital to risk (0.01 = 1%)
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Number of shares to trade
        
        TODO 9: Implement fixed fractional sizing
        - Risk per trade = capital * risk_per_trade
        - Risk per share = abs(entry_price - stop_loss)
        - Position size = risk per trade / risk per share
        - Simple and effective method
        
        Example:
            capital_at_risk = self.total_capital * risk_per_trade
            risk_per_share = abs(entry_price - stop_loss)
            shares = int(capital_at_risk / risk_per_share)
        """
        # TODO: Calculate fixed fractional size
        pass
    
    def volatility_targeting(
        self,
        target_volatility: float,
        current_volatility: float,
        base_position: int
    ) -> int:
        """
        Adjust position size to target volatility.
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            current_volatility: Current asset volatility
            base_position: Base position size
        
        Returns:
            Adjusted position size
        
        TODO 10: Implement volatility targeting
        - Scale position inversely with volatility
        - When volatility high, reduce position
        - When volatility low, increase position
        - Maintains consistent risk exposure
        
        Example:
            vol_scalar = target_volatility / current_volatility
            adjusted_position = int(base_position * vol_scalar)
        """
        # TODO: Calculate volatility-targeted size
        pass
    
    def risk_parity_weights(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate risk parity portfolio weights.
        
        Args:
            returns: DataFrame with returns for multiple assets
        
        Returns:
            Array of portfolio weights
        
        TODO 11: Implement risk parity
        - Each asset contributes equally to portfolio risk
        - Weight inversely proportional to volatility
        - More diversified than market-cap weighting
        
        Steps:
        1. Calculate volatility for each asset
        2. Initial weights = 1 / volatility
        3. Normalize weights to sum to 1
        4. Iterate to equalize risk contributions
        
        Simple version:
            vols = returns.std()
            weights = 1 / vols
            weights = weights / weights.sum()
        """
        # TODO: Calculate risk parity weights
        pass


class PortfolioOptimizer:
    """
    Optimizes portfolio weights using modern portfolio theory.
    
    TODO 12: Implement portfolio optimization
    - Mean-variance optimization (Markowitz)
    - Minimum variance portfolio
    - Maximum Sharpe ratio portfolio
    - Black-Litterman model
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
        
        TODO: Set up optimizer
        """
        self.risk_free_rate = risk_free_rate
        pass
    
    def calculate_portfolio_stats(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio statistics.
        
        Args:
            weights: Portfolio weights
            mean_returns: Mean returns for each asset
            cov_matrix: Covariance matrix
        
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        
        TODO 13: Calculate portfolio metrics
        - Portfolio return = weights @ mean_returns
        - Portfolio variance = weights.T @ cov_matrix @ weights
        - Portfolio volatility = sqrt(variance)
        - Sharpe ratio = (return - rf) / volatility
        """
        # TODO: Calculate portfolio statistics
        pass
    
    def minimize_variance(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Find minimum variance portfolio.
        
        Args:
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
        
        Returns:
            Optimal weights
        
        TODO 14: Implement minimum variance optimization
        - Minimize: weights.T @ cov_matrix @ weights
        - Subject to: weights.sum() = 1
        - Optional: weights >= 0 (no short selling)
        - Use scipy.optimize.minimize
        
        Example:
            def objective(w):
                return w @ cov_matrix @ w
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(mean_returns))]
            initial_guess = np.ones(len(mean_returns)) / len(mean_returns)
            
            result = optimize.minimize(
                objective, initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        """
        # TODO: Minimize variance
        pass
    
    def maximize_sharpe(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Find maximum Sharpe ratio portfolio.
        
        Args:
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
        
        Returns:
            Optimal weights
        
        TODO 15: Implement maximum Sharpe optimization
        - Maximize: (portfolio_return - rf) / portfolio_volatility
        - Equivalent to minimize: -Sharpe ratio
        - Constraints: weights sum to 1, weights >= 0
        
        Example:
            def neg_sharpe(w):
                ret = w @ mean_returns
                vol = np.sqrt(w @ cov_matrix @ w)
                return -(ret - self.risk_free_rate) / vol
        """
        # TODO: Maximize Sharpe ratio
        pass
    
    def efficient_frontier(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate efficient frontier.
        
        Args:
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            n_points: Number of points on frontier
        
        Returns:
            Tuple of (returns, volatilities)
        
        TODO 16: Generate efficient frontier
        - For target returns from min to max:
          1. Minimize variance for that return
          2. Record return and volatility
        - Plot shows return vs risk trade-off
        - All points are optimal for given risk level
        """
        # TODO: Generate efficient frontier
        pass
    
    def black_litterman(
        self,
        market_caps: np.ndarray,
        cov_matrix: np.ndarray,
        views: Dict[int, float],
        view_confidences: Dict[int, float],
        tau: float = 0.05
    ) -> np.ndarray:
        """
        Implement Black-Litterman model.
        
        Args:
            market_caps: Market capitalizations
            cov_matrix: Covariance matrix
            views: Dictionary of {asset_index: expected_return}
            view_confidences: Dictionary of {asset_index: confidence}
            tau: Uncertainty scaling parameter
        
        Returns:
            Posterior expected returns
        
        TODO 17: Implement Black-Litterman
        - Combines market equilibrium with investor views
        - More stable than pure mean-variance
        - Incorporates view uncertainty
        
        Steps:
        1. Calculate implied equilibrium returns
        2. Incorporate views with uncertainty
        3. Combine using Bayes rule
        4. Return posterior expected returns
        
        Note: This is advanced - start with simpler methods
        """
        # TODO: Implement Black-Litterman
        pass


class DynamicHedger:
    """
    Implements dynamic hedging strategies.
    
    TODO 18: Implement hedging strategies
    - Delta hedging
    - Beta hedging
    - Dynamic portfolio insurance
    """
    
    def __init__(self):
        """Initialize dynamic hedger."""
        pass
    
    def calculate_portfolio_beta(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate portfolio beta relative to market.
        
        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market return series
        
        Returns:
            Portfolio beta
        
        TODO 19: Calculate beta
        - Beta = Cov(portfolio, market) / Var(market)
        - Measures systematic risk
        - Beta > 1: more volatile than market
        - Beta < 1: less volatile than market
        
        Example:
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance
        """
        # TODO: Calculate beta
        pass
    
    def calculate_hedge_ratio(
        self,
        asset_returns: pd.Series,
        hedge_returns: pd.Series
    ) -> float:
        """
        Calculate optimal hedge ratio.
        
        Args:
            asset_returns: Asset to hedge
            hedge_returns: Hedging instrument returns
        
        Returns:
            Optimal hedge ratio
        
        TODO 20: Calculate hedge ratio
        - Run regression: asset_returns ~ hedge_returns
        - Hedge ratio = regression coefficient
        - Minimizes variance of hedged position
        
        Example:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                hedge_returns, asset_returns
            )
            return slope
        """
        # TODO: Calculate hedge ratio
        pass
    
    def beta_hedge(
        self,
        portfolio_value: float,
        portfolio_beta: float,
        target_beta: float = 0.0,
        futures_price: float = 4000,
        futures_multiplier: float = 50
    ) -> int:
        """
        Calculate futures contracts needed for beta hedge.
        
        Args:
            portfolio_value: Portfolio value
            portfolio_beta: Current portfolio beta
            target_beta: Desired beta (0 for market neutral)
            futures_price: Futures price
            futures_multiplier: Futures contract multiplier
        
        Returns:
            Number of futures contracts to short
        
        TODO 21: Calculate beta hedge
        - Need to offset (portfolio_beta - target_beta)
        - Each futures contract has beta = 1
        - Contracts = (beta_diff * portfolio_value) / (futures_price * multiplier)
        
        Example:
            beta_exposure = portfolio_value * (portfolio_beta - target_beta)
            contract_value = futures_price * futures_multiplier
            contracts = int(beta_exposure / contract_value)
        """
        # TODO: Calculate beta hedge
        pass
    
    def calculate_cppi(
        self,
        portfolio_value: float,
        floor_value: float,
        multiplier: float = 3.0
    ) -> float:
        """
        Calculate Constant Proportion Portfolio Insurance allocation.
        
        Args:
            portfolio_value: Current portfolio value
            floor_value: Floor value (protection level)
            multiplier: CPPI multiplier (typically 2-5)
        
        Returns:
            Amount to allocate to risky assets
        
        TODO 22: Implement CPPI
        - Cushion = portfolio_value - floor_value
        - Risky allocation = multiplier * cushion
        - Remaining in safe assets
        - Dynamically adjusts exposure based on cushion
        - Provides downside protection
        
        Example:
            cushion = max(0, portfolio_value - floor_value)
            risky_allocation = min(
                portfolio_value,
                multiplier * cushion
            )
        """
        # TODO: Calculate CPPI allocation
        pass
    
    def rebalance_signals(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        threshold: float = 0.05
    ) -> Dict[int, float]:
        """
        Generate rebalancing signals.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Rebalancing threshold
        
        Returns:
            Dictionary of {asset_index: trade_amount}
        
        TODO 23: Generate rebalancing signals
        - Compare current vs target weights
        - Only rebalance if difference > threshold
        - Reduces transaction costs
        - Return trades needed to reach target
        """
        # TODO: Generate rebalancing signals
        pass


class RiskMonitor:
    """
    Monitors portfolio risk in real-time.
    
    TODO 24: Implement risk monitoring
    - Position limits
    - Concentration limits
    - Loss limits
    - Margin requirements
    """
    
    def __init__(self, limits: Dict):
        """
        Initialize risk monitor.
        
        Args:
            limits: Dictionary of risk limits
        
        TODO: Set up monitoring
        """
        self.limits = limits
        self.alerts = []
        pass
    
    def check_position_limits(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> List[str]:
        """
        Check if positions exceed limits.
        
        Args:
            positions: Current positions {symbol: quantity}
            prices: Current prices {symbol: price}
            portfolio_value: Total portfolio value
        
        Returns:
            List of limit violations
        
        TODO 25: Check position limits
        - Single position limit (e.g., max 20% per position)
        - Sector concentration (e.g., max 40% per sector)
        - Long/short exposure limits
        - Return list of violations
        """
        # TODO: Check position limits
        pass
    
    def check_loss_limits(
        self,
        current_pnl: float,
        daily_loss_limit: float,
        drawdown_limit: float,
        peak_value: float,
        current_value: float
    ) -> List[str]:
        """
        Check if loss limits exceeded.
        
        Args:
            current_pnl: Current day P&L
            daily_loss_limit: Max daily loss allowed
            drawdown_limit: Max drawdown allowed
            peak_value: Historical peak value
            current_value: Current portfolio value
        
        Returns:
            List of violations
        
        TODO 26: Check loss limits
        - Daily loss limit
        - Maximum drawdown from peak
        - Return violations for immediate action
        - Critical for risk management
        """
        # TODO: Check loss limits
        pass
    
    def calculate_margin_requirement(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        margin_rates: Dict[str, float]
    ) -> float:
        """
        Calculate margin requirement.
        
        Args:
            positions: Current positions
            prices: Current prices
            margin_rates: Margin rates per symbol
        
        Returns:
            Total margin required
        
        TODO 27: Calculate margin
        - Margin = position_value * margin_rate
        - Different rates for different assets
        - Must have enough capital for margin
        - Critical for leveraged strategies
        """
        # TODO: Calculate margin requirement
        pass


# Testing and example usage
if __name__ == "__main__":
    """
    TODO 28: Test your implementation
    
    Test cases to implement:
    1. Calculate VaR using all three methods
    2. Test position sizing with Kelly and fixed fractional
    3. Optimize portfolio using Markowitz
    4. Calculate hedge ratios
    5. Monitor risk limits
    """
    
    print("Test 1: VaR Calculation...")
    # TODO: Generate sample returns and calculate VaR
    # - Compare historical, parametric, Monte Carlo
    # - Calculate CVaR
    # - Verify Monte Carlo converges to parametric for normal data
    
    print("\nTest 2: Position Sizing...")
    # TODO: Test position sizing methods
    # - Kelly criterion with sample win rate and ratios
    # - Fixed fractional with stop loss
    # - Volatility targeting
    # - Compare position sizes
    
    print("\nTest 3: Portfolio Optimization...")
    # TODO: Optimize portfolio
    # - Create sample return data for 5 assets
    # - Find minimum variance portfolio
    # - Find maximum Sharpe portfolio
    # - Generate efficient frontier
    # - Plot results
    
    print("\nTest 4: Hedging...")
    # TODO: Test hedging strategies
    # - Calculate portfolio beta
    # - Calculate hedge ratio
    # - Determine futures contracts needed
    # - Test CPPI allocation
    
    print("\nTest 5: Risk Monitoring...")
    # TODO: Test risk monitoring
    # - Set up position limits
    # - Check for violations
    # - Monitor loss limits
    # - Calculate margin requirements
    
    print("\nAll tests completed!")


"""
Implementation Guidelines:
==========================

Phase 1: VaR Calculation (45 minutes)
- Start with historical VaR (simplest)
- Implement parametric VaR
- Add Monte Carlo VaR
- Calculate CVaR
- Compare all methods on same data
- Historical: no assumptions, but past-dependent
- Parametric: fast, assumes normality
- Monte Carlo: flexible, computationally intensive

Phase 2: Position Sizing (45 minutes)
- Implement Kelly criterion
- Add fixed fractional
- Implement volatility targeting
- Calculate risk parity weights
- Test with realistic trading scenarios
- Kelly often too aggressive - use fraction
- Fixed fractional is simple and effective
- Volatility targeting maintains constant risk

Phase 3: Portfolio Optimization (60 minutes)
- Calculate portfolio statistics
- Implement minimum variance
- Implement maximum Sharpe
- Generate efficient frontier
- Understand limitations (estimation error)
- Mean-variance sensitive to inputs
- Use robust estimation methods
- Consider transaction costs

Phase 4: Hedging (45 minutes)
- Calculate beta
- Implement beta hedging with futures
- Add CPPI for downside protection
- Test rebalancing logic
- Hedging reduces risk but costs money
- Dynamic hedging requires frequent rebalancing
- Consider transaction costs

Phase 5: Risk Monitoring (30 minutes)
- Set up limit framework
- Check position limits
- Monitor loss limits
- Calculate margin requirements
- Automated alerts crucial for live trading
- Hard limits prevent catastrophic losses

Tips:
- VaR has limitations - use multiple methods
- CVaR is better than VaR (coherent risk measure)
- Position sizing is critical - controls risk per trade
- Kelly criterion often too aggressive
- Diversification is free lunch (Markowitz)
- Rebalancing has costs - use thresholds
- Risk limits are last line of defense

Common Pitfalls:
- Relying only on VaR (doesn't capture tail risk)
- Using full Kelly (too aggressive)
- Ignoring estimation error in optimization
- Not accounting for transaction costs
- Over-concentrating in few positions
- Not monitoring risk in real-time
- Assuming normal distributions

Key Concepts:
- VaR: Maximum loss at confidence level
- CVaR: Expected loss beyond VaR
- Kelly Criterion: Optimal bet size for long-run growth
- Sharpe Ratio: Risk-adjusted return metric
- Beta: Systematic risk relative to market
- Efficient Frontier: Optimal risk-return combinations

Risk Management Rules:
- Never risk more than 1-2% per trade
- Diversify across uncorrelated assets
- Use stop losses religiously
- Monitor drawdowns constantly
- Have hard loss limits
- Size positions inversely to volatility
- Rebalance to maintain risk targets

Portfolio Optimization Insights:
- Mean-variance very sensitive to inputs
- Small changes in expected returns -> big weight changes
- Use robust estimation (shrinkage, resampling)
- Constraints improve practical performance
- Transaction costs matter a lot
- Out-of-sample performance often disappoints
- Consider Black-Litterman for stability

Hedging Considerations:
- Perfect hedge eliminates returns too
- Hedge ratio changes over time
- Futures hedging has roll costs
- Options expensive but provide asymmetry
- Dynamic hedging needs frequent rebalancing
- Consider basis risk (hedge != underlying)

Resources:
- "Active Portfolio Management" by Grinold & Kahn
- "The Black Swan" by Nassim Taleb (VaR limitations)
- "Fortune's Formula" by William Poundstone (Kelly)
- Modern Portfolio Theory papers by Markowitz
- Risk management frameworks (Basel, industry standards)
"""
