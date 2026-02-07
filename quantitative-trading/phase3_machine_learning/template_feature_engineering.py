"""
Feature Engineering for Trading - Template
===========================================
This module implements feature engineering techniques for machine learning trading strategies.

TODO: Implement the following components:
1. Technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Feature selection techniques (correlation, mutual information, PCA)
3. Time-based features (day of week, month, time of day)
4. Lag features and rolling statistics
5. Feature transformation and scaling

Learning objectives:
- Build comprehensive technical indicators
- Engineer predictive features
- Select relevant features
- Transform and normalize features
- Handle temporal aspects of financial data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Calculates various technical indicators.
    
    TODO 1: Implement technical indicators
    - Momentum indicators (RSI, Stochastic)
    - Trend indicators (MACD, ADX)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV, VWAP)
    """
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        pass
    
    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
        
        Returns:
            RSI series (0-100)
        
        TODO 2: Implement RSI
        - Calculate price changes
        - Separate gains and losses
        - Calculate average gain and average loss
        - RS = average_gain / average_loss
        - RSI = 100 - (100 / (1 + RS))
        - RSI > 70: overbought, RSI < 30: oversold
        
        Example:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        """
        # TODO: Calculate RSI
        pass
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        
        Returns:
            DataFrame with MACD, signal, and histogram
        
        TODO 3: Implement MACD
        - Calculate fast EMA and slow EMA
        - MACD line = fast_ema - slow_ema
        - Signal line = EMA of MACD line
        - Histogram = MACD - Signal
        - Crossovers generate trading signals
        
        Example:
            fast_ema = prices.ewm(span=fast_period).mean()
            slow_ema = prices.ewm(span=slow_period).mean()
            macd = fast_ema - slow_ema
            signal = macd.ewm(span=signal_period).mean()
            histogram = macd - signal
        """
        # TODO: Calculate MACD
        pass
    
    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period
            num_std: Number of standard deviations
        
        Returns:
            DataFrame with middle, upper, and lower bands
        
        TODO 4: Implement Bollinger Bands
        - Middle band = SMA
        - Upper band = SMA + (std * num_std)
        - Lower band = SMA - (std * num_std)
        - Bands expand/contract with volatility
        - Price touching bands indicates extremes
        """
        # TODO: Calculate Bollinger Bands
        pass
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
        
        Returns:
            ATR series
        
        TODO 5: Implement ATR
        - True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        - ATR = EMA of True Range
        - Measures volatility
        - Used for stop loss placement
        
        Example:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=period).mean()
        """
        # TODO: Calculate ATR
        pass
    
    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
        
        Returns:
            DataFrame with %K and %D
        
        TODO 6: Implement Stochastic
        - %K = (Close - Low_n) / (High_n - Low_n) * 100
        - %D = 3-period SMA of %K
        - Range: 0-100
        - >80 overbought, <20 oversold
        """
        # TODO: Calculate Stochastic
        pass
    
    def calculate_obv(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            close: Close prices
            volume: Volume
        
        Returns:
            OBV series
        
        TODO 7: Implement OBV
        - If close > prev_close: OBV += volume
        - If close < prev_close: OBV -= volume
        - If close == prev_close: OBV unchanged
        - Confirms price trends with volume
        """
        # TODO: Calculate OBV
        pass
    
    def calculate_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
        
        Returns:
            VWAP series
        
        TODO 8: Implement VWAP
        - Typical Price = (High + Low + Close) / 3
        - VWAP = Sum(Typical Price * Volume) / Sum(Volume)
        - Often calculated intraday
        - Institutional benchmark
        """
        # TODO: Calculate VWAP
        pass


class FeatureEngineer:
    """
    Engineers features for machine learning models.
    
    TODO 9: Implement feature engineering
    - Lag features
    - Rolling statistics
    - Time-based features
    - Price transformations
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.technical = TechnicalIndicators()
        pass
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            df: Input DataFrame
            columns: Columns to lag
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features added
        
        TODO 10: Create lag features
        - For each column and lag, create column_lag_N
        - Essential for time series ML
        - Captures temporal dependencies
        - Be careful not to look ahead!
        
        Example:
            for col in columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        """
        # TODO: Create lag features
        pass
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        column: str,
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            df: Input DataFrame
            column: Column to compute statistics on
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling features added
        
        TODO 11: Create rolling features
        - Mean, std, min, max for each window
        - Skewness and kurtosis
        - Quantiles (25th, 75th percentile)
        - Useful for capturing recent behavior
        
        Example:
            for window in windows:
                df[f'{column}_mean_{window}'] = df[column].rolling(window).mean()
                df[f'{column}_std_{window}'] = df[column].rolling(window).std()
                df[f'{column}_min_{window}'] = df[column].rolling(window).min()
                df[f'{column}_max_{window}'] = df[column].rolling(window).max()
        """
        # TODO: Create rolling features
        pass
    
    def create_time_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with datetime index
        
        Returns:
            DataFrame with time features added
        
        TODO 12: Create time features
        - Day of week (Monday=0, Friday=4)
        - Month
        - Quarter
        - Week of year
        - Day of month
        - Hour/minute for intraday
        - Capture seasonality and patterns
        
        Example:
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end
            df['is_month_start'] = df.index.is_month_start
        """
        # TODO: Create time features
        pass
    
    def create_price_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with price features added
        
        TODO 13: Create price features
        - Returns (log and simple)
        - Price ranges (high-low, close-open)
        - Price momentum (ROC)
        - Price relative to moving averages
        - Volume changes
        
        Example:
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
            df['close_open_pct'] = (df['Close'] - df['Open']) / df['Open']
        """
        # TODO: Create price features
        pass
    
    def create_technical_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add all technical indicators as features.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators added
        
        TODO 14: Add technical indicators
        - RSI
        - MACD
        - Bollinger Bands
        - ATR
        - Stochastic
        - OBV
        - Any other indicators from TechnicalIndicators
        """
        # TODO: Add technical indicators
        pass
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples
        
        Returns:
            DataFrame with interaction features
        
        TODO 15: Create interactions
        - Multiply features together
        - Ratios of features
        - Can capture non-linear relationships
        - Use sparingly to avoid overfitting
        
        Example:
            for feat1, feat2 in feature_pairs:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                df[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2]
        """
        # TODO: Create interaction features
        pass


class FeatureSelector:
    """
    Selects relevant features for modeling.
    
    TODO 16: Implement feature selection
    - Correlation-based selection
    - Mutual information
    - Recursive feature elimination
    - PCA for dimensionality reduction
    """
    
    def __init__(self):
        """Initialize feature selector."""
        pass
    
    def select_by_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.05
    ) -> List[str]:
        """
        Select features by correlation with target.
        
        Args:
            X: Feature DataFrame
            y: Target series
            threshold: Minimum absolute correlation
        
        Returns:
            List of selected feature names
        
        TODO 17: Select by correlation
        - Calculate correlation with target
        - Keep features with |correlation| > threshold
        - Simple and interpretable
        - May miss non-linear relationships
        
        Example:
            correlations = X.corrwith(y).abs()
            selected = correlations[correlations > threshold].index.tolist()
        """
        # TODO: Select by correlation
        pass
    
    def select_by_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10
    ) -> List[str]:
        """
        Select features by mutual information.
        
        Args:
            X: Feature DataFrame
            y: Target series
            k: Number of features to select
        
        Returns:
            List of selected feature names
        
        TODO 18: Select by mutual information
        - Measures dependency between features and target
        - Captures non-linear relationships
        - Better than correlation for complex relationships
        
        Example:
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y)
            mi_scores = pd.Series(mi_scores, index=X.columns)
            selected = mi_scores.nlargest(k).index.tolist()
        """
        # TODO: Select by mutual information
        pass
    
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
        
        Returns:
            List of features to keep
        
        TODO 19: Remove redundant features
        - Calculate correlation matrix
        - If two features highly correlated, drop one
        - Reduces multicollinearity
        - Improves model stability
        
        Example:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            return [col for col in X.columns if col not in to_drop]
        """
        # TODO: Remove correlated features
        pass
    
    def apply_pca(
        self,
        X: pd.DataFrame,
        n_components: int = None,
        variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature DataFrame
            n_components: Number of components (if None, use variance_threshold)
            variance_threshold: Cumulative variance to preserve
        
        Returns:
            Tuple of (transformed data, fitted PCA object)
        
        TODO 20: Apply PCA
        - Standardize features first
        - Apply PCA
        - Select components that explain variance_threshold of variance
        - Loss of interpretability but reduces dimensions
        
        Example:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if n_components is None:
                pca = PCA(n_components=variance_threshold)
            else:
                pca = PCA(n_components=n_components)
            
            X_pca = pca.fit_transform(X_scaled)
        """
        # TODO: Apply PCA
        pass


class FeatureTransformer:
    """
    Transforms and scales features.
    
    TODO 21: Implement feature transformation
    - Standard scaling
    - Robust scaling (resistant to outliers)
    - Log transformation
    - Rank transformation
    """
    
    def __init__(self):
        """Initialize feature transformer."""
        self.scalers = {}
        pass
    
    def standard_scale(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply standard scaling (z-score normalization).
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit scaler (True for train, False for test)
        
        Returns:
            Scaled DataFrame
        
        TODO 22: Implement standard scaling
        - Scale to mean=0, std=1
        - Assumes normal distribution
        - Store scaler for consistent test set scaling
        
        Example:
            from sklearn.preprocessing import StandardScaler
            
            if fit:
                self.scalers['standard'] = StandardScaler()
                X_scaled = self.scalers['standard'].fit_transform(X)
            else:
                X_scaled = self.scalers['standard'].transform(X)
            
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        """
        # TODO: Implement standard scaling
        pass
    
    def robust_scale(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply robust scaling using median and IQR.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit scaler
        
        Returns:
            Scaled DataFrame
        
        TODO 23: Implement robust scaling
        - Uses median and IQR instead of mean and std
        - Resistant to outliers
        - Better for financial data with fat tails
        
        Example:
            from sklearn.preprocessing import RobustScaler
            
            if fit:
                self.scalers['robust'] = RobustScaler()
                X_scaled = self.scalers['robust'].fit_transform(X)
            else:
                X_scaled = self.scalers['robust'].transform(X)
        """
        # TODO: Implement robust scaling
        pass
    
    def log_transform(
        self,
        X: pd.DataFrame,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply log transformation.
        
        Args:
            X: Feature DataFrame
            columns: Columns to transform (if None, all)
        
        Returns:
            Transformed DataFrame
        
        TODO 24: Implement log transformation
        - Use log(1 + x) for stability
        - Good for skewed features
        - Only for positive features
        - Reduces impact of extreme values
        """
        # TODO: Implement log transformation
        pass
    
    def rank_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply rank transformation.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Rank-transformed DataFrame
        
        TODO 25: Implement rank transformation
        - Convert values to ranks
        - Very robust to outliers
        - Loses magnitude information
        - Good for tree-based models
        
        Example:
            X_ranked = X.rank(pct=True)  # Percentile ranks
        """
        # TODO: Implement rank transformation
        pass


# Testing and example usage
if __name__ == "__main__":
    """
    TODO 26: Test your implementation
    
    Test cases to implement:
    1. Calculate technical indicators
    2. Engineer features from OHLCV data
    3. Select relevant features
    4. Transform and scale features
    5. Create complete feature pipeline
    """
    
    print("Test 1: Technical Indicators...")
    # TODO: Load sample data and calculate indicators
    # - RSI, MACD, Bollinger Bands, ATR
    # - Verify values make sense
    # - Plot indicators
    
    print("\nTest 2: Feature Engineering...")
    # TODO: Engineer features
    # - Lag features
    # - Rolling statistics
    # - Time features
    # - Technical indicators
    # - Verify no look-ahead bias
    
    print("\nTest 3: Feature Selection...")
    # TODO: Select features
    # - Calculate correlations
    # - Use mutual information
    # - Remove redundant features
    # - Apply PCA
    # - Compare feature sets
    
    print("\nTest 4: Feature Transformation...")
    # TODO: Transform features
    # - Standard scaling
    # - Robust scaling
    # - Log transformation
    # - Compare distributions
    
    print("\nTest 5: Complete Pipeline...")
    # TODO: Build complete pipeline
    # - Load data
    # - Engineer features
    # - Select features
    # - Transform features
    # - Verify output ready for ML
    
    print("\nAll tests completed!")


"""
Implementation Guidelines:
==========================

Phase 1: Technical Indicators (60 minutes)
- Implement RSI, MACD, Bollinger Bands
- Add ATR, Stochastic, OBV
- Test on real data
- Verify against known values (e.g., TradingView)
- Plot indicators to check correctness

Phase 2: Feature Engineering (90 minutes)
- Create lag features (1, 2, 3, 5, 10 days)
- Add rolling statistics (5, 10, 20, 60 days)
- Time features (day of week, month, quarter)
- Price transformations (returns, ranges)
- Combine technical indicators
- Total features can be 50-200

Phase 3: Feature Selection (60 minutes)
- Start with correlation analysis
- Apply mutual information
- Remove highly correlated features (>0.95)
- Try PCA for comparison
- Reduce to 10-30 most important features
- Balance: more features = more data needed

Phase 4: Feature Transformation (45 minutes)
- Standard scaling for linear models
- Robust scaling for data with outliers
- Log transformation for skewed features
- Rank transformation for tree models
- Always fit on train, transform test

Phase 5: Complete Pipeline (45 minutes)
- Combine all components
- Test on multiple stocks
- Verify no data leakage
- Save pipeline for reuse
- Document feature meanings

Tips:
- Always avoid look-ahead bias (use shift())
- More features != better (curse of dimensionality)
- Feature engineering is iterative
- Domain knowledge helps immensely
- Document what each feature means
- Monitor feature importance over time

Common Pitfalls:
- Using future data (look-ahead bias)
- Not standardizing features
- Too many correlated features
- Overfitting with too many features
- Not handling missing values
- Fitting scaler on test data
- Using close-to-close returns (misses gaps)

Feature Engineering Best Practices:
- Start simple, add complexity gradually
- Test each feature's predictive power
- Remove redundant features
- Use domain knowledge
- Create features that make economic sense
- Consider transaction costs when engineering signals

Technical Indicator Tips:
- RSI: 14-period is standard
- MACD: 12/26/9 is standard
- Bollinger Bands: 20-period, 2 std
- ATR: useful for position sizing
- Combine multiple indicators for confirmation

Feature Selection Guidelines:
- Correlation >0.9 with target: keep
- Correlation >0.95 between features: remove one
- Mutual information: more robust than correlation
- PCA: loses interpretability but reduces dimensions
- Start with 20-50 features, reduce to 10-20

Scaling Importance:
- Critical for: SVM, Neural Networks, Linear Models
- Not needed for: Tree-based models (RF, XGBoost)
- Standard scaler: assumes normal distribution
- Robust scaler: better for financial data
- Always fit on train only!

Time-Based Features:
- Day of week: Monday/Friday effects
- Month: seasonality patterns
- Quarter: earnings seasons
- Hour: intraday patterns
- Be careful: markets change, patterns fade

Resources:
- "Evidence-Based Technical Analysis" by Aronson
- Technical Analysis library (TA-Lib)
- Scikit-learn feature selection documentation
- "Feature Engineering for Machine Learning" by Zheng & Casari
"""
