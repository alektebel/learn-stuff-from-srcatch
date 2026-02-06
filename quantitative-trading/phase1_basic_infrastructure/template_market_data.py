"""
Market Data Handler - Template
================================
This module handles fetching, storing, and processing market data for trading strategies.

TODO: Implement the following components:
1. Data fetcher for various sources (Yahoo Finance, Alpha Vantage)
2. Data normalization and cleaning
3. Real-time data streaming simulation
4. Historical data storage and retrieval
5. OHLCV bar aggregation from tick data

Learning objectives:
- Understand market data formats and conventions
- Handle API rate limits and errors
- Implement efficient data storage
- Process time series data for trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MarketDataHandler:
    """
    Handles market data fetching, storage, and retrieval.
    """
    
    def __init__(self, data_source: str = 'yahoo'):
        """
        Initialize the market data handler.
        
        Args:
            data_source: Data source to use ('yahoo', 'alpha_vantage', etc.)
        
        TODO 1: Initialize data source configuration
        - Set up API keys if needed
        - Configure default parameters (timeframe, etc.)
        - Initialize data cache dictionary
        """
        self.data_source = data_source
        self.cache = {}  # TODO: Implement caching mechanism
        # TODO: Add more initialization code
        pass
    
    def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1m', '5m', '1h', '1d', etc.)
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        
        TODO 2: Implement data fetching
        - Use yfinance or alpha_vantage to fetch data
        - Handle API errors and retries
        - Validate data quality (no missing values in critical periods)
        - Return standardized DataFrame format
        
        Example usage:
            handler = MarketDataHandler()
            data = handler.fetch_historical_data('AAPL', '2020-01-01', '2023-01-01')
        """
        # TODO: Implement historical data fetching
        pass
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """
        Get real-time (or latest available) data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with latest price, volume, timestamp
        
        TODO 3: Implement real-time data fetching
        - Fetch latest available data
        - Return standardized format with timestamp
        - Handle market closed scenarios
        """
        # TODO: Implement real-time data fetching
        pass
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and clean market data.
        
        Args:
            df: Raw market data DataFrame
        
        Returns:
            Cleaned and normalized DataFrame
        
        TODO 4: Implement data normalization
        - Handle missing values (forward fill for prices)
        - Adjust for stock splits and dividends
        - Ensure proper datetime indexing
        - Remove outliers if necessary
        - Validate data integrity
        """
        # TODO: Implement data normalization
        pass
    
    def aggregate_to_bars(
        self, 
        tick_data: pd.DataFrame, 
        interval: str = '5T'
    ) -> pd.DataFrame:
        """
        Aggregate tick data to OHLCV bars.
        
        Args:
            tick_data: DataFrame with tick-level data (timestamp, price, volume)
            interval: Aggregation interval (e.g., '5T' for 5 minutes)
        
        Returns:
            OHLCV bars DataFrame
        
        TODO 5: Implement bar aggregation
        - Resample tick data to specified interval
        - Calculate Open, High, Low, Close, Volume
        - Handle gaps in data
        """
        # TODO: Implement tick data aggregation
        pass
    
    def calculate_returns(
        self, 
        df: pd.DataFrame, 
        method: str = 'log'
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            df: DataFrame with price data
            method: 'log' for log returns, 'simple' for simple returns
        
        Returns:
            DataFrame with returns column added
        
        TODO 6: Implement return calculations
        - Calculate log returns: log(P_t / P_{t-1})
        - Calculate simple returns: (P_t - P_{t-1}) / P_{t-1}
        - Handle first row with NaN appropriately
        """
        # TODO: Implement return calculations
        pass
    
    def save_data(self, df: pd.DataFrame, symbol: str, filepath: str = None):
        """
        Save market data to disk.
        
        Args:
            df: DataFrame to save
            symbol: Stock symbol
            filepath: Optional custom file path
        
        TODO 7: Implement data persistence
        - Save to CSV or Parquet format
        - Create directory structure if needed
        - Handle overwrites appropriately
        """
        # TODO: Implement data saving
        pass
    
    def load_data(self, symbol: str, filepath: str = None) -> pd.DataFrame:
        """
        Load market data from disk.
        
        Args:
            symbol: Stock symbol
            filepath: Optional custom file path
        
        Returns:
            Loaded DataFrame
        
        TODO 8: Implement data loading
        - Load from CSV or Parquet
        - Validate data integrity
        - Return None if file doesn't exist
        """
        # TODO: Implement data loading
        pass


# Testing and example usage
if __name__ == "__main__":
    """
    TODO 9: Test your implementation
    
    Test cases to implement:
    1. Fetch historical data for a stock (e.g., AAPL)
    2. Verify data has correct columns and date range
    3. Test normalization function
    4. Test return calculations
    5. Test data saving and loading
    
    Example test workflow:
    """
    # Initialize handler
    handler = MarketDataHandler(data_source='yahoo')
    
    # Test 1: Fetch historical data
    print("Test 1: Fetching historical data...")
    # TODO: Implement test
    
    # Test 2: Normalize data
    print("Test 2: Normalizing data...")
    # TODO: Implement test
    
    # Test 3: Calculate returns
    print("Test 3: Calculating returns...")
    # TODO: Implement test
    
    # Test 4: Save and load data
    print("Test 4: Testing data persistence...")
    # TODO: Implement test
    
    print("\nAll tests completed!")


"""
Implementation Guidelines:
==========================

Phase 1: Basic Fetching (30 minutes)
- Start with yfinance for Yahoo Finance data
- Implement fetch_historical_data() first
- Test with a single symbol and date range
- Verify data structure and completeness

Phase 2: Data Quality (20 minutes)
- Implement normalize_data()
- Handle missing values with forward fill
- Add basic validation (price > 0, volume >= 0)
- Test with data containing gaps

Phase 3: Returns and Storage (20 minutes)
- Implement calculate_returns() for both log and simple returns
- Implement save_data() and load_data()
- Test round-trip (save and load)
- Use appropriate format (CSV for readability, Parquet for performance)

Phase 4: Real-time and Aggregation (30 minutes)
- Implement get_real_time_data()
- Implement aggregate_to_bars() for tick data
- Test bar aggregation with different intervals
- Handle edge cases (market closed, no data)

Tips:
- Use pandas resample() for bar aggregation
- Cache data to avoid repeated API calls
- Add logging for debugging
- Consider rate limits for API calls
- Validate data types and ranges

Common Pitfalls:
- Not handling timezone correctly
- Missing data causing calculation errors
- API rate limits causing failures
- Stock splits not accounted for
- Look-ahead bias in data processing
"""
