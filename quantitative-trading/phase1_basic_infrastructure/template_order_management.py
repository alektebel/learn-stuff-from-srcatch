"""
Order Management System - Template
===================================
This module handles order creation, execution simulation, and position tracking.

TODO: Implement the following components:
1. Order creation and validation
2. Paper trading execution simulator
3. Position tracking and portfolio management
4. Transaction cost modeling
5. Performance metrics calculation

Learning objectives:
- Understand order types and execution
- Implement position and P&L tracking
- Model realistic transaction costs
- Calculate trading performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side: buy or sell."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order:
    """
    Represents a trading order.
    
    TODO 1: Complete the Order class
    - Add validation in __init__
    - Implement order modification methods
    - Add order status tracking
    """
    
    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize an order.
        
        Args:
            symbol: Stock ticker symbol
            order_type: Type of order (market, limit, etc.)
            side: Buy or sell
            quantity: Number of shares
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            timestamp: Order creation time
        
        TODO: Implement order validation
        - Validate quantity > 0
        - Validate price > 0 for limit orders
        - Validate stop_price for stop orders
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.timestamp = timestamp or datetime.now()
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.filled_price = 0.0
        
        # TODO: Add validation
        pass
    
    def __repr__(self):
        return (f"Order({self.symbol}, {self.side.value}, "
                f"{self.order_type.value}, qty={self.quantity}, "
                f"status={self.status.value})")


class Position:
    """
    Represents a position in a security.
    
    TODO 2: Complete the Position class
    - Implement position update logic
    - Calculate realized and unrealized P&L
    - Track average entry price
    """
    
    def __init__(self, symbol: str):
        """
        Initialize a position.
        
        Args:
            symbol: Stock ticker symbol
        """
        self.symbol = symbol
        self.quantity = 0
        self.average_price = 0.0
        self.realized_pnl = 0.0
        
    def update(self, quantity_change: int, price: float):
        """
        Update position after a trade.
        
        Args:
            quantity_change: Change in position (positive for buy, negative for sell)
            price: Execution price
        
        TODO 3: Implement position update logic
        - Update quantity
        - Calculate new average price for buys
        - Calculate realized P&L for sells
        - Handle position flips (long to short or vice versa)
        """
        # TODO: Implement position update
        pass
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L.
        
        Args:
            current_price: Current market price
        
        Returns:
            Unrealized P&L
        
        TODO 4: Implement unrealized P&L calculation
        - Calculate as: (current_price - average_price) * quantity
        """
        # TODO: Implement unrealized P&L
        pass
    
    def __repr__(self):
        return f"Position({self.symbol}, qty={self.quantity}, avg_price={self.average_price:.2f})"


class Portfolio:
    """
    Manages multiple positions and tracks portfolio value.
    
    TODO 5: Complete the Portfolio class
    - Track positions across multiple symbols
    - Calculate total portfolio value
    - Generate portfolio statistics
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash balance
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        
    def get_position(self, symbol: str) -> Position:
        """
        Get position for a symbol.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Position object
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def execute_order(self, order: Order, execution_price: float, commission: float = 0.0):
        """
        Execute an order and update portfolio.
        
        Args:
            order: Order to execute
            execution_price: Price at which order is executed
            commission: Transaction commission
        
        TODO 6: Implement order execution
        - Update position
        - Update cash balance
        - Subtract commission
        - Record trade in history
        - Validate sufficient cash for buys
        """
        # TODO: Implement order execution
        pass
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Dictionary of symbol -> current price
        
        Returns:
            Total portfolio value (cash + positions)
        
        TODO 7: Implement portfolio valuation
        - Sum cash balance
        - Add value of all positions at current prices
        """
        # TODO: Implement portfolio valuation
        pass
    
    def get_portfolio_stats(self, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate portfolio statistics.
        
        Args:
            current_prices: Dictionary of symbol -> current price
        
        Returns:
            Dictionary with portfolio statistics
        
        TODO 8: Implement statistics calculation
        - Total value
        - Total return (%)
        - Total P&L
        - Number of positions
        - Largest position
        """
        # TODO: Implement portfolio statistics
        pass


class ExecutionSimulator:
    """
    Simulates order execution for backtesting.
    
    TODO 9: Complete the ExecutionSimulator class
    - Implement realistic execution logic
    - Model slippage and market impact
    - Handle different order types
    """
    
    def __init__(
        self,
        slippage_pct: float = 0.001,  # 0.1% slippage
        commission_pct: float = 0.0005  # 0.05% commission
    ):
        """
        Initialize execution simulator.
        
        Args:
            slippage_pct: Slippage as percentage of price
            commission_pct: Commission as percentage of trade value
        """
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
    
    def simulate_execution(
        self,
        order: Order,
        market_data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Simulate order execution.
        
        Args:
            order: Order to execute
            market_data: Market data at execution time
        
        Returns:
            Execution details (price, commission) or None if can't execute
        
        TODO 10: Implement execution simulation
        - For MARKET orders: execute at current price + slippage
        - For LIMIT orders: check if price reached
        - For STOP orders: check if stop triggered
        - Calculate commission
        - Add realistic constraints (liquidity, etc.)
        """
        # TODO: Implement execution simulation
        pass
    
    def calculate_slippage(self, order: Order, base_price: float) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            order: Order object
            base_price: Base price before slippage
        
        Returns:
            Execution price after slippage
        
        TODO 11: Implement slippage calculation
        - Add slippage in direction of trade (worse execution)
        - Consider order size impact
        - Return adjusted price
        """
        # TODO: Implement slippage calculation
        pass
    
    def calculate_commission(self, order: Order, execution_price: float) -> float:
        """
        Calculate commission for an order.
        
        Args:
            order: Order object
            execution_price: Execution price
        
        Returns:
            Commission amount
        
        TODO 12: Implement commission calculation
        - Calculate as percentage of trade value
        - Add minimum commission if applicable
        - Consider volume discounts
        """
        # TODO: Implement commission calculation
        pass


# Testing and example usage
if __name__ == "__main__":
    """
    TODO 13: Test your implementation
    
    Test cases to implement:
    1. Create and validate orders
    2. Test position updates (buy, sell, flip)
    3. Test portfolio with multiple positions
    4. Test execution simulation
    5. Calculate portfolio statistics
    """
    
    print("Test 1: Creating orders...")
    # TODO: Create sample orders and validate
    
    print("\nTest 2: Managing positions...")
    # TODO: Test position updates
    
    print("\nTest 3: Portfolio management...")
    # TODO: Test portfolio operations
    
    print("\nTest 4: Execution simulation...")
    # TODO: Test order execution
    
    print("\nAll tests completed!")


"""
Implementation Guidelines:
==========================

Phase 1: Order and Position Classes (30 minutes)
- Implement Order class with validation
- Implement Position class with P&L tracking
- Test with simple buy/sell scenarios
- Verify average price calculation

Phase 2: Portfolio Management (30 minutes)
- Implement Portfolio class
- Add position tracking for multiple symbols
- Implement cash management
- Test with multiple trades

Phase 3: Execution Simulation (40 minutes)
- Implement ExecutionSimulator class
- Model slippage and commissions
- Handle different order types
- Test execution logic with market data

Phase 4: Statistics and Reporting (20 minutes)
- Implement portfolio statistics
- Add trade history tracking
- Calculate returns and P&L
- Generate summary reports

Tips:
- Start with market orders, then add limit/stop
- Keep track of all transactions for audit trail
- Use proper data types (Decimal for money in production)
- Add logging for debugging
- Validate all inputs

Common Pitfalls:
- Not handling position flips correctly
- Forgetting to subtract commissions
- Average price calculation when adding to position
- Insufficient cash validation
- Slippage direction (should worsen execution)
- Not tracking realized vs unrealized P&L
"""
