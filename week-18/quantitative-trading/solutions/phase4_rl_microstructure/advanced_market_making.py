"""
Advanced Market Making Models - Solution
==========================================
Complete reference implementations of four advanced quantitative
market-making models:

  1. Avellaneda-Stoikov adapted for binary settlement
  2. GLFT (Guéant-Lehalle-Fernandez-Tapia) inventory bounds
  3. Glosten-Milgrom adverse selection
  4. VPIN kill switch

References
----------
- Avellaneda & Stoikov (2008). "High-frequency trading in a limit order book."
  Quantitative Finance, 8(3), 217-224.
- Guéant, Lehalle & Fernandez-Tapia (2013). "Dealing with the inventory risk:
  a solution to the market making problem." Mathematics and Financial Economics.
- Glosten & Milgrom (1985). "Bid, ask and transaction prices in a specialist
  market with heterogeneously informed traders." Journal of Financial Economics.
- Easley, de Prado & O'Hara (2012). "Flow toxicity and liquidity in a high
  frequency world." Review of Financial Studies.

⚠️ Educational purposes only. Do NOT use with real money without extensive testing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Avellaneda-Stoikov model adapted for binary settlement
# ---------------------------------------------------------------------------

class AvellanedaStoikovBinary:
    """
    Market-making optimal quotes for an asset that settles to {0, 1} at time T.

    Modifications vs. the vanilla AS model:
      - σ²(s, t) = s · (1-s) · (T-t)/T  (Bernoulli residual variance).
      - Quotes are clipped to the valid price range (0, 1).

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient (γ > 0).
    sigma : float
        Baseline volatility scaling factor (unused in the binary-only formula
        but kept for API compatibility with the continuous-volatility variant).
    kappa : float
        Order-arrival intensity κ.
    T : float
        Time to settlement.
    """

    def __init__(self, gamma: float, sigma: float, kappa: float, T: float):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T

    # ------------------------------------------------------------------

    def residual_variance(self, mid: float, t: float) -> float:
        """
        Residual variance for a binary contract:

            σ²(s, t) = s · (1-s) · max(T-t, 0) / T

        - At t = 0 this equals the Bernoulli variance  s(1-s).
        - At t ≥ T this equals 0 (contract has settled).
        - At mid = 0 or 1 this equals 0 (no uncertainty left).
        """
        time_left = max(self.T - t, 0.0)
        if self.T <= 0:
            return 0.0
        return mid * (1.0 - mid) * time_left / self.T

    # ------------------------------------------------------------------

    def reservation_price(self, mid: float, inventory: float, t: float) -> float:
        """
        AS reservation price:

            r = s - q · γ · σ²(s, t) · (T - t)

        For binary contracts, σ²(s,t) = s(1-s)(T-t)/T already incorporates
        one factor of (T-t), so the full inventory-adjustment term has a
        quadratic time-decay of the form (T-t)²/T.
        """
        var = self.residual_variance(mid, t)
        time_left = max(self.T - t, 0.0)
        return mid - inventory * self.gamma * var * time_left

    def optimal_spread(self, mid: float, t: float) -> float:
        """
        AS closed-form optimal spread:

            δ* = γ · σ²(s, t) · (T-t)  +  (2/γ) · ln(1 + γ/κ)

        Returns a minimal spread (1e-8) when past settlement.
        """
        time_left = max(self.T - t, 0.0)
        if time_left <= 0:
            return 1e-8
        var = self.residual_variance(mid, t)
        inventory_term = self.gamma * var * time_left
        liquidity_term = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return inventory_term + liquidity_term

    def quotes(
        self, mid: float, inventory: float, t: float
    ) -> Tuple[float, float]:
        """Return (bid, ask) clipped to (0, 1)."""
        r = self.reservation_price(mid, inventory, t)
        half_spread = self.optimal_spread(mid, t) / 2.0
        bid = np.clip(r - half_spread, 0.0, 1.0)
        ask = np.clip(r + half_spread, 0.0, 1.0)
        return float(bid), float(ask)


# ---------------------------------------------------------------------------
# 2. GLFT inventory bounds
# ---------------------------------------------------------------------------

class GLFTInventoryBounds:
    """
    GLFT (2013) closed-form market-making solution with explicit inventory bounds.

    The inventory bound derived from the GLFT value function is:

        Q_max = floor( 1 / (γ · σ² · dt)  +  0.5 )

    The market maker stops quoting on one side once this bound is reached.
    """

    def __init__(
        self,
        gamma: float,
        sigma: float,
        kappa: float,
        T: float,
        dt: float = 1.0,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T
        self.dt = dt

    # ------------------------------------------------------------------

    def inventory_bounds(self) -> Tuple[int, int]:
        """
        GLFT inventory bounds:

            Q_max = floor(1 / (γ · σ² · dt) + 0.5)
            Q_min = -Q_max

        Returns (0, 0) if γ · σ² · dt = 0 (degenerate case).
        """
        denom = self.gamma * (self.sigma ** 2) * self.dt
        if denom <= 0:
            return (0, 0)
        q_max = int(np.floor(1.0 / denom + 0.5))
        return (-q_max, q_max)

    # ------------------------------------------------------------------

    def skewed_quotes(
        self,
        mid: float,
        inventory: int,
        t: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        GLFT quotes with one-sided quoting at the inventory bounds.

        GLFT spread:
            δ_glft = (1/γ) · ln(1 + γ/κ)  +  (1/2) · γ · σ² · (T-t)

        Returns
        -------
        (bid, ask) — either value may be None if that side is suppressed.
        """
        q_min, q_max = self.inventory_bounds()
        time_left = max(self.T - t, 0.0)

        half_spread = (
            (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
            + 0.5 * self.gamma * (self.sigma ** 2) * time_left
        ) / 2.0

        bid: Optional[float] = mid - half_spread
        ask: Optional[float] = mid + half_spread

        if inventory >= q_max:
            bid = None   # already long max — stop buying
        if inventory <= q_min:
            ask = None   # already short max — stop selling

        return bid, ask


# ---------------------------------------------------------------------------
# 3. Glosten-Milgrom adverse selection
# ---------------------------------------------------------------------------

class GlostenMilgrom:
    """
    Glosten-Milgrom (1985) competitive-equilibrium spread model.

    The market-maker sets prices so that each trade is zero-profit in
    expectation when facing a mixture of informed and uninformed traders.
    """

    def __init__(
        self,
        alpha: float,
        v_low: float,
        v_high: float,
        prob_high: float = 0.5,
    ):
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        assert v_high > v_low, "v_high must exceed v_low"
        assert 0 < prob_high < 1, "prob_high must be in (0, 1)"
        self.alpha = alpha
        self.v_low = v_low
        self.v_high = v_high
        self.prob_high = prob_high

    @property
    def fundamental_value(self) -> float:
        """E[V] = P(V_H)·V_H + P(V_L)·V_L."""
        return self.prob_high * self.v_high + (1.0 - self.prob_high) * self.v_low

    # ------------------------------------------------------------------
    # Zero-profit pricing
    # ------------------------------------------------------------------

    def ask(self) -> float:
        """
        Zero-profit ask = E[V | buy order].

        P(buy | V=V_H) = α·1 + (1-α)·0.5  = α + (1-α)/2
        P(buy | V=V_L) = α·0 + (1-α)·0.5  = (1-α)/2

        E[V | buy] = [P(buy|V_H)·P(V_H)·V_H + P(buy|V_L)·P(V_L)·V_L]
                     / [P(buy|V_H)·P(V_H)  +  P(buy|V_L)·P(V_L)]
        """
        p_h = self.prob_high
        p_l = 1.0 - p_h
        buy_if_high = self.alpha + (1.0 - self.alpha) * 0.5   # α + (1-α)/2
        buy_if_low  = (1.0 - self.alpha) * 0.5                # (1-α)/2

        numerator   = buy_if_high * p_h * self.v_high + buy_if_low * p_l * self.v_low
        denominator = buy_if_high * p_h               + buy_if_low * p_l
        return numerator / denominator

    def bid(self) -> float:
        """
        Zero-profit bid = E[V | sell order].

        P(sell | V=V_L) = α·1 + (1-α)·0.5
        P(sell | V=V_H) = α·0 + (1-α)·0.5
        """
        p_h = self.prob_high
        p_l = 1.0 - p_h
        sell_if_low  = self.alpha + (1.0 - self.alpha) * 0.5
        sell_if_high = (1.0 - self.alpha) * 0.5

        numerator   = sell_if_high * p_h * self.v_high + sell_if_low * p_l * self.v_low
        denominator = sell_if_high * p_h               + sell_if_low * p_l
        return numerator / denominator

    def spread(self) -> float:
        """Bid-ask spread = ask - bid."""
        return self.ask() - self.bid()

    # ------------------------------------------------------------------
    # Bayesian belief update
    # ------------------------------------------------------------------

    def update_belief(self, order_side: str) -> "GlostenMilgrom":
        """
        Return a new GlostenMilgrom instance with the posterior prob_high
        after observing a buy or sell order.

        After a *buy*:
            P(V_H | buy) ∝ P(buy | V_H) · P(V_H)
        """
        p_h = self.prob_high
        p_l = 1.0 - p_h

        if order_side == "buy":
            lh_high = self.alpha + (1.0 - self.alpha) * 0.5   # P(buy|V_H)
            lh_low  = (1.0 - self.alpha) * 0.5                # P(buy|V_L)
        elif order_side == "sell":
            lh_high = (1.0 - self.alpha) * 0.5                # P(sell|V_H)
            lh_low  = self.alpha + (1.0 - self.alpha) * 0.5   # P(sell|V_L)
        else:
            raise ValueError(f"order_side must be 'buy' or 'sell', got {order_side!r}")

        posterior_h = lh_high * p_h / (lh_high * p_h + lh_low * p_l)
        return GlostenMilgrom(
            alpha=self.alpha,
            v_low=self.v_low,
            v_high=self.v_high,
            prob_high=posterior_h,
        )


# ---------------------------------------------------------------------------
# 4. VPIN kill switch
# ---------------------------------------------------------------------------

class VPINKillSwitch:
    """
    VPIN (Easley, de Prado & O'Hara 2012) kill switch.

    Accumulates trades into volume buckets, estimates the fraction of
    order-flow imbalance (VPIN), and activates a kill switch when VPIN
    exceeds a threshold.
    """

    def __init__(
        self,
        bucket_size: float,
        window: int = 50,
        threshold: float = 0.7,
    ):
        self.bucket_size = bucket_size
        self.window = window
        self.threshold = threshold

        self._buckets: List[float] = []
        self._current_buy_vol: float = 0.0
        self._current_sell_vol: float = 0.0
        self._current_vol: float = 0.0
        self._kill_switch_active: bool = False

    # ------------------------------------------------------------------

    def _classify_side(self, price_change: float, volume: float) -> Tuple[float, float]:
        """
        Tick-rule classification:
          - price_change > 0 → all volume is buy-initiated
          - price_change < 0 → all volume is sell-initiated
          - price_change == 0 → split 50 / 50
        """
        if price_change > 0:
            return volume, 0.0
        elif price_change < 0:
            return 0.0, volume
        else:
            half = volume / 2.0
            return half, half

    # ------------------------------------------------------------------

    def process_trade(self, price_change: float, volume: float) -> bool:
        """
        Ingest one trade, filling buckets as needed, and return kill-switch status.

        A trade whose volume spans multiple buckets is split proportionally.
        """
        buy_vol, sell_vol = self._classify_side(price_change, volume)
        remaining = volume
        buy_ratio  = buy_vol  / volume if volume > 0 else 0.5
        sell_ratio = sell_vol / volume if volume > 0 else 0.5

        while remaining > 0:
            space_in_bucket = self.bucket_size - self._current_vol
            fill = min(remaining, space_in_bucket)

            self._current_buy_vol  += fill * buy_ratio
            self._current_sell_vol += fill * sell_ratio
            self._current_vol      += fill
            remaining              -= fill

            if self._current_vol >= self.bucket_size:
                # Bucket complete — record imbalance ratio
                imbalance = (
                    abs(self._current_buy_vol - self._current_sell_vol)
                    / self.bucket_size
                )
                self._buckets.append(imbalance)
                if len(self._buckets) > self.window:
                    self._buckets.pop(0)

                # Reset accumulators
                self._current_buy_vol  = 0.0
                self._current_sell_vol = 0.0
                self._current_vol      = 0.0

                self.check_kill_switch()

        return self._kill_switch_active

    # ------------------------------------------------------------------

    def current_vpin(self) -> float:
        """VPIN = mean of completed-bucket imbalance ratios (0.0 if none)."""
        if not self._buckets:
            return 0.0
        return float(np.mean(self._buckets))

    def check_kill_switch(self) -> bool:
        """Activate kill switch if VPIN ≥ threshold."""
        self._kill_switch_active = self.current_vpin() >= self.threshold
        return self._kill_switch_active

    def reset(self) -> None:
        """Reset all state (e.g. after a trading halt)."""
        self._buckets.clear()
        self._current_buy_vol  = 0.0
        self._current_sell_vol = 0.0
        self._current_vol      = 0.0
        self._kill_switch_active = False


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _demo():
    print("=" * 60)
    print("1. Avellaneda-Stoikov Binary")
    print("=" * 60)
    # kappa=100 gives a tight liquidity term suitable for [0,1] prices
    as_model = AvellanedaStoikovBinary(gamma=0.5, sigma=0.2, kappa=100.0, T=1.0)
    for q in [-2, 0, 2]:
        bid, ask = as_model.quotes(mid=0.55, inventory=q, t=0.5)
        print(f"  inventory={q:+d}  bid={bid:.4f}  ask={ask:.4f}")

    print()
    print("=" * 60)
    print("2. GLFT Inventory Bounds")
    print("=" * 60)
    glft = GLFTInventoryBounds(gamma=0.1, sigma=0.2, kappa=1.5, T=1.0, dt=1.0)
    q_min, q_max = glft.inventory_bounds()
    print(f"  Bounds: Q_min={q_min}, Q_max={q_max}")
    for q in [q_min, 0, q_max]:
        b, a = glft.skewed_quotes(mid=100.0, inventory=q, t=0.5)
        bid_str = f"{b:.4f}" if b is not None else "None"
        ask_str = f"{a:.4f}" if a is not None else "None"
        print(f"  inventory={q:+d}  bid={bid_str}  ask={ask_str}")

    print()
    print("=" * 60)
    print("3. Glosten-Milgrom")
    print("=" * 60)
    gm = GlostenMilgrom(alpha=0.3, v_low=98.0, v_high=102.0)
    print(f"  Fundamental value: {gm.fundamental_value:.2f}")
    print(f"  Ask: {gm.ask():.4f}  Bid: {gm.bid():.4f}  Spread: {gm.spread():.4f}")
    gm_after_buy = gm.update_belief("buy")
    print(f"  After buy: prob_high={gm_after_buy.prob_high:.4f}  "
          f"Spread={gm_after_buy.spread():.4f}")

    print()
    print("=" * 60)
    print("4. VPIN Kill Switch")
    print("=" * 60)
    # Each bucket = 1000 units; trades are 100 units so 10 trades per bucket
    vpin = VPINKillSwitch(bucket_size=1000.0, window=5, threshold=0.6)
    # Toxic flow: 10 buckets each filled with all-buy trades
    for _ in range(100):
        vpin.process_trade(price_change=1.0, volume=100.0)
    print(f"  VPIN after one-sided flow: {vpin.current_vpin():.3f}  "
          f"Kill switch: {vpin._kill_switch_active}")
    vpin.reset()
    # Balanced flow: alternating buy/sell within each bucket
    for i in range(100):
        side = 1.0 if i % 2 == 0 else -1.0
        vpin.process_trade(price_change=side, volume=100.0)
    print(f"  VPIN after balanced flow:  {vpin.current_vpin():.3f}  "
          f"Kill switch: {vpin._kill_switch_active}")


if __name__ == "__main__":
    _demo()
