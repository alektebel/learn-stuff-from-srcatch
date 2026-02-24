"""
Advanced Market Making Models - Template
==========================================
Implements four advanced models used in quantitative market making:
  1. Avellaneda-Stoikov adapted for binary settlement
  2. GLFT (Guéant-Lehalle-Fernandez-Tapia) inventory bounds
  3. Glosten-Milgrom adverse selection
  4. VPIN kill switch

Each class contains TODO sections where you should fill in the implementation.
See solutions/phase4_rl_microstructure/advanced_market_making.py for reference answers.

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

    In the original Avellaneda-Stoikov (2008) framework the mid-price follows
    Brownian motion with constant volatility σ.  When the underlying is a
    *binary* contract (e.g. prediction-market YES token, binary option at
    expiry) the price is bounded in [0, 1] and the residual variance shrinks
    as t → T.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient (γ > 0).
    sigma : float
        Baseline volatility of the mid-price (annualised if T is in years).
    kappa : float
        Order-arrival intensity parameter κ (from the AS Poisson model).
    T : float
        Time to settlement (same units as the time steps passed to `quotes`).
    """

    def __init__(self, gamma: float, sigma: float, kappa: float, T: float):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T

    # ------------------------------------------------------------------
    # Helper: residual variance for a binary contract
    # ------------------------------------------------------------------

    def residual_variance(self, mid: float, t: float) -> float:
        """
        Return the instantaneous variance of the mid-price at time *t* given
        the current mid-price *mid* ∈ (0, 1).

        For a binary contract the terminal value is Bernoulli(p) where p ≈ mid.
        The residual variance at time t can be approximated as:

            Var_t ≈ mid * (1 - mid) * f(t, T)

        where f(t, T) is a time-decay factor that equals 1 at t=0 and 0 at t=T.
        One common choice is f(t, T) = (T - t) / T.

        You may also blend with the constant-sigma term from the original AS
        model using a mixing parameter.

        TODO: Implement the residual variance formula for a binary asset.
              Your formula must:
              - Return 0 when mid = 0 or mid = 1 (settled contract).
              - Return 0 when t >= T (at or past settlement).
              - Decrease monotonically as t approaches T.

        Parameters
        ----------
        mid : float
            Current mid-price in (0, 1).
        t : float
            Current time (0 ≤ t ≤ T).

        Returns
        -------
        float
            Residual variance σ²(mid, t).
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core model outputs
    # ------------------------------------------------------------------

    def reservation_price(self, mid: float, inventory: float, t: float) -> float:
        """
        Compute the market-maker's *reservation price* (indifference price).

        The AS reservation price is:
            r(s, q, t) = s - q · γ · σ²(s, t) · (T - t)

        where σ²(s, t) is the residual variance from `residual_variance`.

        TODO: Implement the reservation price formula.

        Parameters
        ----------
        mid : float
            Current mid-price s.
        inventory : float
            Current signed inventory q (positive = long).
        t : float
            Current time.

        Returns
        -------
        float
            Reservation price r.
        """
        # TODO: implement
        raise NotImplementedError

    def optimal_spread(self, mid: float, t: float) -> float:
        """
        Compute the optimal total bid-ask spread δ* = δ_ask + δ_bid.

        The AS closed-form spread is:
            δ*(t) = γ · σ²(s, t) · (T - t) + (2/γ) · ln(1 + γ/κ)

        TODO: Implement the optimal spread formula.
              Handle the edge case where (T - t) ≤ 0 by returning a minimal
              spread (e.g. 1e-8).

        Parameters
        ----------
        mid : float
            Current mid-price.
        t : float
            Current time.

        Returns
        -------
        float
            Total optimal spread δ*.
        """
        # TODO: implement
        raise NotImplementedError

    def quotes(
        self, mid: float, inventory: float, t: float
    ) -> Tuple[float, float]:
        """
        Return optimal (bid, ask) quotes.

        Using the reservation price r and optimal spread δ*:
            bid = r - δ*/2
            ask = r + δ*/2

        Both prices are clipped to the valid range (0, 1).

        TODO: Implement using `reservation_price` and `optimal_spread`.

        Returns
        -------
        Tuple[float, float]
            (bid, ask) prices clipped to (0, 1).
        """
        # TODO: implement
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. GLFT inventory bounds
# ---------------------------------------------------------------------------

class GLFTInventoryBounds:
    """
    Guéant-Lehalle-Fernandez-Tapia (2013) closed-form market-making solution
    with explicit inventory bounds Q_max / Q_min.

    GLFT extend AS with a closed-form solution and show that the market maker
    should stop quoting on one side once the inventory hits the bound
    Q_max = ⌊1/(γ·σ²·Δt) + 1/2⌋  (symmetric, so Q_min = -Q_max).

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Asset volatility.
    kappa : float
        Order-arrival intensity.
    T : float
        Horizon (time to end of trading session).
    dt : float
        Discrete time step used in simulations.
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

    def inventory_bounds(self) -> Tuple[int, int]:
        """
        Return the GLFT inventory bounds (Q_min, Q_max).

        The maximum inventory bound from GLFT is:
            Q_max = floor(1 / (γ · σ² · dt) + 0.5)
        and Q_min = -Q_max.

        Edge case: if the formula yields 0, return (0, 0) meaning the
        market maker cannot carry any inventory.

        TODO: Implement the GLFT inventory bound formula.

        Returns
        -------
        Tuple[int, int]
            (Q_min, Q_max) as integers.
        """
        # TODO: implement
        raise NotImplementedError

    def skewed_quotes(
        self,
        mid: float,
        inventory: int,
        t: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Return GLFT bid/ask quotes with one-sided quoting at inventory bounds.

        Rules:
        - If inventory >= Q_max: only post the ask (bid = None, stop buying).
        - If inventory <= Q_min: only post the bid (ask = None, stop selling).
        - Otherwise: post both sides using the standard AS formula adapted with
          the GLFT closed-form spread.

        The GLFT spread for the symmetric case is:
            δ_glft = (1/γ) · ln(1 + γ/κ) + (1/2) · γ · σ² · (T - t)

        Half-spread is applied symmetrically around mid.

        TODO: Implement the GLFT quoting logic.

        Parameters
        ----------
        mid : float
            Current mid-price.
        inventory : int
            Current signed inventory.
        t : float
            Current time.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            (bid, ask) where None means "do not quote that side".
        """
        # TODO: implement
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Glosten-Milgrom adverse selection
# ---------------------------------------------------------------------------

class GlostenMilgrom:
    """
    Glosten-Milgrom (1985) model of bid-ask spread arising from adverse selection.

    In this model the market-maker faces two types of traders:
      - Informed traders who know the true asset value V and trade profitably.
      - Uninformed (noise) traders who trade for exogenous reasons.

    The market-maker sets prices so that each trade is zero-profit in expectation.

    Parameters
    ----------
    alpha : float
        Probability that the next trader is *informed* (0 < α < 1).
    v_low : float
        Low fundamental value (V_L).
    v_high : float
        High fundamental value (V_H, V_H > V_L).
    prob_high : float
        Prior probability that the true value is V_H (default 0.5).
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
        """Expected fundamental value μ = P(V_H) · V_H + P(V_L) · V_L."""
        return self.prob_high * self.v_high + (1 - self.prob_high) * self.v_low

    def ask(self) -> float:
        """
        Compute the zero-profit ask price.

        The ask is set so that:
            E[V | buy order arrives] = ask

        Given:
          - Informed traders buy only when V = V_H (P(buy | informed, V=V_H) = 1;
            P(buy | informed, V=V_L) = 0), so the marginal P(buy | informed) = prob_high.
          - Uninformed traders buy with prob 0.5 regardless of value

        By Bayes' rule:
            P(buy) = α · prob_high + (1-α) · 0.5

        The conditional expected value given a buy order:
            E[V | buy] = [α · prob_high · V_H + (1-α) · 0.5 · (prob_high · V_H + (1-prob_high) · V_L)]
                         / P(buy)

        TODO: Implement the zero-profit ask formula.

        Returns
        -------
        float
            Ask price.
        """
        # TODO: implement
        raise NotImplementedError

    def bid(self) -> float:
        """
        Compute the zero-profit bid price.

        The bid is set so that:
            E[V | sell order arrives] = bid

        Given:
          - Informed traders sell iff V = V_L
          - Uninformed traders sell with prob 0.5

        TODO: Implement the zero-profit bid formula (symmetric to `ask`).

        Returns
        -------
        float
            Bid price.
        """
        # TODO: implement
        raise NotImplementedError

    def spread(self) -> float:
        """
        Return the Glosten-Milgrom bid-ask spread = ask - bid.

        TODO: Use `ask()` and `bid()`.

        Returns
        -------
        float
            Spread > 0.
        """
        # TODO: implement
        raise NotImplementedError

    def update_belief(self, order_side: str) -> "GlostenMilgrom":
        """
        Return a *new* GlostenMilgrom instance with the posterior belief
        prob_high updated after observing a buy or sell order (Bayesian update).

        After a *buy* order:
            P(V_H | buy) = P(buy | V_H) · P(V_H) / P(buy)
            P(buy | V_H) = α · 1 + (1-α) · 0.5   (informed buy when V=V_H)
            P(buy | V_L) = α · 0 + (1-α) · 0.5   (informed never buy when V=V_L)

        After a *sell* order: apply the symmetric calculation.

        TODO: Implement the Bayesian update for both "buy" and "sell".
              Raise ValueError for any other order_side string.

        Parameters
        ----------
        order_side : str
            "buy" or "sell".

        Returns
        -------
        GlostenMilgrom
            New instance with updated prob_high.
        """
        # TODO: implement
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. VPIN kill switch
# ---------------------------------------------------------------------------

class VPINKillSwitch:
    """
    Volume-synchronized Probability of Informed Trading (VPIN) kill switch.

    VPIN (Easley, de Prado & O'Hara 2012) estimates the fraction of volume
    that is order-flow imbalance, used as a real-time proxy for toxic flow.

    Algorithm
    ---------
    1. Accumulate trades until a *volume bucket* of size V_bucket is full.
    2. For each bucket classify volume into buy-initiated V_b and
       sell-initiated V_s using bulk classification (price change sign or
       tick rule).
    3. VPIN = (1/n) · Σ_{i=1}^{n} |V_b_i - V_s_i| / V_bucket
       where n is the rolling window of buckets.
    4. If VPIN ≥ threshold → trigger kill switch (halt quoting).

    Parameters
    ----------
    bucket_size : float
        Target volume for each bucket V_bucket.
    window : int
        Number of buckets in the rolling average (n).
    threshold : float
        VPIN level above which trading is halted (e.g. 0.7).
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

        # Internal state — do not modify these names (used in tests)
        self._buckets: List[float] = []          # imbalance ratios per bucket
        self._current_buy_vol: float = 0.0
        self._current_sell_vol: float = 0.0
        self._current_vol: float = 0.0
        self._kill_switch_active: bool = False

    # ------------------------------------------------------------------

    def _classify_side(self, price_change: float, volume: float) -> Tuple[float, float]:
        """
        Classify a trade into buy/sell volume using the *tick rule*:
          - price_change > 0  → buy-initiated
          - price_change < 0  → sell-initiated
          - price_change == 0 → split 50/50

        TODO: Implement tick-rule classification.

        Returns
        -------
        Tuple[float, float]
            (buy_volume, sell_volume) for this trade.
        """
        # TODO: implement
        raise NotImplementedError

    def process_trade(self, price_change: float, volume: float) -> bool:
        """
        Ingest one trade and update the VPIN estimate.

        Steps:
        1. Classify the trade using `_classify_side`.
        2. Add volume to the current bucket accumulator.
        3. When the current bucket is full (accumulated volume ≥ bucket_size):
           a. Compute imbalance ratio = |V_b - V_s| / V_bucket.
           b. Append to `self._buckets` (keep at most `self.window` entries).
           c. Reset the current-bucket accumulators.
           d. Update `self._kill_switch_active` via `check_kill_switch`.
        4. Return the current kill-switch status.

        A single trade may fill *more than one* bucket. Handle this by
        splitting the trade volume proportionally across buckets.

        TODO: Implement the bucket-filling and VPIN update logic.

        Parameters
        ----------
        price_change : float
            Price difference from previous trade (used for tick rule).
        volume : float
            Trade volume (positive).

        Returns
        -------
        bool
            True if the kill switch is currently active.
        """
        # TODO: implement
        raise NotImplementedError

    def current_vpin(self) -> float:
        """
        Return the current VPIN estimate.

        VPIN = mean of imbalance ratios in `self._buckets`.
        Return 0.0 if no buckets have been completed yet.

        TODO: Implement.
        """
        # TODO: implement
        raise NotImplementedError

    def check_kill_switch(self) -> bool:
        """
        Return True if the current VPIN ≥ threshold.

        Also update `self._kill_switch_active`.

        TODO: Implement.
        """
        # TODO: implement
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset all internal state (called when trading is resumed after a halt).

        TODO: Reset all accumulators and kill-switch flag.
        """
        # TODO: implement
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Quick smoke-test (not a substitute for full unit tests)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Advanced Market Making Template — fill in the TODO sections.")
    print("See solutions/phase4_rl_microstructure/advanced_market_making.py")
    print("for reference implementations.")
