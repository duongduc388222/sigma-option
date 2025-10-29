"""
Black-Scholes Option Pricing Model

This module implements the Black-Scholes-Merton model for pricing European options
and calculating the Greeks (sensitivity measures).

Mathematical Background:
    The Black-Scholes formula assumes:
    - European-style exercise (only at expiration)
    - No dividends
    - Constant volatility and risk-free rate
    - Log-normal distribution of stock prices
    - No transaction costs or taxes
    - Continuous trading

Author: Duong Hong Duc
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


class BlackScholesModel:
    """
    Black-Scholes option pricing model with Greeks calculation.

    This class provides methods to price European call and put options
    and compute all first-order Greeks (Delta, Gamma, Vega, Theta, Rho).
    """

    def __init__(self):
        """Initialize the Black-Scholes model."""
        pass

    @staticmethod
    def _validate_inputs(S: float, K: float, T: float, sigma: float, r: float) -> None:
        """
        Validate input parameters for option pricing.

        Args:
            S: Current stock price (spot price)
            K: Strike price
            T: Time to maturity in years
            sigma: Volatility (annualized standard deviation)
            r: Risk-free interest rate (annualized)

        Raises:
            ValueError: If any input parameter is invalid
        """
        if S <= 0:
            raise ValueError(f"Spot price must be positive, got S={S}")
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got K={K}")
        if T < 0:
            raise ValueError(f"Time to maturity must be non-negative, got T={T}")
        if sigma < 0:
            raise ValueError(f"Volatility must be non-negative, got sigma={sigma}")
        if r < -1:
            raise ValueError(f"Risk-free rate must be greater than -100%, got r={r}")

    @staticmethod
    def _d1(S: float, K: float, T: float, sigma: float, r: float) -> float:
        """
        Calculate d1 parameter in Black-Scholes formula.

        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity in years
            sigma: Volatility
            r: Risk-free rate

        Returns:
            d1 value
        """
        if T == 0:
            # At expiration, return a large value indicating deep ITM/OTM
            return np.inf if S > K else -np.inf

        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def _d2(S: float, K: float, T: float, sigma: float, r: float) -> float:
        """
        Calculate d2 parameter in Black-Scholes formula.

        d2 = d1 - σ√T

        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity in years
            sigma: Volatility
            r: Risk-free rate

        Returns:
            d2 value
        """
        if T == 0:
            return np.inf if S > K else -np.inf

        d1 = BlackScholesModel._d1(S, K, T, sigma, r)
        return d1 - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, sigma: float, r: float) -> float:
        """
        Calculate European call option price using Black-Scholes formula.

        C = S₀N(d₁) - Ke^(-rT)N(d₂)

        Args:
            S: Current stock price (spot price)
            K: Strike price
            T: Time to maturity in years
            sigma: Volatility (annualized)
            r: Risk-free interest rate (annualized)

        Returns:
            Call option price

        Example:
            >>> model = BlackScholesModel()
            >>> price = model.call_price(S=100, K=100, T=1.0, sigma=0.2, r=0.05)
            >>> print(f"Call price: ${price:.2f}")
        """
        BlackScholesModel._validate_inputs(S, K, T, sigma, r)

        # Handle special case: at expiration
        if T == 0:
            return max(S - K, 0)

        d1 = BlackScholesModel._d1(S, K, T, sigma, r)
        d2 = BlackScholesModel._d2(S, K, T, sigma, r)

        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

    @staticmethod
    def put_price(S: float, K: float, T: float, sigma: float, r: float) -> float:
        """
        Calculate European put option price using Black-Scholes formula.

        P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

        Args:
            S: Current stock price (spot price)
            K: Strike price
            T: Time to maturity in years
            sigma: Volatility (annualized)
            r: Risk-free interest rate (annualized)

        Returns:
            Put option price

        Example:
            >>> model = BlackScholesModel()
            >>> price = model.put_price(S=100, K=100, T=1.0, sigma=0.2, r=0.05)
            >>> print(f"Put price: ${price:.2f}")
        """
        BlackScholesModel._validate_inputs(S, K, T, sigma, r)

        # Handle special case: at expiration
        if T == 0:
            return max(K - S, 0)

        d1 = BlackScholesModel._d1(S, K, T, sigma, r)
        d2 = BlackScholesModel._d2(S, K, T, sigma, r)

        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put

    @staticmethod
    def greeks(S: float, K: float, T: float, sigma: float, r: float,
               option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for a European option.

        Greeks are sensitivity measures that describe how option prices
        change with respect to various parameters.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity in years
            sigma: Volatility
            r: Risk-free rate
            option_type: Type of option ('call' or 'put')

        Returns:
            Dictionary containing:
                - delta: ∂V/∂S (sensitivity to stock price)
                - gamma: ∂²V/∂S² (sensitivity of delta to stock price)
                - vega: ∂V/∂σ (sensitivity to volatility)
                - theta: ∂V/∂T (sensitivity to time decay)
                - rho: ∂V/∂r (sensitivity to interest rate)

        Raises:
            ValueError: If option_type is not 'call' or 'put'

        Example:
            >>> model = BlackScholesModel()
            >>> greeks = model.greeks(S=100, K=100, T=1.0, sigma=0.2, r=0.05, option_type='call')
            >>> print(f"Delta: {greeks['delta']:.4f}")
        """
        BlackScholesModel._validate_inputs(S, K, T, sigma, r)

        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        # Handle special case: at expiration
        if T == 0:
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0

            return {
                'delta': delta,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }

        d1 = BlackScholesModel._d1(S, K, T, sigma, r)
        d2 = BlackScholesModel._d2(S, K, T, sigma, r)

        sqrt_T = np.sqrt(T)

        # Standard normal PDF at d1
        nd1 = norm.pdf(d1)

        # Delta: ∂V/∂S
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1

        # Gamma: ∂²V/∂S² (same for calls and puts)
        gamma = nd1 / (S * sigma * sqrt_T)

        # Vega: ∂V/∂σ (same for calls and puts)
        # Note: Vega is often expressed per 1% change in volatility
        vega = S * nd1 * sqrt_T / 100  # Divided by 100 for 1% vol change

        # Theta: ∂V/∂t (negative of ∂V/∂T)
        # Expressed as daily time decay
        if option_type == 'call':
            theta = (-(S * nd1 * sigma) / (2 * sqrt_T)
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # put
            theta = (-(S * nd1 * sigma) / (2 * sqrt_T)
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        # Rho: ∂V/∂r
        # Expressed per 1% change in interest rate
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    @staticmethod
    def implied_volatility(option_price: float, S: float, K: float, T: float,
                          r: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Implied volatility is the volatility value that, when input into the
        Black-Scholes formula, yields the observed market price.

        Args:
            option_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to maturity in years
            r: Risk-free rate
            option_type: Type of option ('call' or 'put')
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility (annualized)

        Raises:
            ValueError: If convergence fails or inputs are invalid
        """
        BlackScholesModel._validate_inputs(S, K, T, 0.01, r)

        if option_price <= 0:
            raise ValueError(f"Option price must be positive, got {option_price}")

        # Initial guess: use Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * (option_price / S)

        for i in range(max_iterations):
            # Calculate option price with current sigma
            if option_type == 'call':
                price = BlackScholesModel.call_price(S, K, T, sigma, r)
            else:
                price = BlackScholesModel.put_price(S, K, T, sigma, r)

            # Calculate vega (∂V/∂σ)
            greeks = BlackScholesModel.greeks(S, K, T, sigma, r, option_type)
            vega_raw = greeks['vega'] * 100  # Convert back to raw vega

            # Price difference
            diff = price - option_price

            # Check convergence
            if abs(diff) < tolerance:
                return sigma

            # Newton-Raphson update
            if vega_raw == 0:
                raise ValueError("Vega is zero, cannot compute implied volatility")

            sigma = sigma - diff / vega_raw

            # Ensure sigma stays positive
            if sigma <= 0:
                sigma = 0.01

        raise ValueError(f"Implied volatility did not converge after {max_iterations} iterations")


def verify_put_call_parity(S: float, K: float, T: float, sigma: float, r: float) -> Tuple[bool, float]:
    """
    Verify put-call parity relationship.

    Put-call parity states: C - P = S - K*e^(-rT)

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        sigma: Volatility
        r: Risk-free rate

    Returns:
        Tuple of (is_valid, difference) where is_valid is True if parity holds
        within numerical tolerance, and difference is the actual difference

    Example:
        >>> is_valid, diff = verify_put_call_parity(100, 100, 1.0, 0.2, 0.05)
        >>> print(f"Parity valid: {is_valid}, Difference: {diff:.10f}")
    """
    model = BlackScholesModel()

    call = model.call_price(S, K, T, sigma, r)
    put = model.put_price(S, K, T, sigma, r)

    left_side = call - put
    right_side = S - K * np.exp(-r * T)

    difference = abs(left_side - right_side)
    is_valid = difference < 1e-10

    return is_valid, difference
