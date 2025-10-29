"""
Unit tests for Black-Scholes option pricing model.

This test suite validates:
- Pricing accuracy against known values
- Put-call parity
- Edge cases and boundary conditions
- Greeks calculations and bounds
- Input validation

Run with: pytest tests/test_black_scholes.py -v
"""

import pytest
import numpy as np
from core.black_scholes import BlackScholesModel, verify_put_call_parity


class TestBlackScholesBasicPricing:
    """Test basic option pricing functionality."""

    def setup_method(self):
        """Initialize model and standard test parameters."""
        self.model = BlackScholesModel()
        # Standard parameters
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.sigma = 0.2
        self.r = 0.05

    def test_atm_call_price(self):
        """Test at-the-money call option price."""
        price = self.model.call_price(self.S, self.K, self.T, self.sigma, self.r)

        # ATM call with these parameters should be around $10.45
        assert 10.0 < price < 11.0, f"ATM call price {price} outside expected range"

    def test_atm_put_price(self):
        """Test at-the-money put option price."""
        price = self.model.put_price(self.S, self.K, self.T, self.sigma, self.r)

        # ATM put with these parameters should be around $5.57
        assert 5.0 < price < 6.5, f"ATM put price {price} outside expected range"

    def test_deep_itm_call(self):
        """Test deep in-the-money call option."""
        # Strike = 50, Spot = 100 (deep ITM)
        price = self.model.call_price(self.S, 50.0, self.T, self.sigma, self.r)

        # Deep ITM call should be close to intrinsic value (S - K)
        intrinsic = self.S - 50.0
        assert price > intrinsic, "ITM call should be worth more than intrinsic value"
        assert price < intrinsic + 10, "ITM call shouldn't be too far above intrinsic"

    def test_deep_otm_call(self):
        """Test deep out-of-the-money call option."""
        # Strike = 150, Spot = 100 (deep OTM)
        price = self.model.call_price(self.S, 150.0, self.T, self.sigma, self.r)

        # Deep OTM call should be close to zero but positive
        assert 0 < price < 2, f"Deep OTM call price {price} should be near zero"

    def test_deep_itm_put(self):
        """Test deep in-the-money put option."""
        # Strike = 150, Spot = 100 (deep ITM for put)
        price = self.model.put_price(self.S, 150.0, self.T, self.sigma, self.r)

        # Deep ITM put should be close to present value of (K - S)
        intrinsic = 150.0 - self.S
        pv_intrinsic = intrinsic * np.exp(-self.r * self.T)
        assert price > intrinsic - 10, "ITM put should be valuable"
        assert price > pv_intrinsic * 0.9, "ITM put should be near PV of intrinsic"

    def test_deep_otm_put(self):
        """Test deep out-of-the-money put option."""
        # Strike = 50, Spot = 100 (deep OTM for put)
        price = self.model.put_price(self.S, 50.0, self.T, self.sigma, self.r)

        # Deep OTM put should be close to zero
        assert 0 < price < 0.5, f"Deep OTM put price {price} should be near zero"


class TestPutCallParity:
    """Test put-call parity relationship."""

    def setup_method(self):
        """Initialize model."""
        self.model = BlackScholesModel()

    @pytest.mark.parametrize("S,K,T,sigma,r", [
        (100, 100, 1.0, 0.2, 0.05),  # ATM
        (110, 100, 1.0, 0.2, 0.05),  # ITM call
        (90, 100, 1.0, 0.2, 0.05),   # OTM call
        (100, 100, 0.5, 0.3, 0.03),  # Different params
        (50, 60, 2.0, 0.4, 0.02),    # Different scale
    ])
    def test_put_call_parity(self, S, K, T, sigma, r):
        """Test put-call parity: C - P = S - K*e^(-rT)"""
        call = self.model.call_price(S, K, T, sigma, r)
        put = self.model.put_price(S, K, T, sigma, r)

        left = call - put
        right = S - K * np.exp(-r * T)

        assert abs(left - right) < 1e-10, \
            f"Put-call parity violated: {left} != {right}"

    def test_put_call_parity_helper(self):
        """Test the put-call parity verification helper function."""
        is_valid, diff = verify_put_call_parity(100, 100, 1.0, 0.2, 0.05)

        assert is_valid, f"Put-call parity should hold, difference: {diff}"
        assert diff < 1e-10, "Difference should be negligible"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Initialize model."""
        self.model = BlackScholesModel()

    def test_option_at_expiration_itm_call(self):
        """Test call option at expiration (T=0) when in-the-money."""
        S, K = 110, 100
        price = self.model.call_price(S, K, T=0, sigma=0.2, r=0.05)

        # At expiration, option worth exactly intrinsic value
        assert price == max(S - K, 0), \
            f"Call at expiration should equal intrinsic value: {S - K}"

    def test_option_at_expiration_otm_call(self):
        """Test call option at expiration (T=0) when out-of-the-money."""
        S, K = 90, 100
        price = self.model.call_price(S, K, T=0, sigma=0.2, r=0.05)

        assert price == 0, "OTM call at expiration should be worthless"

    def test_option_at_expiration_itm_put(self):
        """Test put option at expiration (T=0) when in-the-money."""
        S, K = 90, 100
        price = self.model.put_price(S, K, T=0, sigma=0.2, r=0.05)

        assert price == max(K - S, 0), \
            f"Put at expiration should equal intrinsic value: {K - S}"

    def test_option_at_expiration_otm_put(self):
        """Test put option at expiration (T=0) when out-of-the-money."""
        S, K = 110, 100
        price = self.model.put_price(S, K, T=0, sigma=0.2, r=0.05)

        assert price == 0, "OTM put at expiration should be worthless"

    def test_zero_volatility_itm_call(self):
        """Test call option with zero volatility (deterministic)."""
        # With σ=0, option should converge to discounted intrinsic value
        S, K = 110, 100
        price = self.model.call_price(S, K, T=1.0, sigma=1e-10, r=0.05)

        # Should be close to (S - K*e^(-rT))
        expected = S - K * np.exp(-0.05)
        assert abs(price - expected) < 0.01, \
            "With zero vol, ITM call should equal forward intrinsic value"

    def test_very_long_maturity(self):
        """Test option with very long time to maturity."""
        price = self.model.call_price(100, 100, T=10.0, sigma=0.2, r=0.05)

        # Long-dated ATM call should be valuable
        assert 20 < price < 100, \
            f"Long-dated call price {price} seems unreasonable"

    def test_high_volatility(self):
        """Test option with high volatility."""
        low_vol_price = self.model.call_price(100, 100, 1.0, sigma=0.1, r=0.05)
        high_vol_price = self.model.call_price(100, 100, 1.0, sigma=0.8, r=0.05)

        # Higher vol should mean higher option price
        assert high_vol_price > low_vol_price, \
            "Higher volatility should increase option value"


class TestGreeks:
    """Test Greeks calculations and properties."""

    def setup_method(self):
        """Initialize model and standard parameters."""
        self.model = BlackScholesModel()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.sigma = 0.2
        self.r = 0.05

    def test_call_delta_bounds(self):
        """Test that call delta is between 0 and 1."""
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')

        assert 0 <= greeks['delta'] <= 1, \
            f"Call delta {greeks['delta']} should be between 0 and 1"

    def test_put_delta_bounds(self):
        """Test that put delta is between -1 and 0."""
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'put')

        assert -1 <= greeks['delta'] <= 0, \
            f"Put delta {greeks['delta']} should be between -1 and 0"

    def test_atm_call_delta_near_half(self):
        """Test that ATM call delta is near 0.5."""
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')

        # ATM call delta should be around 0.5 (slightly higher due to drift)
        assert 0.45 < greeks['delta'] < 0.65, \
            f"ATM call delta {greeks['delta']} should be near 0.5"

    def test_gamma_positive(self):
        """Test that gamma is always positive."""
        # Test for various scenarios
        scenarios = [
            (100, 100, 1.0, 0.2, 0.05, 'call'),  # ATM call
            (100, 100, 1.0, 0.2, 0.05, 'put'),   # ATM put
            (110, 100, 1.0, 0.2, 0.05, 'call'),  # ITM call
            (90, 100, 1.0, 0.2, 0.05, 'put'),    # ITM put
        ]

        for S, K, T, sigma, r, opt_type in scenarios:
            greeks = self.model.greeks(S, K, T, sigma, r, opt_type)
            assert greeks['gamma'] >= 0, \
                f"Gamma should be positive, got {greeks['gamma']}"

    def test_gamma_symmetry(self):
        """Test that gamma is the same for calls and puts."""
        call_greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')
        put_greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'put')

        assert abs(call_greeks['gamma'] - put_greeks['gamma']) < 1e-10, \
            "Gamma should be identical for calls and puts"

    def test_vega_positive(self):
        """Test that vega is always positive."""
        call_greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')
        put_greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'put')

        assert call_greeks['vega'] > 0, "Call vega should be positive"
        assert put_greeks['vega'] > 0, "Put vega should be positive"

    def test_vega_symmetry(self):
        """Test that vega is the same for calls and puts."""
        call_greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')
        put_greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'put')

        assert abs(call_greeks['vega'] - put_greeks['vega']) < 1e-10, \
            "Vega should be identical for calls and puts"

    def test_call_theta_negative(self):
        """Test that call theta is typically negative (time decay)."""
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')

        # Theta should be negative for most options (time decay)
        assert greeks['theta'] < 0, \
            f"Call theta {greeks['theta']} should be negative (time decay)"

    def test_call_rho_positive(self):
        """Test that call rho is positive."""
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')

        # Call rho should be positive (benefit from higher rates)
        assert greeks['rho'] > 0, \
            "Call rho should be positive"

    def test_put_rho_negative(self):
        """Test that put rho is negative."""
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'put')

        # Put rho should be negative (hurt by higher rates)
        assert greeks['rho'] < 0, \
            "Put rho should be negative"

    def test_delta_numerical_accuracy(self):
        """Test delta by comparing with numerical derivative."""
        # Analytical delta
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')
        analytical_delta = greeks['delta']

        # Numerical delta using finite difference
        h = 0.01
        price_up = self.model.call_price(self.S + h, self.K, self.T, self.sigma, self.r)
        price_down = self.model.call_price(self.S - h, self.K, self.T, self.sigma, self.r)
        numerical_delta = (price_up - price_down) / (2 * h)

        # Should be very close
        assert abs(analytical_delta - numerical_delta) < 0.001, \
            f"Analytical delta {analytical_delta} differs from numerical {numerical_delta}"

    def test_gamma_numerical_accuracy(self):
        """Test gamma by comparing with numerical second derivative."""
        # Analytical gamma
        greeks = self.model.greeks(self.S, self.K, self.T, self.sigma, self.r, 'call')
        analytical_gamma = greeks['gamma']

        # Numerical gamma using finite difference
        h = 0.01
        price_up = self.model.call_price(self.S + h, self.K, self.T, self.sigma, self.r)
        price_center = self.model.call_price(self.S, self.K, self.T, self.sigma, self.r)
        price_down = self.model.call_price(self.S - h, self.K, self.T, self.sigma, self.r)
        numerical_gamma = (price_up - 2 * price_center + price_down) / (h ** 2)

        # Should be reasonably close (numerical methods have some error)
        assert abs(analytical_gamma - numerical_gamma) < 0.001, \
            f"Analytical gamma {analytical_gamma} differs from numerical {numerical_gamma}"


class TestInputValidation:
    """Test input validation and error handling."""

    def setup_method(self):
        """Initialize model."""
        self.model = BlackScholesModel()

    def test_negative_spot_price(self):
        """Test that negative spot price raises ValueError."""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            self.model.call_price(S=-100, K=100, T=1.0, sigma=0.2, r=0.05)

    def test_negative_strike_price(self):
        """Test that negative strike price raises ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            self.model.call_price(S=100, K=-100, T=1.0, sigma=0.2, r=0.05)

    def test_negative_time(self):
        """Test that negative time to maturity raises ValueError."""
        with pytest.raises(ValueError, match="Time to maturity must be non-negative"):
            self.model.call_price(S=100, K=100, T=-1.0, sigma=0.2, r=0.05)

    def test_negative_volatility(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="Volatility must be non-negative"):
            self.model.call_price(S=100, K=100, T=1.0, sigma=-0.2, r=0.05)

    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            self.model.greeks(100, 100, 1.0, 0.2, 0.05, 'invalid')


class TestImpliedVolatility:
    """Test implied volatility calculation."""

    def setup_method(self):
        """Initialize model and parameters."""
        self.model = BlackScholesModel()
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.sigma = 0.2
        self.r = 0.05

    def test_implied_vol_recovery(self):
        """Test that we can recover known volatility from price."""
        # Calculate option price with known vol
        true_vol = 0.25
        option_price = self.model.call_price(self.S, self.K, self.T, true_vol, self.r)

        # Recover implied vol
        implied_vol = self.model.implied_volatility(
            option_price, self.S, self.K, self.T, self.r, 'call'
        )

        # Should recover the original volatility
        assert abs(implied_vol - true_vol) < 0.001, \
            f"Implied vol {implied_vol} should match true vol {true_vol}"

    def test_implied_vol_put(self):
        """Test implied volatility for put options."""
        true_vol = 0.3
        option_price = self.model.put_price(self.S, self.K, self.T, true_vol, self.r)

        implied_vol = self.model.implied_volatility(
            option_price, self.S, self.K, self.T, self.r, 'put'
        )

        assert abs(implied_vol - true_vol) < 0.001, \
            f"Put implied vol {implied_vol} should match true vol {true_vol}"

    def test_implied_vol_invalid_price(self):
        """Test that invalid option price raises ValueError."""
        with pytest.raises(ValueError, match="Option price must be positive"):
            self.model.implied_volatility(-10, self.S, self.K, self.T, self.r, 'call')


class TestKnownValues:
    """Test against known option prices from literature/calculators."""

    def setup_method(self):
        """Initialize model."""
        self.model = BlackScholesModel()

    def test_hull_example_13_6(self):
        """
        Test against example from Hull's Options, Futures, and Other Derivatives.
        Example 13.6: S=42, K=40, T=0.5, r=0.1, sigma=0.2
        Expected call price: approximately 4.76
        """
        S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
        call_price = self.model.call_price(S, K, T, sigma, r)

        # Allow small tolerance for rounding
        expected = 4.76
        assert abs(call_price - expected) < 0.01, \
            f"Price {call_price:.2f} should match Hull example {expected}"

    def test_atm_call_standard_params(self):
        """
        Test ATM call with standard academic parameters.
        S=100, K=100, T=1, r=0.05, sigma=0.2
        Expected: approximately 10.45
        """
        call_price = self.model.call_price(100, 100, 1.0, 0.2, 0.05)
        expected = 10.45

        assert abs(call_price - expected) < 0.1, \
            f"ATM call {call_price:.2f} should be approximately {expected}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
