"""
Visualization module for Black-Scholes option pricing analysis.

This module provides professional heatmap visualizations for:
- Price sensitivity analysis (volatility vs spot price)
- P&L scenario analysis (profit/loss at expiration)

Author: Duong Hong Duc
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
from matplotlib.figure import Figure
from core.black_scholes import BlackScholesModel


class OptionVisualizer:
    """
    Professional visualization tools for option pricing analysis.

    This class creates publication-quality heatmaps for analyzing
    option prices and profit/loss scenarios.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use (default: 'seaborn-v0_8-darkgrid')
        """
        self.model = BlackScholesModel()
        # Set up plotting style
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            plt.style.use('default')

        # Professional color schemes
        # For prices: reversed RdYlGn (green=high, red=low)
        self.price_cmap = 'RdYlGn'
        # For P&L: normal RdYlGn (green=profit, red=loss)
        self.pnl_cmap = 'RdYlGn'

    def plot_price_heatmap(self,
                          base_params: Dict,
                          spot_range: Optional[Tuple[float, float]] = None,
                          vol_range: Optional[Tuple[float, float]] = None,
                          spot_points: int = 20,
                          vol_points: int = 20,
                          figsize: Tuple[float, float] = (10, 8)) -> Figure:
        """
        Create a heatmap showing option price sensitivity to spot price and volatility.

        This visualization helps understand how option prices change across
        different market conditions.

        Args:
            base_params: Dictionary with keys:
                - strike_price (float): Strike price K
                - time_to_maturity (float): Time to expiration T
                - risk_free_rate (float): Risk-free rate r
                - option_type (str): 'call' or 'put'
                - spot_price (float): Current spot (used for range if not specified)
                - volatility (float): Current vol (used for range if not specified)
            spot_range: (min, max) spot prices, defaults to ±30% of base spot
            vol_range: (min, max) volatilities, defaults to ±50% of base vol
            spot_points: Number of spot price grid points
            vol_points: Number of volatility grid points
            figsize: Figure size in inches

        Returns:
            Matplotlib Figure object

        Example:
            >>> params = {
            ...     'spot_price': 100,
            ...     'strike_price': 100,
            ...     'time_to_maturity': 1.0,
            ...     'volatility': 0.2,
            ...     'risk_free_rate': 0.05,
            ...     'option_type': 'call'
            ... }
            >>> viz = OptionVisualizer()
            >>> fig = viz.plot_price_heatmap(params)
            >>> fig.savefig('price_heatmap.png')
        """
        # Extract base parameters
        K = base_params['strike_price']
        T = base_params['time_to_maturity']
        r = base_params['risk_free_rate']
        option_type = base_params['option_type']

        # Determine ranges
        if spot_range is None:
            S_base = base_params.get('spot_price', K)
            spot_range = (S_base * 0.7, S_base * 1.3)

        if vol_range is None:
            vol_base = base_params.get('volatility', 0.2)
            vol_range = (vol_base * 0.5, vol_base * 1.5)

        # Create meshgrid
        spot_prices = np.linspace(spot_range[0], spot_range[1], spot_points)
        volatilities = np.linspace(vol_range[0], vol_range[1], vol_points)
        S_grid, vol_grid = np.meshgrid(spot_prices, volatilities)

        # Calculate option prices
        prices = np.zeros_like(S_grid)
        for i in range(vol_points):
            for j in range(spot_points):
                S = S_grid[i, j]
                sigma = vol_grid[i, j]

                if option_type.lower() == 'call':
                    prices[i, j] = self.model.call_price(S, K, T, sigma, r)
                else:
                    prices[i, j] = self.model.put_price(S, K, T, sigma, r)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create discrete cell-based heatmap using pcolormesh
        # Reverse colormap for prices (green=high, red=low)
        im = ax.pcolormesh(S_grid, vol_grid * 100, prices,
                          cmap=self.price_cmap + '_r',
                          shading='auto', edgecolors='white', linewidth=0.5)

        # Add text annotations showing values in each cell
        for i in range(vol_points):
            for j in range(spot_points):
                text_value = prices[i, j]
                # Position text at center of cell
                text_x = spot_prices[j]
                text_y = volatilities[i] * 100

                # Determine text color based on background (for readability)
                # Normalize price to 0-1 range for color determination
                norm_val = (text_value - np.min(prices)) / (np.max(prices) - np.min(prices))
                text_color = 'white' if norm_val < 0.5 else 'black'

                ax.text(text_x, text_y, f'{text_value:.2f}',
                       ha='center', va='center', fontsize=7,
                       color=text_color, fontweight='bold')

        # Mark current spot and volatility if provided
        if 'spot_price' in base_params and 'volatility' in base_params:
            current_S = base_params['spot_price']
            current_vol = base_params['volatility'] * 100
            ax.plot(current_S, current_vol, 'r*', markersize=15,
                   label='Current Position', markeredgecolor='yellow',
                   markeredgewidth=2)
            ax.legend(loc='upper left')

        # Labels and title
        ax.set_xlabel('Spot Price ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')

        option_label = option_type.capitalize()
        title = f'{option_label} Option Price Sensitivity\n' \
                f'Strike: ${K:.2f}, Maturity: {T:.2f}Y, Rate: {r*100:.1f}%'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Option Price ($)', fontsize=11, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    def plot_pnl_heatmap(self,
                        entry_price: float,
                        base_params: Dict,
                        spot_range: Optional[Tuple[float, float]] = None,
                        vol_range: Optional[Tuple[float, float]] = None,
                        spot_points: int = 20,
                        vol_points: int = 20,
                        figsize: Tuple[float, float] = (10, 8),
                        show_percentage: bool = False) -> Figure:
        """
        Create a P&L heatmap showing profit/loss scenarios at expiration.

        This visualization helps assess potential outcomes and risk/reward
        across different spot price and volatility scenarios.

        Args:
            entry_price: Premium paid/received when entering the position
            base_params: Dictionary with option parameters (same as plot_price_heatmap)
            spot_range: (min, max) spot prices at expiration
            vol_range: (min, max) volatilities (affects price before expiration)
            spot_points: Number of spot price grid points
            vol_points: Number of volatility grid points
            figsize: Figure size in inches
            show_percentage: If True, show P&L as percentage of entry price

        Returns:
            Matplotlib Figure object

        Example:
            >>> params = {
            ...     'spot_price': 100,
            ...     'strike_price': 100,
            ...     'time_to_maturity': 1.0,
            ...     'volatility': 0.2,
            ...     'risk_free_rate': 0.05,
            ...     'option_type': 'call'
            ... }
            >>> entry = 10.45  # Premium paid for the call
            >>> viz = OptionVisualizer()
            >>> fig = viz.plot_pnl_heatmap(entry, params)
            >>> fig.savefig('pnl_heatmap.png')
        """
        # Extract base parameters
        K = base_params['strike_price']
        T = base_params['time_to_maturity']
        r = base_params['risk_free_rate']
        option_type = base_params['option_type']

        # Determine ranges
        if spot_range is None:
            S_base = base_params.get('spot_price', K)
            spot_range = (S_base * 0.7, S_base * 1.3)

        if vol_range is None:
            vol_base = base_params.get('volatility', 0.2)
            vol_range = (vol_base * 0.5, vol_base * 1.5)

        # Create meshgrid
        spot_prices = np.linspace(spot_range[0], spot_range[1], spot_points)
        volatilities = np.linspace(vol_range[0], vol_range[1], vol_points)
        S_grid, vol_grid = np.meshgrid(spot_prices, volatilities)

        # Calculate P&L
        # For simplicity, we calculate current option value and compare to entry price
        # More sophisticated: could model time decay scenarios
        pnl = np.zeros_like(S_grid)
        for i in range(vol_points):
            for j in range(spot_points):
                S = S_grid[i, j]
                sigma = vol_grid[i, j]

                if option_type.lower() == 'call':
                    current_price = self.model.call_price(S, K, T, sigma, r)
                else:
                    current_price = self.model.put_price(S, K, T, sigma, r)

                # P&L = current value - entry price
                if show_percentage:
                    pnl[i, j] = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl[i, j] = current_price - entry_price

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Determine symmetric color scale around zero for diverging colormap
        vmax = max(abs(np.min(pnl)), abs(np.max(pnl)))
        vmin = -vmax

        # Create discrete cell-based heatmap using pcolormesh
        # Normal RdYlGn colormap (green=profit, red=loss)
        im = ax.pcolormesh(S_grid, vol_grid * 100, pnl,
                          cmap=self.pnl_cmap,
                          vmin=vmin, vmax=vmax,
                          shading='auto', edgecolors='white', linewidth=0.5)

        # Add text annotations showing P&L values in each cell
        for i in range(vol_points):
            for j in range(spot_points):
                text_value = pnl[i, j]
                # Position text at center of cell
                text_x = spot_prices[j]
                text_y = volatilities[i] * 100

                # Determine text color based on P&L value (for readability)
                # For diverging colormap centered at 0
                norm_val = (text_value - vmin) / (vmax - vmin)
                text_color = 'white' if 0.3 < norm_val < 0.7 else 'black'

                # Format based on whether showing percentage or dollar
                if show_percentage:
                    display_text = f'{text_value:.1f}%'
                else:
                    display_text = f'${text_value:.2f}'

                ax.text(text_x, text_y, display_text,
                       ha='center', va='center', fontsize=7,
                       color=text_color, fontweight='bold')

        # Mark current spot and volatility
        if 'spot_price' in base_params and 'volatility' in base_params:
            current_S = base_params['spot_price']
            current_vol = base_params['volatility'] * 100
            ax.plot(current_S, current_vol, 'k*', markersize=15,
                   label='Current Position', markeredgecolor='yellow',
                   markeredgewidth=2)
            ax.legend(loc='upper left')

        # Labels and title
        ax.set_xlabel('Spot Price at Expiration ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')

        option_label = option_type.capitalize()
        pnl_unit = '%' if show_percentage else '$'
        title = f'{option_label} Option P&L Scenario Analysis\n' \
                f'Entry Price: ${entry_price:.2f}, Strike: ${K:.2f}, ' \
                f'Maturity: {T:.2f}Y'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar_label = 'P&L (%)' if show_percentage else 'P&L ($)'
        cbar.set_label(cbar_label, fontsize=11, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    def plot_greeks_heatmap(self,
                           greek: str,
                           base_params: Dict,
                           spot_range: Optional[Tuple[float, float]] = None,
                           vol_range: Optional[Tuple[float, float]] = None,
                           spot_points: int = 20,
                           vol_points: int = 20,
                           figsize: Tuple[float, float] = (10, 8)) -> Figure:
        """
        Create a heatmap for a specific Greek (Delta, Gamma, Vega, Theta, or Rho).

        Args:
            greek: Name of Greek to plot ('delta', 'gamma', 'vega', 'theta', 'rho')
            base_params: Dictionary with option parameters
            spot_range: (min, max) spot prices
            vol_range: (min, max) volatilities
            spot_points: Number of spot price grid points
            vol_points: Number of volatility grid points
            figsize: Figure size in inches

        Returns:
            Matplotlib Figure object
        """
        greek = greek.lower()
        valid_greeks = ['delta', 'gamma', 'vega', 'theta', 'rho']
        if greek not in valid_greeks:
            raise ValueError(f"greek must be one of {valid_greeks}, got '{greek}'")

        # Extract base parameters
        K = base_params['strike_price']
        T = base_params['time_to_maturity']
        r = base_params['risk_free_rate']
        option_type = base_params['option_type']

        # Determine ranges
        if spot_range is None:
            S_base = base_params.get('spot_price', K)
            spot_range = (S_base * 0.7, S_base * 1.3)

        if vol_range is None:
            vol_base = base_params.get('volatility', 0.2)
            vol_range = (vol_base * 0.5, vol_base * 1.5)

        # Create meshgrid
        spot_prices = np.linspace(spot_range[0], spot_range[1], spot_points)
        volatilities = np.linspace(vol_range[0], vol_range[1], vol_points)
        S_grid, vol_grid = np.meshgrid(spot_prices, volatilities)

        # Calculate Greek values
        greek_values = np.zeros_like(S_grid)
        for i in range(vol_points):
            for j in range(spot_points):
                S = S_grid[i, j]
                sigma = vol_grid[i, j]

                greeks_dict = self.model.greeks(S, K, T, sigma, r, option_type)
                greek_values[i, j] = greeks_dict[greek]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create discrete cell-based heatmap using pcolormesh
        # Use reversed RdYlGn (green=high, red=low) for Greeks
        im = ax.pcolormesh(S_grid, vol_grid * 100, greek_values,
                          cmap=self.price_cmap + '_r',
                          shading='auto', edgecolors='white', linewidth=0.5)

        # Add text annotations showing Greek values in each cell
        for i in range(vol_points):
            for j in range(spot_points):
                text_value = greek_values[i, j]
                # Position text at center of cell
                text_x = spot_prices[j]
                text_y = volatilities[i] * 100

                # Determine text color based on background (for readability)
                norm_val = (text_value - np.min(greek_values)) / (np.max(greek_values) - np.min(greek_values))
                text_color = 'white' if norm_val < 0.5 else 'black'

                ax.text(text_x, text_y, f'{text_value:.3f}',
                       ha='center', va='center', fontsize=7,
                       color=text_color, fontweight='bold')

        # Mark current position
        if 'spot_price' in base_params and 'volatility' in base_params:
            current_S = base_params['spot_price']
            current_vol = base_params['volatility'] * 100
            ax.plot(current_S, current_vol, 'r*', markersize=15,
                   label='Current Position', markeredgecolor='yellow',
                   markeredgewidth=2)
            ax.legend(loc='upper left')

        # Labels and title
        ax.set_xlabel('Spot Price ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')

        option_label = option_type.capitalize()
        greek_label = greek.capitalize()
        title = f'{option_label} Option {greek_label} Sensitivity\n' \
                f'Strike: ${K:.2f}, Maturity: {T:.2f}Y, Rate: {r*100:.1f}%'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(greek_label, fontsize=11, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    def plot_payoff_diagram(self,
                           base_params: Dict,
                           entry_price: float,
                           spot_range: Optional[Tuple[float, float]] = None,
                           spot_points: int = 100,
                           figsize: Tuple[float, float] = (10, 6)) -> Figure:
        """
        Create a classic option payoff diagram showing P&L at expiration.

        Args:
            base_params: Dictionary with option parameters
            entry_price: Premium paid/received
            spot_range: (min, max) spot prices at expiration
            spot_points: Number of points to plot
            figsize: Figure size in inches

        Returns:
            Matplotlib Figure object
        """
        K = base_params['strike_price']
        option_type = base_params['option_type']

        # Determine range
        if spot_range is None:
            S_base = base_params.get('spot_price', K)
            spot_range = (S_base * 0.5, S_base * 1.5)

        # Create spot price array
        spot_prices = np.linspace(spot_range[0], spot_range[1], spot_points)

        # Calculate payoff at expiration
        if option_type.lower() == 'call':
            intrinsic = np.maximum(spot_prices - K, 0)
        else:
            intrinsic = np.maximum(K - spot_prices, 0)

        # P&L = intrinsic value - entry price
        pnl = intrinsic - entry_price

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot P&L line
        ax.plot(spot_prices, pnl, 'b-', linewidth=2.5, label='P&L at Expiration')

        # Plot zero line (breakeven)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Plot strike price line
        ax.axvline(x=K, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                  label=f'Strike: ${K:.2f}')

        # Shade profit and loss regions
        ax.fill_between(spot_prices, 0, pnl, where=(pnl >= 0),
                       alpha=0.3, color='green', label='Profit Region')
        ax.fill_between(spot_prices, 0, pnl, where=(pnl < 0),
                       alpha=0.3, color='red', label='Loss Region')

        # Calculate breakeven point
        if option_type.lower() == 'call':
            breakeven = K + entry_price
        else:
            breakeven = K - entry_price

        if spot_range[0] <= breakeven <= spot_range[1]:
            ax.axvline(x=breakeven, color='orange', linestyle=':', linewidth=2,
                      label=f'Breakeven: ${breakeven:.2f}')

        # Labels and title
        ax.set_xlabel('Spot Price at Expiration ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Profit / Loss ($)', fontsize=12, fontweight='bold')

        option_label = option_type.capitalize()
        title = f'{option_label} Option Payoff Diagram\n' \
                f'Entry Price: ${entry_price:.2f}, Strike: ${K:.2f}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Legend and grid
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig


# Convenience functions for quick plotting
def quick_price_heatmap(S: float, K: float, T: float, sigma: float, r: float,
                       option_type: str = 'call') -> Figure:
    """
    Quick price heatmap with default settings.

    Example:
        >>> fig = quick_price_heatmap(100, 100, 1.0, 0.2, 0.05, 'call')
        >>> fig.savefig('heatmap.png')
    """
    viz = OptionVisualizer()
    params = {
        'spot_price': S,
        'strike_price': K,
        'time_to_maturity': T,
        'volatility': sigma,
        'risk_free_rate': r,
        'option_type': option_type
    }
    return viz.plot_price_heatmap(params)


def quick_pnl_heatmap(S: float, K: float, T: float, sigma: float, r: float,
                      entry_price: float, option_type: str = 'call') -> Figure:
    """
    Quick P&L heatmap with default settings.

    Example:
        >>> fig = quick_pnl_heatmap(100, 100, 1.0, 0.2, 0.05, 10.45, 'call')
        >>> fig.savefig('pnl.png')
    """
    viz = OptionVisualizer()
    params = {
        'spot_price': S,
        'strike_price': K,
        'time_to_maturity': T,
        'volatility': sigma,
        'risk_free_rate': r,
        'option_type': option_type
    }
    return viz.plot_pnl_heatmap(entry_price, params)
