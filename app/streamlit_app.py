"""
Black-Scholes Option Pricer - Streamlit Application

An interactive web application for pricing European options using the
Black-Scholes model, visualizing sensitivity analysis, and managing
calculation history.

Run with: streamlit run app/streamlit_app.py

Author: Duong Hong Duc
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.black_scholes import BlackScholesModel, verify_put_call_parity
from visuals.heatmap import OptionVisualizer
from db.db_handler import get_handler

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .greek-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'calculation_saved' not in st.session_state:
        st.session_state.calculation_saved = False
    if 'last_calc_id' not in st.session_state:
        st.session_state.last_calc_id = None


def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">📈 Black-Scholes Option Pricer</h1>',
                unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Professional option pricing, Greeks calculation, and sensitivity analysis
        </p>
    """, unsafe_allow_html=True)


def get_input_parameters():
    """
    Get option parameters from sidebar input.

    Returns:
        Dictionary containing all input parameters
    """
    st.sidebar.header("📊 Option Parameters")

    # Option type
    option_type = st.sidebar.radio(
        "Option Type",
        ["Call", "Put"],
        help="Select the type of option to price"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Market Parameters")

    # Spot price
    spot_price = st.sidebar.number_input(
        "Spot Price (S)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="Current price of the underlying asset"
    )

    # Strike price
    strike_price = st.sidebar.number_input(
        "Strike Price (K)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="Exercise price of the option"
    )

    # Time to maturity
    time_to_maturity = st.sidebar.number_input(
        "Time to Maturity (T) [Years]",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.25,
        format="%.2f",
        help="Time until option expiration in years"
    )

    # Volatility
    volatility = st.sidebar.number_input(
        "Volatility (σ) [%]",
        min_value=0.0,
        max_value=200.0,
        value=20.0,
        step=1.0,
        format="%.1f",
        help="Annualized volatility of the underlying asset"
    ) / 100.0  # Convert from percentage

    # Risk-free rate
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (r) [%]",
        min_value=-10.0,
        max_value=50.0,
        value=5.0,
        step=0.25,
        format="%.2f",
        help="Annualized risk-free interest rate"
    ) / 100.0  # Convert from percentage

    return {
        'option_type': option_type.lower(),
        'spot_price': spot_price,
        'strike_price': strike_price,
        'time_to_maturity': time_to_maturity,
        'volatility': volatility,
        'risk_free_rate': risk_free_rate
    }


def display_option_price(params, model):
    """Display the calculated option price and moneyness."""
    st.subheader("💰 Option Valuation")

    # Calculate price
    if params['option_type'] == 'call':
        price = model.call_price(
            params['spot_price'],
            params['strike_price'],
            params['time_to_maturity'],
            params['volatility'],
            params['risk_free_rate']
        )
    else:
        price = model.put_price(
            params['spot_price'],
            params['strike_price'],
            params['time_to_maturity'],
            params['volatility'],
            params['risk_free_rate']
        )

    # Determine moneyness
    S, K = params['spot_price'], params['strike_price']
    if params['option_type'] == 'call':
        if S > K * 1.05:
            moneyness = "In-the-Money (ITM)"
            moneyness_color = "green"
        elif S < K * 0.95:
            moneyness = "Out-of-the-Money (OTM)"
            moneyness_color = "red"
        else:
            moneyness = "At-the-Money (ATM)"
            moneyness_color = "blue"
    else:  # put
        if S < K * 0.95:
            moneyness = "In-the-Money (ITM)"
            moneyness_color = "green"
        elif S > K * 1.05:
            moneyness = "Out-of-the-Money (OTM)"
            moneyness_color = "red"
        else:
            moneyness = "At-the-Money (ATM)"
            moneyness_color = "blue"

    # Display in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Option Price",
            value=f"${price:.4f}",
            help="Theoretical fair value of the option"
        )

    with col2:
        st.metric(
            label="Intrinsic Value",
            value=f"${max(S - K, 0) if params['option_type'] == 'call' else max(K - S, 0):.4f}",
            help="Value if exercised immediately"
        )

    with col3:
        st.markdown(f"**Moneyness**")
        st.markdown(f"<p style='color: {moneyness_color}; font-size: 1.2rem; font-weight: bold;'>{moneyness}</p>",
                   unsafe_allow_html=True)

    return price


def display_greeks(params, model):
    """Display the Greeks for the option."""
    st.subheader("📐 The Greeks")

    # Calculate Greeks
    greeks = model.greeks(
        params['spot_price'],
        params['strike_price'],
        params['time_to_maturity'],
        params['volatility'],
        params['risk_free_rate'],
        params['option_type']
    )

    # Display Greeks in columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Delta (Δ)",
            value=f"{greeks['delta']:.4f}",
            help="Sensitivity to underlying price change"
        )

    with col2:
        st.metric(
            label="Gamma (Γ)",
            value=f"{greeks['gamma']:.4f}",
            help="Rate of change of Delta"
        )

    with col3:
        st.metric(
            label="Vega (ν)",
            value=f"{greeks['vega']:.4f}",
            help="Sensitivity to 1% volatility change"
        )

    with col4:
        st.metric(
            label="Theta (Θ)",
            value=f"{greeks['theta']:.4f}",
            help="Daily time decay"
        )

    with col5:
        st.metric(
            label="Rho (ρ)",
            value=f"{greeks['rho']:.4f}",
            help="Sensitivity to 1% rate change"
        )

    # Add explanations in expander
    with st.expander("ℹ️ Understanding the Greeks"):
        st.markdown("""
        **Delta (Δ):** Change in option price for $1 change in underlying price
        - Call: 0 to 1 | Put: -1 to 0

        **Gamma (Γ):** Change in Delta for $1 change in underlying price
        - Higher Gamma = Delta changes faster

        **Vega (ν):** Change in option price for 1% change in volatility
        - Always positive (higher vol = higher option value)

        **Theta (Θ):** Change in option price per day (time decay)
        - Usually negative (options lose value over time)

        **Rho (ρ):** Change in option price for 1% change in interest rate
        - Call: positive | Put: negative
        """)

    return greeks


def display_put_call_parity(params, model):
    """Display put-call parity verification."""
    with st.expander("🔍 Put-Call Parity Verification"):
        is_valid, diff = verify_put_call_parity(
            params['spot_price'],
            params['strike_price'],
            params['time_to_maturity'],
            params['volatility'],
            params['risk_free_rate']
        )

        if is_valid:
            st.success(f"✅ Put-Call Parity holds! Difference: {diff:.2e}")
        else:
            st.warning(f"⚠️ Put-Call Parity violation detected. Difference: {diff:.6f}")

        st.markdown("""
        **Put-Call Parity:** C - P = S - K·e^(-rT)

        This fundamental relationship ensures arbitrage-free pricing.
        """)


def display_visualizations(params, price):
    """Display interactive visualizations."""
    st.subheader("📊 Sensitivity Analysis")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Heatmap",
        "P&L Scenarios",
        "Greeks Heatmap",
        "Payoff Diagram"
    ])

    viz = OptionVisualizer()

    with tab1:
        st.markdown("**Option Price Sensitivity to Spot Price and Volatility**")

        # Customization options
        col1, col2 = st.columns(2)
        with col1:
            spot_range_pct = st.slider(
                "Spot Price Range (%)",
                10, 50, 30,
                help="Range around current spot price"
            )
        with col2:
            vol_range_pct = st.slider(
                "Volatility Range (%)",
                10, 100, 50,
                help="Range around current volatility"
            )

        # Create heatmap
        S_min = params['spot_price'] * (1 - spot_range_pct / 100)
        S_max = params['spot_price'] * (1 + spot_range_pct / 100)
        vol_min = params['volatility'] * (1 - vol_range_pct / 100)
        vol_max = params['volatility'] * (1 + vol_range_pct / 100)

        fig = viz.plot_price_heatmap(
            params,
            spot_range=(S_min, S_max),
            vol_range=(vol_min, vol_max)
        )
        st.pyplot(fig)

    with tab2:
        st.markdown("**Profit & Loss Scenarios**")

        show_pct = st.checkbox("Show P&L as percentage", value=False)

        fig = viz.plot_pnl_heatmap(
            price,
            params,
            show_percentage=show_pct
        )
        st.pyplot(fig)

    with tab3:
        st.markdown("**Greek Sensitivity Analysis**")

        greek_choice = st.selectbox(
            "Select Greek to visualize",
            ["Delta", "Gamma", "Vega", "Theta", "Rho"]
        )

        fig = viz.plot_greeks_heatmap(
            greek_choice.lower(),
            params
        )
        st.pyplot(fig)

    with tab4:
        st.markdown("**Classic Payoff Diagram at Expiration**")

        fig = viz.plot_payoff_diagram(params, price)
        st.pyplot(fig)


def database_operations(params, price, greeks):
    """Handle database save and history viewing."""
    st.subheader("💾 Database Operations")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Calculation to Database", use_container_width=True):
            try:
                handler = get_handler()

                inputs = {
                    'spot_price': params['spot_price'],
                    'strike_price': params['strike_price'],
                    'time_to_maturity': params['time_to_maturity'],
                    'volatility': params['volatility'],
                    'risk_free_rate': params['risk_free_rate'],
                    'option_type': params['option_type']
                }

                outputs = {
                    'option_price': price,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'vega': greeks['vega'],
                    'theta': greeks['theta'],
                    'rho': greeks['rho']
                }

                calc_id = handler.save_calculation(inputs, outputs)
                st.session_state.calculation_saved = True
                st.session_state.last_calc_id = calc_id
                st.success(f"✅ Calculation saved! ID: {calc_id[:8]}...")

            except Exception as e:
                st.error(f"❌ Error saving calculation: {str(e)}")

    with col2:
        if st.button("📋 View Calculation History", use_container_width=True):
            st.session_state.show_history = True


def display_calculation_history():
    """Display calculation history from database."""
    if st.session_state.get('show_history', False):
        st.subheader("📚 Calculation History")

        try:
            handler = get_handler()

            # Get statistics
            stats = handler.get_statistics()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Calculations", stats['total_calculations'])
            with col2:
                st.metric("Call Options", stats['call_count'])
            with col3:
                st.metric("Put Options", stats['put_count'])
            with col4:
                st.metric("Avg Price", f"${stats['avg_option_price']:.2f}")

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_type = st.selectbox(
                    "Filter by type",
                    ["All", "Call", "Put"]
                )
            with col2:
                limit = st.number_input(
                    "Number of records",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5
                )

            # Get history
            option_filter = filter_type.lower() if filter_type != "All" else None
            df = handler.get_calculation_history_df(
                limit=limit,
                option_type=option_filter
            )

            if not df.empty:
                # Format the dataframe
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df['volatility'] = (df['volatility'] * 100).round(2)
                df['risk_free_rate'] = (df['risk_free_rate'] * 100).round(2)

                # Rename columns for display
                display_df = df[[
                    'timestamp', 'option_type', 'spot_price', 'strike_price',
                    'time_to_maturity', 'volatility', 'risk_free_rate',
                    'option_price', 'delta'
                ]].copy()

                display_df.columns = [
                    'Timestamp', 'Type', 'Spot', 'Strike', 'Maturity',
                    'Vol (%)', 'Rate (%)', 'Price', 'Delta'
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"option_calculations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No calculations found in database.")

        except Exception as e:
            st.error(f"❌ Error loading history: {str(e)}")


def display_sidebar_info():
    """Display additional information in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")

    st.sidebar.markdown("""
    This application uses the **Black-Scholes-Merton** model to price
    European options and calculate their Greeks.

    **Features:**
    - Real-time option pricing
    - Complete Greeks calculation
    - Interactive visualizations
    - Calculation history storage
    - P&L scenario analysis

    **Assumptions:**
    - European-style exercise
    - No dividends
    - Constant volatility and rates
    - Log-normal price distribution
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <p style='text-align: center; color: #666; font-size: 0.8rem;'>
    Built with Python, Streamlit, and NumPy<br>
    By Duong Hong Duc
    </p>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Initialize
    initialize_session_state()

    # Display header
    display_header()

    # Get input parameters
    params = get_input_parameters()

    # Display sidebar info
    display_sidebar_info()

    # Create model instance
    model = BlackScholesModel()

    # Main content
    try:
        # Display option price
        price = display_option_price(params, model)

        # Display Greeks
        greeks = display_greeks(params, model)

        # Put-call parity verification
        display_put_call_parity(params, model)

        st.markdown("---")

        # Visualizations
        display_visualizations(params, price)

        st.markdown("---")

        # Database operations
        database_operations(params, price, greeks)

        # Display history if requested
        display_calculation_history()

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
