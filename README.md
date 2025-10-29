# 🗿 Sigma Option

A professional Black-Scholes option pricing application with interactive P&L analysis and advanced visualization capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌐 Live Demo

**Website:** [🗿 Sigma Option babyyyyyyy!](https://sigmaoption-ducduong.streamlit.app/)

*The application is currently available for local deployment. Web hosting coming soon!*

---

## 📊 Project Overview

**Sigma Option** is a full-stack quantitative finance application implementing the **Black-Scholes-Merton** model with an emphasis on practical trading analysis. Built with modern Python tools, it offers real-time option pricing, comprehensive Greek calculations, and professional-grade visualization.

### 🎯 Key Differentiators

- **💰 P&L-First Approach**: Immediate profit/loss visualization with customizable entry prices
- **🎨 Interactive Heatmaps**: Cell-based 10×10 grids with clean value annotations
- **🔄 Side-by-Side Comparison**: Compare Call and Put options simultaneously
- **📈 Professional Visualizations**: Publication-quality charts with color-coded insights
- **💾 Historical Tracking**: SQLite database for calculation persistence and analysis

---

## ✨ Features

### 🚀 Latest Implementation (Phase 1: Interactive Heatmap Redesign)

#### Redesigned Heatmap Visualization
- **Cell-based Display**: Replaced continuous contour plots with discrete 10×10 cell grids
- **Value Annotations**: Each cell displays its numerical value for precise analysis
- **Smart Color Coding**:
  - P&L: Green (profit) → Yellow (breakeven) → Red (loss)
  - Prices: Green (high) → Yellow (mid) → Red (low)
- **Enhanced Readability**: Clean cell boundaries with optimized text contrast

#### Purchase Price Input & P&L Analysis
- **Entry Price Control**: Dedicated input field for option purchase price
- **Auto-Population**: Automatically populated with calculated fair value
- **Manual Override**: Modify entry price for "what-if" scenario analysis
- **Real-time P&L**: Instant profit/loss calculation: `Current Value - Purchase Price`

#### P&L-First Display Logic
- **Default P&L View**: Opens directly to profit/loss analysis
- **Dual Heatmap Comparison**: Side-by-side Call and Put P&L scenarios
- **Toggle Functionality**: Switch between "P&L View" and "Price View" modes
- **Comprehensive Metrics**: Fair value, purchase price, P&L ($), and P&L (%) displayed prominently

#### Enhanced Color Scheme
- **Diverging Colormap**: RdYlGn palette centered at zero for P&L
- **Visual Clarity**: Profit zones (green) and loss zones (red) immediately identifiable
- **Breakeven Zones**: Yellow/white regions near ±$0.50 for marginal scenarios

---

## 🛠️ Core Capabilities

### Option Pricing Engine
- ✅ European call and put option valuation
- ✅ Real-time fair value calculation
- ✅ Put-call parity verification
- ✅ Edge case handling (expiration, zero volatility)

### The Greeks (Sensitivity Measures)
- **Delta (Δ)**: Sensitivity to underlying price changes
- **Gamma (Γ)**: Rate of change of Delta
- **Vega (ν)**: Sensitivity to volatility changes
- **Theta (Θ)**: Time decay measurement
- **Rho (ρ)**: Sensitivity to interest rate changes

### Advanced Visualizations
1. **P&L Analysis** (Primary Tab)
   - Side-by-side Call/Put P&L heatmaps
   - Toggle between P&L and Price views
   - Customizable spot and volatility ranges

2. **Price Heatmaps**
   - Single option detailed analysis
   - Spot price vs. volatility sensitivity

3. **Greeks Heatmaps**
   - Visualize any Greek across parameter space
   - Delta, Gamma, Vega, Theta, Rho support

4. **Payoff Diagrams**
   - Classic profit/loss at expiration
   - Breakeven point identification

### Database & History
- SQLite persistence layer
- Calculation history tracking
- Advanced filtering by option type
- CSV export functionality
- Statistical summaries

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: 3.9 or higher
- **pip**: Package manager (usually bundled with Python)
- **Git**: For cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/blackschole_option.git
cd blackschole_option
```

### Step 2: Set Up Virtual Environment (Recommended)

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy>=1.24.0` - Numerical computations
- `scipy>=1.10.0` - Statistical functions
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualizations
- `streamlit>=1.28.0` - Web interface
- `pytest>=7.4.0` - Testing framework

### Step 4: Run on Localhost

```bash
streamlit run app/streamlit_app.py
```

**OR** if you encounter path issues:

```bash
python3 -m streamlit run app/streamlit_app.py
```

The application will automatically open in your default browser at:
- **Local URL**: `http://localhost:8501`
- **Network URL**: Your local IP (displayed in terminal)

### Step 5: Using the Application

1. **Sidebar Inputs**:
   - Select option type (Call/Put)
   - Enter purchase/entry price (auto-populated with fair value)
   - Set market parameters (S, K, T, σ, r)

2. **Main Dashboard**:
   - View fair value, purchase price, and P&L metrics
   - See moneyness status (ITM/ATM/OTM)
   - Check intrinsic and time value

3. **Sensitivity Analysis**:
   - Navigate to "💰 P&L Analysis" tab (default view)
   - Compare Call vs Put scenarios side-by-side
   - Toggle between P&L and Price views
   - Explore Greeks and payoff diagrams

4. **Database Operations**:
   - Save calculations for future reference
   - View calculation history
   - Export to CSV

---

## 💾 SQL Database Implementation Guide

### Automatic Initialization

The SQLite database is **automatically created** when you first run the application. No manual setup required!

**Database Location**: `option_calculations.db` (root directory)

### Manual Database Setup (Optional)

If you need to manually initialize or reset the database:

```bash
# Using Python
python3 -c "from db.db_handler import get_handler; get_handler().initialize_db()"
```

### Database Schema

#### Table: `inputs`
```sql
CREATE TABLE inputs (
    calculation_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    spot_price REAL NOT NULL,
    strike_price REAL NOT NULL,
    time_to_maturity REAL NOT NULL,
    volatility REAL NOT NULL,
    risk_free_rate REAL NOT NULL,
    option_type TEXT NOT NULL CHECK(option_type IN ('call', 'put'))
);
```

#### Table: `outputs`
```sql
CREATE TABLE outputs (
    calculation_id TEXT PRIMARY KEY,
    option_price REAL NOT NULL,
    delta REAL NOT NULL,
    gamma REAL NOT NULL,
    vega REAL NOT NULL,
    theta REAL NOT NULL,
    rho REAL NOT NULL,
    FOREIGN KEY (calculation_id) REFERENCES inputs(calculation_id) ON DELETE CASCADE
);
```

### Database Operations

#### Save a Calculation
```python
from db.db_handler import get_handler

handler = get_handler()

inputs = {
    'spot_price': 100,
    'strike_price': 100,
    'time_to_maturity': 1.0,
    'volatility': 0.2,
    'risk_free_rate': 0.05,
    'option_type': 'call'
}

outputs = {
    'option_price': 10.45,
    'delta': 0.637,
    'gamma': 0.019,
    'vega': 0.375,
    'theta': -0.012,
    'rho': 0.532
}

calc_id = handler.save_calculation(inputs, outputs)
print(f"Saved with ID: {calc_id}")
```

#### Query History
```python
# Get recent calculations
history = handler.get_calculation_history(limit=20)

# Filter by option type
call_history = handler.get_calculation_history(limit=50, option_type='call')

# Get as pandas DataFrame
df = handler.get_calculation_history_df(limit=100)
```

#### Database Statistics
```python
stats = handler.get_statistics()
print(f"Total calculations: {stats['total_calculations']}")
print(f"Call options: {stats['call_count']}")
print(f"Put options: {stats['put_count']}")
print(f"Average price: ${stats['avg_option_price']:.2f}")
```

### Database Management

**View database contents** (requires SQLite CLI):
```bash
sqlite3 option_calculations.db
.tables
SELECT * FROM inputs LIMIT 5;
.exit
```

**Backup database**:
```bash
cp option_calculations.db option_calculations_backup_$(date +%Y%m%d).db
```

**Reset database** (deletes all data):
```bash
rm option_calculations.db
# Database will be recreated on next app launch
```

---

## 📝 Usage Examples

### Web Interface Workflow

```
1. Open http://localhost:8501
2. Select "Call" option type
3. Enter parameters:
   - Spot Price: $100
   - Strike Price: $100
   - Time to Maturity: 1.0 years
   - Volatility: 20%
   - Risk-Free Rate: 5%
4. Set purchase price: $10.45 (auto-filled)
5. View results:
   - Fair Value: $10.45
   - P&L: $0.00 (0%)
6. Adjust spot price to $105 → See updated P&L
7. Navigate to "P&L Analysis" tab
8. Compare Call vs Put scenarios
9. Save calculation to database
```

### Programmatic Usage

#### Basic Option Pricing
```python
from core.black_scholes import BlackScholesModel

model = BlackScholesModel()

# Price a call option
call = model.call_price(S=100, K=100, T=1.0, sigma=0.2, r=0.05)
print(f"Call: ${call:.2f}")  # Call: $10.45

# Price a put option
put = model.put_price(S=100, K=100, T=1.0, sigma=0.2, r=0.05)
print(f"Put: ${put:.2f}")    # Put: $5.57
```

#### Calculate Greeks
```python
greeks = model.greeks(S=100, K=100, T=1.0, sigma=0.2, r=0.05, option_type='call')

for name, value in greeks.items():
    print(f"{name.capitalize()}: {value:.4f}")
```

#### Generate Heatmaps
```python
from visuals.heatmap import OptionVisualizer

viz = OptionVisualizer()

params = {
    'spot_price': 100,
    'strike_price': 100,
    'time_to_maturity': 1.0,
    'volatility': 0.2,
    'risk_free_rate': 0.05,
    'option_type': 'call'
}

# P&L heatmap
fig = viz.plot_pnl_heatmap(
    entry_price=10.45,
    base_params=params,
    show_percentage=False
)
fig.savefig('pnl_analysis.png', dpi=300, bbox_inches='tight')
```

---

## 🗂️ Project Structure

```
blackschole_option/
├── core/
│   ├── __init__.py
│   └── black_scholes.py          # Pricing engine & Greeks
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # Web interface (Sigma Option UI)
├── visuals/
│   ├── __init__.py
│   └── heatmap.py                # Heatmap visualizations (10×10 grids)
├── db/
│   ├── __init__.py
│   ├── schema.sql                # SQLite schema
│   └── db_handler.py             # Database operations
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py     # 41 unit tests
│   └── test_db_handler.py        # 36 unit tests
├── requirements.txt               # Python dependencies
├── milestones.md                  # Implementation roadmap
├── README.md                      # This file
└── option_calculations.db         # SQLite database (auto-created)
```

---

## 🧪 Testing

Run the complete test suite:

```bash
# All tests
pytest

# With coverage report
pytest --cov=core --cov=db --cov=visuals tests/

# Verbose output
pytest -v

# Specific test file
pytest tests/test_black_scholes.py
```

**Test Coverage Summary:**
- **Total Tests**: 77 passing
- **Black-Scholes Model**: 41 tests (pricing, Greeks, edge cases, validation)
- **Database Handler**: 36 tests (CRUD, constraints, queries, statistics)

---

## 📐 Mathematical Background

### Black-Scholes Formula

**Call Option:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Option:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

**Where:**
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

- **S₀**: Current spot price
- **K**: Strike price
- **T**: Time to maturity (years)
- **σ**: Volatility (annualized)
- **r**: Risk-free rate (annualized)
- **N(x)**: Cumulative standard normal distribution

### Model Assumptions

1. European-style exercise only
2. No dividends during option life
3. Constant volatility and risk-free rate
4. Log-normal distribution of stock prices
5. Frictionless markets (no transaction costs/taxes)
6. Continuous trading possible
7. Unlimited borrowing/lending at risk-free rate

---

## 🛠️ Technical Stack

| Category | Technology |
|----------|-----------|
| **Core Language** | Python 3.9+ |
| **Scientific Computing** | NumPy, SciPy |
| **Data Analysis** | Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Database** | SQLite3 |
| **Testing** | pytest, pytest-cov |
| **Version Control** | Git |

---

## 🚧 Roadmap & Future Enhancements

### Phase 2: Advanced Features (Planned)
- [ ] Real-time market data integration (Yahoo Finance API)
- [ ] Implied volatility surface visualization
- [ ] Historical volatility calculation
- [ ] Multi-leg option strategies (spreads, straddles, strangles)

### Phase 3: Infrastructure (Planned)
- [ ] Docker containerization
- [ ] REST API (FastAPI)
- [ ] Cloud deployment (AWS/Heroku/Railway)
- [ ] User authentication system
- [ ] WebSocket for real-time updates

### Phase 4: Advanced Models (Future)
- [ ] American option pricing (Binomial tree)
- [ ] Monte Carlo simulation engine
- [ ] Exotic options (Asian, Barrier, Lookback)
- [ ] Volatility smile/skew modeling

---

## 🌐 Deployment

### Coming Soon: Web Hosting

The application will be deployed to a public URL for easy access without local installation. Planned hosting options:

- **Streamlit Cloud** (Primary)
- **Heroku** (Alternative)
- **Railway** (Alternative)

**Current Status**: Local deployment only
**Expected Launch**: [TBD]

**Check back soon for the live demo link!**

---

## 📜 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 👨‍💻 Author

**Duong Hong Duc**
Grinnell College

This project demonstrates:
- Quantitative finance modeling
- Full-stack development skills
- Data visualization expertise
- Software engineering best practices
- Database design and management

---

## 🙏 Acknowledgments

- Hull, J. C. (2017). *Options, Futures, and Other Derivatives*
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Merton, R. C. (1973). *Theory of Rational Option Pricing*

---

## 📚 References

1. [Black-Scholes Model - Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
2. [Option Greeks - Investopedia](https://www.investopedia.com/trading/getting-to-know-the-greeks/)
3. [Put-Call Parity - CFI](https://corporatefinanceinstitute.com/resources/derivatives/put-call-parity/)
4. [Streamlit Documentation](https://docs.streamlit.io/)

---

<p align="center">
  <strong>🗿 Sigma Option</strong><br>
  <i>Professional Black-Scholes option pricing with interactive P&L analysis</i><br>
  Built with ❤️ for quantitative finance enthusiasts
</p>
