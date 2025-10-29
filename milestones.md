# Black-Scholes Option Pricer - Implementation Plan

## Project Structure
```
blackschole_option/
├── core/
│   ├── __init__.py
│   └── black_scholes.py          # BS formula + Greeks
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # Interactive GUI
├── visuals/
│   ├── __init__.py
│   └── heatmap.py                # Price & P&L heatmaps
├── db/
│   ├── __init__.py
│   ├── schema.sql                # SQLite schema
│   └── db_handler.py             # Database operations
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py     # Model accuracy tests
│   └── test_db_handler.py        # DB tests
├── requirements.txt
├── milestones.md                 # This implementation plan
└── README.md
```

---

## Stage 1: Core Black-Scholes Model
**File:** `core/black_scholes.py`

**Objectives:**
- Implement `BlackScholesModel` class with clean, professional code
- Calculate European call and put option prices
- Compute all Greeks: Delta, Gamma, Vega, Theta, Rho (for both calls and puts)
- Use `scipy.stats.norm` for cumulative distribution and probability density functions
- Add comprehensive input validation and error handling
- Include detailed docstrings explaining the mathematics

**Key Functions:**
- `call_price(S, K, T, sigma, r)` → float
- `put_price(S, K, T, sigma, r)` → float
- `greeks(S, K, T, sigma, r, option_type)` → dict with all Greeks
- Helper functions: `_d1()`, `_d2()` for clean code organization

---

## Stage 2: Streamlit GUI
**File:** `app/streamlit_app.py`

**Objectives:**
- Create an intuitive, professional interface for option pricing
- Real-time calculation and display of results
- Clean layout with organized sections

**Features:**
- **Input Section:** Sidebar with number inputs for S, K, T, σ, r
- **Option Type:** Radio buttons for Call/Put selection
- **Results Display:**
  - Price prominently displayed
  - Greeks organized in columns or expandable sections
  - Color coding for positive/negative values
- **Action Buttons:**
  - "Save to Database" button
  - "View History" to query past calculations
- **Visual Integration:** Embed heatmaps directly in the app

---

## Stage 3: Visualization Layer
**File:** `visuals/heatmap.py`

**Objectives:**
- Create publication-quality heatmaps for analysis
- Show both theoretical prices and P&L scenarios

**Visualizations:**

1. **Price Sensitivity Heatmap**
   - X-axis: Stock price (S) range around current spot
   - Y-axis: Volatility (σ) range
   - Z-axis (color): Option price
   - Function: `plot_price_heatmap(base_params, S_range, sigma_range)`

2. **P&L Scenario Heatmap**
   - Show profit/loss at expiration for different spot prices
   - Reference line at entry price (breakeven analysis)
   - X-axis: Spot price at expiration
   - Y-axis: Entry volatility levels
   - Z-axis (color): P&L in dollars/percentage
   - Function: `plot_pnl_heatmap(entry_price, base_params, S_range)`

**Styling:**
- Use `seaborn` with professional color schemes (e.g., RdYlGn for P&L)
- Clear labels, titles, and colorbars
- Matplotlib backend for Streamlit integration

---

## Stage 4: Database Layer (SQLite)
**Files:** `db/schema.sql`, `db/db_handler.py`

**Schema Design:**

**Table: `inputs`**
- `calculation_id` (TEXT PRIMARY KEY) - UUID
- `timestamp` (DATETIME)
- `spot_price` (REAL)
- `strike_price` (REAL)
- `time_to_maturity` (REAL)
- `volatility` (REAL)
- `risk_free_rate` (REAL)
- `option_type` (TEXT) - 'call' or 'put'

**Table: `outputs`**
- `calculation_id` (TEXT PRIMARY KEY, FOREIGN KEY)
- `option_price` (REAL)
- `delta` (REAL)
- `gamma` (REAL)
- `vega` (REAL)
- `theta` (REAL)
- `rho` (REAL)

**Database Handler Functions:**
- `initialize_db()` - Create tables if not exist
- `save_calculation(inputs_dict, outputs_dict)` - Insert new calculation
- `get_calculation_history(limit=50)` - Query recent calculations
- `get_calculation_by_id(calc_id)` - Retrieve specific calculation
- `delete_calculation(calc_id)` - Cleanup operations

---

## Stage 5: Testing Suite
**Files:** `tests/test_black_scholes.py`, `tests/test_db_handler.py`

**Black-Scholes Tests:**
- **Accuracy Tests:** Validate against known option prices from literature
- **Put-Call Parity:** Verify C - P = S - K*e^(-rT)
- **Edge Cases:**
  - T → 0 (intrinsic value convergence)
  - σ → 0 (deterministic case)
  - Deep ITM/OTM options
- **Greeks Validation:**
  - Delta bounds: [0,1] for calls, [-1,0] for puts
  - Gamma symmetry between calls and puts
  - Numerical accuracy using finite differences

**Database Tests:**
- CRUD operation integrity
- Foreign key constraint validation
- Query performance with mock data

**Test Framework:**
- Use `pytest` with fixtures
- Parameterized tests for multiple scenarios
- Coverage reports

---

## Stage 6: Documentation
**File:** `README.md`

**Sections:**
1. **Project Overview** - Purpose and features
2. **Mathematical Background** - Brief BS formula explanation
3. **Installation** - Dependencies and setup steps
4. **Usage Guide** - Running the Streamlit app
5. **Project Structure** - File organization
6. **Testing** - How to run tests
7. **Future Enhancements** - Potential extensions (American options, implied vol, etc.)

**requirements.txt:**
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
pytest>=7.4.0
```

---

## Implementation Order

1. ✅ Create project structure (all directories and `__init__.py` files)
2. ✅ Write `milestones.md` (this file)
3. ⏳ Implement `core/black_scholes.py` with full Greeks
4. ⏳ Write `tests/test_black_scholes.py` and validate model
5. ⏳ Create `db/schema.sql` and `db/db_handler.py`
6. ⏳ Write `tests/test_db_handler.py` and validate DB operations
7. ⏳ Implement `visuals/heatmap.py` with both visualization types
8. ⏳ Build `app/streamlit_app.py` integrating all components
9. ⏳ Create `requirements.txt`
10. ⏳ Write comprehensive `README.md`
11. ⏳ Final testing and polish

---

## Success Criteria

- All unit tests pass with >90% coverage
- Streamlit app runs without errors and displays correctly
- Database successfully persists and retrieves calculations
- Heatmaps render clearly and provide actionable insights
- Code is clean, documented, and follows PEP 8
- README enables someone to clone and run the project in <5 minutes

---

## Mathematical Reference

### Black-Scholes Formula

**Call Option Price:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Option Price:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### The Greeks

- **Delta (Δ):** Rate of change of option price with respect to underlying price
  - Call: N(d₁)
  - Put: N(d₁) - 1

- **Gamma (Γ):** Rate of change of delta with respect to underlying price
  - Call/Put: N'(d₁) / (S₀σ√T)

- **Vega (ν):** Rate of change of option price with respect to volatility
  - Call/Put: S₀N'(d₁)√T

- **Theta (Θ):** Rate of change of option price with respect to time
  - Call: -(S₀N'(d₁)σ)/(2√T) - rKe^(-rT)N(d₂)
  - Put: -(S₀N'(d₁)σ)/(2√T) + rKe^(-rT)N(-d₂)

- **Rho (ρ):** Rate of change of option price with respect to interest rate
  - Call: KTe^(-rT)N(d₂)
  - Put: -KTe^(-rT)N(-d₂)

Where N(x) is the cumulative standard normal distribution and N'(x) is the standard normal probability density function.
