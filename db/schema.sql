-- Black-Scholes Option Pricer Database Schema
-- SQLite database for storing option pricing calculations
--
-- Design: Two-table structure with normalized inputs and outputs
-- linked by calculation_id (UUID) for data integrity

-- Table: inputs
-- Stores all input parameters for each option pricing calculation
CREATE TABLE IF NOT EXISTS inputs (
    calculation_id TEXT PRIMARY KEY,          -- UUID unique identifier
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,  -- When calculation was performed
    spot_price REAL NOT NULL CHECK(spot_price > 0),  -- Current stock price (S)
    strike_price REAL NOT NULL CHECK(strike_price > 0),  -- Strike price (K)
    time_to_maturity REAL NOT NULL CHECK(time_to_maturity >= 0),  -- Time to expiration in years (T)
    volatility REAL NOT NULL CHECK(volatility >= 0),  -- Annualized volatility (σ)
    risk_free_rate REAL NOT NULL CHECK(risk_free_rate > -1),  -- Annualized risk-free rate (r)
    option_type TEXT NOT NULL CHECK(option_type IN ('call', 'put'))  -- Option type
);

-- Table: outputs
-- Stores calculated option prices and Greeks for each calculation
CREATE TABLE IF NOT EXISTS outputs (
    calculation_id TEXT PRIMARY KEY,          -- Links to inputs table
    option_price REAL NOT NULL,               -- Calculated option price
    delta REAL NOT NULL,                      -- Delta: ∂V/∂S
    gamma REAL NOT NULL,                      -- Gamma: ∂²V/∂S²
    vega REAL NOT NULL,                       -- Vega: ∂V/∂σ
    theta REAL NOT NULL,                      -- Theta: ∂V/∂t
    rho REAL NOT NULL,                        -- Rho: ∂V/∂r
    FOREIGN KEY (calculation_id) REFERENCES inputs(calculation_id) ON DELETE CASCADE
);

-- Index on timestamp for efficient historical queries
CREATE INDEX IF NOT EXISTS idx_inputs_timestamp ON inputs(timestamp DESC);

-- Index on option type for filtering
CREATE INDEX IF NOT EXISTS idx_inputs_option_type ON inputs(option_type);

-- Index on spot and strike prices for range queries
CREATE INDEX IF NOT EXISTS idx_inputs_spot_strike ON inputs(spot_price, strike_price);

-- View: calculation_history
-- Convenient joined view combining inputs and outputs
CREATE VIEW IF NOT EXISTS calculation_history AS
SELECT
    i.calculation_id,
    i.timestamp,
    i.spot_price,
    i.strike_price,
    i.time_to_maturity,
    i.volatility,
    i.risk_free_rate,
    i.option_type,
    o.option_price,
    o.delta,
    o.gamma,
    o.vega,
    o.theta,
    o.rho
FROM inputs i
LEFT JOIN outputs o ON i.calculation_id = o.calculation_id
ORDER BY i.timestamp DESC;
