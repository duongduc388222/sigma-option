"""
Database handler for Black-Scholes option pricing calculations.

This module provides a clean interface for persisting and retrieving
option pricing calculations using SQLite.

Usage:
    >>> handler = DatabaseHandler()
    >>> handler.initialize_db()
    >>> calc_id = handler.save_calculation(inputs, outputs)
    >>> history = handler.get_calculation_history(limit=10)

Author: Duong Hong Duc
"""

import sqlite3
import uuid
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd


class DatabaseHandler:
    """
    Handler for SQLite database operations for option pricing calculations.

    This class manages all database interactions including initialization,
    CRUD operations, and querying calculation history.
    """

    def __init__(self, db_path: str = "option_pricing.db"):
        """
        Initialize the database handler.

        Args:
            db_path: Path to SQLite database file (default: "option_pricing.db")
        """
        self.db_path = db_path
        self.schema_path = Path(__file__).parent / "schema.sql"

    def _get_connection(self) -> sqlite3.Connection:
        """
        Create and return a database connection.

        Returns:
            SQLite connection object with row factory set

        Note:
            Row factory allows accessing columns by name
            Foreign keys are enabled for referential integrity
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        return conn

    def initialize_db(self) -> None:
        """
        Initialize the database by creating tables from schema.

        This method reads the schema.sql file and executes it to create
        tables, indices, and views. It's safe to call multiple times.

        Raises:
            FileNotFoundError: If schema.sql is not found
            sqlite3.Error: If database creation fails
        """
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, 'r') as f:
            schema_sql = f.read()

        try:
            conn = self._get_connection()
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to initialize database: {e}")

    def save_calculation(self, inputs: Dict, outputs: Dict) -> str:
        """
        Save an option pricing calculation to the database.

        Args:
            inputs: Dictionary containing:
                - spot_price (float): Current stock price
                - strike_price (float): Strike price
                - time_to_maturity (float): Time to expiration in years
                - volatility (float): Annualized volatility
                - risk_free_rate (float): Annualized risk-free rate
                - option_type (str): 'call' or 'put'
            outputs: Dictionary containing:
                - option_price (float): Calculated option price
                - delta (float): Delta Greek
                - gamma (float): Gamma Greek
                - vega (float): Vega Greek
                - theta (float): Theta Greek
                - rho (float): Rho Greek

        Returns:
            calculation_id (str): UUID identifier for the saved calculation

        Example:
            >>> inputs = {
            ...     'spot_price': 100,
            ...     'strike_price': 100,
            ...     'time_to_maturity': 1.0,
            ...     'volatility': 0.2,
            ...     'risk_free_rate': 0.05,
            ...     'option_type': 'call'
            ... }
            >>> outputs = {
            ...     'option_price': 10.45,
            ...     'delta': 0.637,
            ...     'gamma': 0.019,
            ...     'vega': 0.375,
            ...     'theta': -0.012,
            ...     'rho': 0.532
            ... }
            >>> calc_id = handler.save_calculation(inputs, outputs)
        """
        calculation_id = str(uuid.uuid4())

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Insert inputs
            cursor.execute("""
                INSERT INTO inputs (
                    calculation_id, spot_price, strike_price, time_to_maturity,
                    volatility, risk_free_rate, option_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                calculation_id,
                inputs['spot_price'],
                inputs['strike_price'],
                inputs['time_to_maturity'],
                inputs['volatility'],
                inputs['risk_free_rate'],
                inputs['option_type']
            ))

            # Insert outputs
            cursor.execute("""
                INSERT INTO outputs (
                    calculation_id, option_price, delta, gamma, vega, theta, rho
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                calculation_id,
                outputs['option_price'],
                outputs['delta'],
                outputs['gamma'],
                outputs['vega'],
                outputs['theta'],
                outputs['rho']
            ))

            conn.commit()
            return calculation_id

        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise ValueError(f"Invalid data provided: {e}")
        except sqlite3.Error as e:
            conn.rollback()
            raise sqlite3.Error(f"Failed to save calculation: {e}")
        finally:
            conn.close()

    def get_calculation_by_id(self, calculation_id: str) -> Optional[Dict]:
        """
        Retrieve a specific calculation by its ID.

        Args:
            calculation_id: UUID of the calculation

        Returns:
            Dictionary containing all inputs and outputs, or None if not found

        Example:
            >>> calc = handler.get_calculation_by_id(calc_id)
            >>> print(f"Option price: ${calc['option_price']:.2f}")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM calculation_history
                WHERE calculation_id = ?
            """, (calculation_id,))

            row = cursor.fetchone()
            if row is None:
                return None

            return dict(row)

        finally:
            conn.close()

    def get_calculation_history(self, limit: int = 50,
                                option_type: Optional[str] = None) -> List[Dict]:
        """
        Retrieve recent calculation history.

        Args:
            limit: Maximum number of records to return (default: 50)
            option_type: Filter by option type ('call' or 'put'), or None for all

        Returns:
            List of dictionaries containing calculation details, ordered by
            most recent first

        Example:
            >>> history = handler.get_calculation_history(limit=10, option_type='call')
            >>> for calc in history:
            ...     print(f"{calc['timestamp']}: ${calc['option_price']:.2f}")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if option_type:
                cursor.execute("""
                    SELECT * FROM calculation_history
                    WHERE option_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (option_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM calculation_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        finally:
            conn.close()

    def get_calculation_history_df(self, limit: int = 50,
                                   option_type: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve calculation history as a pandas DataFrame.

        Args:
            limit: Maximum number of records to return
            option_type: Filter by option type, or None for all

        Returns:
            DataFrame containing calculation history

        Example:
            >>> df = handler.get_calculation_history_df(limit=100)
            >>> print(df.describe())
        """
        history = self.get_calculation_history(limit, option_type)
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def delete_calculation(self, calculation_id: str) -> bool:
        """
        Delete a specific calculation from the database.

        Args:
            calculation_id: UUID of the calculation to delete

        Returns:
            True if deletion was successful, False if calculation not found

        Note:
            Due to CASCADE constraint, deleting from inputs table also
            deletes the corresponding outputs record.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM inputs
                WHERE calculation_id = ?
            """, (calculation_id,))

            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

        finally:
            conn.close()

    def get_calculation_count(self, option_type: Optional[str] = None) -> int:
        """
        Get the total number of calculations stored.

        Args:
            option_type: Filter by option type, or None for all

        Returns:
            Number of calculations in the database

        Example:
            >>> total = handler.get_calculation_count()
            >>> calls = handler.get_calculation_count('call')
            >>> puts = handler.get_calculation_count('put')
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if option_type:
                cursor.execute("""
                    SELECT COUNT(*) FROM inputs
                    WHERE option_type = ?
                """, (option_type,))
            else:
                cursor.execute("SELECT COUNT(*) FROM inputs")

            return cursor.fetchone()[0]

        finally:
            conn.close()

    def get_statistics(self) -> Dict:
        """
        Get summary statistics about stored calculations.

        Returns:
            Dictionary containing:
                - total_calculations: Total number of calculations
                - call_count: Number of call options
                - put_count: Number of put options
                - avg_option_price: Average option price
                - avg_volatility: Average volatility used
                - date_range: Tuple of (earliest, latest) timestamps

        Example:
            >>> stats = handler.get_statistics()
            >>> print(f"Total calculations: {stats['total_calculations']}")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get counts
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN option_type = 'call' THEN 1 ELSE 0 END) as calls,
                    SUM(CASE WHEN option_type = 'put' THEN 1 ELSE 0 END) as puts
                FROM inputs
            """)
            counts = cursor.fetchone()

            # Get averages
            cursor.execute("""
                SELECT
                    AVG(o.option_price) as avg_price,
                    AVG(i.volatility) as avg_vol,
                    MIN(i.timestamp) as earliest,
                    MAX(i.timestamp) as latest
                FROM inputs i
                LEFT JOIN outputs o ON i.calculation_id = o.calculation_id
            """)
            averages = cursor.fetchone()

            return {
                'total_calculations': counts[0],
                'call_count': counts[1] or 0,
                'put_count': counts[2] or 0,
                'avg_option_price': averages[0] if averages[0] else 0,
                'avg_volatility': averages[1] if averages[1] else 0,
                'date_range': (averages[2], averages[3])
            }

        finally:
            conn.close()

    def search_calculations(self,
                           spot_min: Optional[float] = None,
                           spot_max: Optional[float] = None,
                           strike_min: Optional[float] = None,
                           strike_max: Optional[float] = None,
                           option_type: Optional[str] = None,
                           limit: int = 100) -> List[Dict]:
        """
        Search calculations with flexible filtering criteria.

        Args:
            spot_min: Minimum spot price
            spot_max: Maximum spot price
            strike_min: Minimum strike price
            strike_max: Maximum strike price
            option_type: Filter by option type
            limit: Maximum number of results

        Returns:
            List of matching calculations

        Example:
            >>> # Find all ATM call options (S ≈ K)
            >>> results = handler.search_calculations(
            ...     spot_min=95, spot_max=105,
            ...     strike_min=95, strike_max=105,
            ...     option_type='call'
            ... )
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if spot_min is not None:
            conditions.append("spot_price >= ?")
            params.append(spot_min)

        if spot_max is not None:
            conditions.append("spot_price <= ?")
            params.append(spot_max)

        if strike_min is not None:
            conditions.append("strike_price >= ?")
            params.append(strike_min)

        if strike_max is not None:
            conditions.append("strike_price <= ?")
            params.append(strike_max)

        if option_type:
            conditions.append("option_type = ?")
            params.append(option_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        try:
            cursor.execute(f"""
                SELECT * FROM calculation_history
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        finally:
            conn.close()

    def clear_all_calculations(self) -> int:
        """
        Delete all calculations from the database.

        Returns:
            Number of calculations deleted

        Warning:
            This operation cannot be undone!
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM inputs")
            count = cursor.fetchone()[0]

            cursor.execute("DELETE FROM inputs")
            conn.commit()

            return count

        finally:
            conn.close()

    def backup_database(self, backup_path: str) -> None:
        """
        Create a backup copy of the database.

        Args:
            backup_path: Path for the backup file

        Example:
            >>> handler.backup_database('backups/option_pricing_backup.db')
        """
        import shutil

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        # Copy database file
        shutil.copy2(self.db_path, backup_path)

    def export_to_csv(self, csv_path: str, limit: Optional[int] = None) -> None:
        """
        Export calculation history to CSV file.

        Args:
            csv_path: Path for the CSV file
            limit: Maximum number of records to export (None for all)

        Example:
            >>> handler.export_to_csv('exports/calculations.csv')
        """
        df = self.get_calculation_history_df(limit=limit or 999999)

        if df.empty:
            raise ValueError("No calculations to export")

        # Ensure export directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        df.to_csv(csv_path, index=False)


# Convenience function for quick database access
def get_handler(db_path: str = "option_pricing.db") -> DatabaseHandler:
    """
    Get a DatabaseHandler instance and ensure database is initialized.

    Args:
        db_path: Path to database file

    Returns:
        Initialized DatabaseHandler instance

    Example:
        >>> handler = get_handler()
        >>> history = handler.get_calculation_history(limit=5)
    """
    handler = DatabaseHandler(db_path)
    handler.initialize_db()
    return handler
