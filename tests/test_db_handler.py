"""
Unit tests for database handler.

This test suite validates:
- Database initialization
- CRUD operations
- Data integrity and constraints
- Query functionality
- Error handling

Run with: pytest tests/test_db_handler.py -v
"""

import pytest
import os
import tempfile
import time
from datetime import datetime
from db.db_handler import DatabaseHandler, get_handler


@pytest.fixture
def temp_db():
    """
    Create a temporary database for testing.

    Yields:
        Path to temporary database file

    Cleanup:
        Removes the temporary database after test completes
    """
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def handler(temp_db):
    """
    Create and initialize a DatabaseHandler for testing.

    Args:
        temp_db: Temporary database path from temp_db fixture

    Returns:
        Initialized DatabaseHandler instance
    """
    h = DatabaseHandler(temp_db)
    h.initialize_db()
    return h


@pytest.fixture
def sample_inputs():
    """Sample input parameters for testing."""
    return {
        'spot_price': 100.0,
        'strike_price': 100.0,
        'time_to_maturity': 1.0,
        'volatility': 0.2,
        'risk_free_rate': 0.05,
        'option_type': 'call'
    }


@pytest.fixture
def sample_outputs():
    """Sample output values for testing."""
    return {
        'option_price': 10.45,
        'delta': 0.637,
        'gamma': 0.019,
        'vega': 0.375,
        'theta': -0.012,
        'rho': 0.532
    }


class TestDatabaseInitialization:
    """Test database initialization and setup."""

    def test_initialize_db_creates_file(self, temp_db):
        """Test that initialize_db creates the database file."""
        handler = DatabaseHandler(temp_db)

        # Remove file if it exists
        if os.path.exists(temp_db):
            os.remove(temp_db)

        handler.initialize_db()

        assert os.path.exists(temp_db), "Database file should be created"

    def test_initialize_db_creates_tables(self, handler):
        """Test that tables are created correctly."""
        conn = handler._get_connection()
        cursor = conn.cursor()

        # Check inputs table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='inputs'
        """)
        assert cursor.fetchone() is not None, "inputs table should exist"

        # Check outputs table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='outputs'
        """)
        assert cursor.fetchone() is not None, "outputs table should exist"

        conn.close()

    def test_initialize_db_creates_view(self, handler):
        """Test that calculation_history view is created."""
        conn = handler._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='view' AND name='calculation_history'
        """)
        assert cursor.fetchone() is not None, "calculation_history view should exist"

        conn.close()

    def test_initialize_db_idempotent(self, handler):
        """Test that calling initialize_db multiple times is safe."""
        # Should not raise an error
        handler.initialize_db()
        handler.initialize_db()


class TestSaveCalculation:
    """Test saving calculations to the database."""

    def test_save_calculation_returns_id(self, handler, sample_inputs, sample_outputs):
        """Test that save_calculation returns a valid UUID."""
        calc_id = handler.save_calculation(sample_inputs, sample_outputs)

        assert calc_id is not None
        assert len(calc_id) == 36, "UUID should be 36 characters long"
        assert calc_id.count('-') == 4, "UUID should have 4 dashes"

    def test_save_calculation_stores_inputs(self, handler, sample_inputs, sample_outputs):
        """Test that input parameters are stored correctly."""
        calc_id = handler.save_calculation(sample_inputs, sample_outputs)

        result = handler.get_calculation_by_id(calc_id)

        assert result is not None
        assert result['spot_price'] == sample_inputs['spot_price']
        assert result['strike_price'] == sample_inputs['strike_price']
        assert result['time_to_maturity'] == sample_inputs['time_to_maturity']
        assert result['volatility'] == sample_inputs['volatility']
        assert result['risk_free_rate'] == sample_inputs['risk_free_rate']
        assert result['option_type'] == sample_inputs['option_type']

    def test_save_calculation_stores_outputs(self, handler, sample_inputs, sample_outputs):
        """Test that output values are stored correctly."""
        calc_id = handler.save_calculation(sample_inputs, sample_outputs)

        result = handler.get_calculation_by_id(calc_id)

        assert result is not None
        assert abs(result['option_price'] - sample_outputs['option_price']) < 1e-10
        assert abs(result['delta'] - sample_outputs['delta']) < 1e-10
        assert abs(result['gamma'] - sample_outputs['gamma']) < 1e-10
        assert abs(result['vega'] - sample_outputs['vega']) < 1e-10
        assert abs(result['theta'] - sample_outputs['theta']) < 1e-10
        assert abs(result['rho'] - sample_outputs['rho']) < 1e-10

    def test_save_calculation_stores_timestamp(self, handler, sample_inputs, sample_outputs):
        """Test that timestamp is automatically set."""
        calc_id = handler.save_calculation(sample_inputs, sample_outputs)

        result = handler.get_calculation_by_id(calc_id)

        assert result is not None
        assert result['timestamp'] is not None

    def test_save_multiple_calculations(self, handler, sample_inputs, sample_outputs):
        """Test saving multiple calculations."""
        id1 = handler.save_calculation(sample_inputs, sample_outputs)

        # Modify inputs for second calculation
        inputs2 = sample_inputs.copy()
        inputs2['strike_price'] = 105.0
        id2 = handler.save_calculation(inputs2, sample_outputs)

        assert id1 != id2, "Each calculation should have unique ID"

        # Verify both exist
        assert handler.get_calculation_by_id(id1) is not None
        assert handler.get_calculation_by_id(id2) is not None


class TestDataConstraints:
    """Test database constraints and data integrity."""

    def test_negative_spot_price_rejected(self, handler, sample_inputs, sample_outputs):
        """Test that negative spot price violates constraint."""
        inputs = sample_inputs.copy()
        inputs['spot_price'] = -100.0

        with pytest.raises(ValueError):
            handler.save_calculation(inputs, sample_outputs)

    def test_negative_strike_price_rejected(self, handler, sample_inputs, sample_outputs):
        """Test that negative strike price violates constraint."""
        inputs = sample_inputs.copy()
        inputs['strike_price'] = -100.0

        with pytest.raises(ValueError):
            handler.save_calculation(inputs, sample_outputs)

    def test_negative_time_rejected(self, handler, sample_inputs, sample_outputs):
        """Test that negative time to maturity violates constraint."""
        inputs = sample_inputs.copy()
        inputs['time_to_maturity'] = -1.0

        with pytest.raises(ValueError):
            handler.save_calculation(inputs, sample_outputs)

    def test_invalid_option_type_rejected(self, handler, sample_inputs, sample_outputs):
        """Test that invalid option type violates constraint."""
        inputs = sample_inputs.copy()
        inputs['option_type'] = 'invalid'

        with pytest.raises(ValueError):
            handler.save_calculation(inputs, sample_outputs)

    def test_foreign_key_constraint(self, handler):
        """Test that outputs cannot exist without corresponding inputs."""
        conn = handler._get_connection()
        cursor = conn.cursor()

        # Try to insert outputs without inputs
        with pytest.raises(Exception):  # SQLite integrity error
            cursor.execute("""
                INSERT INTO outputs (
                    calculation_id, option_price, delta, gamma, vega, theta, rho
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('non-existent-id', 10.0, 0.5, 0.01, 0.3, -0.01, 0.4))
            conn.commit()

        conn.close()


class TestGetCalculationById:
    """Test retrieving specific calculations."""

    def test_get_existing_calculation(self, handler, sample_inputs, sample_outputs):
        """Test retrieving an existing calculation."""
        calc_id = handler.save_calculation(sample_inputs, sample_outputs)

        result = handler.get_calculation_by_id(calc_id)

        assert result is not None
        assert result['calculation_id'] == calc_id

    def test_get_nonexistent_calculation(self, handler):
        """Test that getting non-existent calculation returns None."""
        result = handler.get_calculation_by_id('non-existent-id')

        assert result is None


class TestGetCalculationHistory:
    """Test retrieving calculation history."""

    def test_get_history_empty_database(self, handler):
        """Test getting history from empty database."""
        history = handler.get_calculation_history()

        assert history == []

    def test_get_history_returns_list(self, handler, sample_inputs, sample_outputs):
        """Test that history returns a list of dictionaries."""
        handler.save_calculation(sample_inputs, sample_outputs)

        history = handler.get_calculation_history()

        assert isinstance(history, list)
        assert len(history) > 0
        assert isinstance(history[0], dict)

    def test_get_history_respects_limit(self, handler, sample_inputs, sample_outputs):
        """Test that limit parameter works correctly."""
        # Save 5 calculations
        for i in range(5):
            inputs = sample_inputs.copy()
            inputs['strike_price'] = 100.0 + i
            handler.save_calculation(inputs, sample_outputs)

        history = handler.get_calculation_history(limit=3)

        assert len(history) == 3

    def test_get_history_ordered_by_recent(self, handler, sample_inputs, sample_outputs):
        """Test that history is ordered by most recent first."""
        # Save calculations with different strikes
        strikes = []
        for strike in [100, 105, 110]:
            inputs = sample_inputs.copy()
            inputs['strike_price'] = float(strike)
            handler.save_calculation(inputs, sample_outputs)
            strikes.append(strike)
            time.sleep(1.1)  # Ensure different timestamps (SQLite uses second precision)

        history = handler.get_calculation_history()

        # Most recent should have strike 110
        assert history[0]['strike_price'] == 110.0
        # Verify all three are present in reverse order
        assert [h['strike_price'] for h in history] == [110.0, 105.0, 100.0]

    def test_get_history_filter_by_type(self, handler, sample_inputs, sample_outputs):
        """Test filtering history by option type."""
        # Save calls
        for i in range(3):
            handler.save_calculation(sample_inputs, sample_outputs)

        # Save puts
        put_inputs = sample_inputs.copy()
        put_inputs['option_type'] = 'put'
        for i in range(2):
            handler.save_calculation(put_inputs, sample_outputs)

        call_history = handler.get_calculation_history(option_type='call')
        put_history = handler.get_calculation_history(option_type='put')

        assert len(call_history) == 3
        assert len(put_history) == 2

        # Verify all are correct type
        assert all(h['option_type'] == 'call' for h in call_history)
        assert all(h['option_type'] == 'put' for h in put_history)


class TestDeleteCalculation:
    """Test deleting calculations."""

    def test_delete_existing_calculation(self, handler, sample_inputs, sample_outputs):
        """Test deleting an existing calculation."""
        calc_id = handler.save_calculation(sample_inputs, sample_outputs)

        success = handler.delete_calculation(calc_id)

        assert success is True
        assert handler.get_calculation_by_id(calc_id) is None

    def test_delete_nonexistent_calculation(self, handler):
        """Test deleting non-existent calculation returns False."""
        success = handler.delete_calculation('non-existent-id')

        assert success is False

    def test_delete_cascade(self, handler, sample_inputs, sample_outputs):
        """Test that deleting inputs also deletes outputs (CASCADE)."""
        calc_id = handler.save_calculation(sample_inputs, sample_outputs)

        # Verify outputs exist
        result = handler.get_calculation_by_id(calc_id)
        assert result['option_price'] is not None

        # Delete
        handler.delete_calculation(calc_id)

        # Verify both inputs and outputs are gone
        conn = handler._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM outputs WHERE calculation_id = ?", (calc_id,))
        assert cursor.fetchone() is None

        conn.close()


class TestCalculationCount:
    """Test counting calculations."""

    def test_count_empty_database(self, handler):
        """Test counting in empty database."""
        count = handler.get_calculation_count()

        assert count == 0

    def test_count_total(self, handler, sample_inputs, sample_outputs):
        """Test counting total calculations."""
        for i in range(5):
            handler.save_calculation(sample_inputs, sample_outputs)

        count = handler.get_calculation_count()

        assert count == 5

    def test_count_by_type(self, handler, sample_inputs, sample_outputs):
        """Test counting by option type."""
        # Save 3 calls
        for i in range(3):
            handler.save_calculation(sample_inputs, sample_outputs)

        # Save 2 puts
        put_inputs = sample_inputs.copy()
        put_inputs['option_type'] = 'put'
        for i in range(2):
            handler.save_calculation(put_inputs, sample_outputs)

        total = handler.get_calculation_count()
        calls = handler.get_calculation_count('call')
        puts = handler.get_calculation_count('put')

        assert total == 5
        assert calls == 3
        assert puts == 2


class TestStatistics:
    """Test statistics functionality."""

    def test_statistics_empty_database(self, handler):
        """Test statistics on empty database."""
        stats = handler.get_statistics()

        assert stats['total_calculations'] == 0
        assert stats['call_count'] == 0
        assert stats['put_count'] == 0

    def test_statistics_with_data(self, handler, sample_inputs, sample_outputs):
        """Test statistics with data."""
        # Save some calculations
        for i in range(3):
            handler.save_calculation(sample_inputs, sample_outputs)

        put_inputs = sample_inputs.copy()
        put_inputs['option_type'] = 'put'
        handler.save_calculation(put_inputs, sample_outputs)

        stats = handler.get_statistics()

        assert stats['total_calculations'] == 4
        assert stats['call_count'] == 3
        assert stats['put_count'] == 1
        assert stats['avg_option_price'] > 0
        assert stats['avg_volatility'] > 0


class TestSearchCalculations:
    """Test search functionality."""

    def test_search_by_spot_range(self, handler, sample_inputs, sample_outputs):
        """Test searching by spot price range."""
        # Save calculations with different spot prices
        for spot in [90, 100, 110, 120]:
            inputs = sample_inputs.copy()
            inputs['spot_price'] = float(spot)
            handler.save_calculation(inputs, sample_outputs)

        results = handler.search_calculations(spot_min=95, spot_max=115)

        assert len(results) == 2  # 100 and 110
        assert all(95 <= r['spot_price'] <= 115 for r in results)

    def test_search_by_strike_range(self, handler, sample_inputs, sample_outputs):
        """Test searching by strike price range."""
        for strike in [90, 100, 110]:
            inputs = sample_inputs.copy()
            inputs['strike_price'] = float(strike)
            handler.save_calculation(inputs, sample_outputs)

        results = handler.search_calculations(strike_min=95, strike_max=105)

        assert len(results) == 1
        assert results[0]['strike_price'] == 100.0

    def test_search_by_option_type(self, handler, sample_inputs, sample_outputs):
        """Test searching by option type."""
        handler.save_calculation(sample_inputs, sample_outputs)

        put_inputs = sample_inputs.copy()
        put_inputs['option_type'] = 'put'
        handler.save_calculation(put_inputs, sample_outputs)

        results = handler.search_calculations(option_type='call')

        assert len(results) == 1
        assert results[0]['option_type'] == 'call'


class TestDataFrameExport:
    """Test pandas DataFrame export functionality."""

    def test_get_history_df_empty(self, handler):
        """Test getting empty DataFrame."""
        df = handler.get_calculation_history_df()

        assert df.empty

    def test_get_history_df_with_data(self, handler, sample_inputs, sample_outputs):
        """Test getting DataFrame with data."""
        handler.save_calculation(sample_inputs, sample_outputs)

        df = handler.get_calculation_history_df()

        assert not df.empty
        assert len(df) == 1
        assert 'spot_price' in df.columns
        assert 'option_price' in df.columns


class TestClearAllCalculations:
    """Test clearing all calculations."""

    def test_clear_all(self, handler, sample_inputs, sample_outputs):
        """Test clearing all calculations."""
        # Save some data
        for i in range(5):
            handler.save_calculation(sample_inputs, sample_outputs)

        count = handler.clear_all_calculations()

        assert count == 5
        assert handler.get_calculation_count() == 0


class TestConvenienceFunction:
    """Test convenience function."""

    def test_get_handler(self, temp_db):
        """Test get_handler convenience function."""
        handler = get_handler(temp_db)

        assert isinstance(handler, DatabaseHandler)
        assert handler.db_path == temp_db

        # Should be initialized
        assert handler.get_calculation_count() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
