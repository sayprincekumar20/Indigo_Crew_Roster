# app/data/load_data.py
import pandas as pd
import os
from datetime import datetime, date
from app.core.config import settings
from .database import db_manager
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.data_path = settings.data_path
        self.crew_df = None
        self.flights_df = None
        self.aircraft_df = None
        self.airports_df = None
        self.crew_preferences_df = None
        self.crew_qualifications_df = None
        self.historical_rosters_df = None
        self.disruption_data_df = None
        
 # app/data/load_data.py (update the load_all_data method)
    def load_all_data(self):
        """Load all data from SQLite database with proper date handling"""
        try:
            # Import CSV data to SQLite if not already done
            db_manager.import_all_data(self.data_path)
            
            # Load data from SQLite into DataFrames
            self.crew_df = pd.read_sql_query("SELECT * FROM crew_members", db_manager.conn)
            self.flights_df = pd.read_sql_query("SELECT * FROM flights", db_manager.conn)
            self.aircraft_df = pd.read_sql_query("SELECT * FROM aircraft", db_manager.conn)
            self.airports_df = pd.read_sql_query("SELECT * FROM airports", db_manager.conn)
            self.crew_preferences_df = pd.read_sql_query("SELECT * FROM crew_preferences", db_manager.conn)
            self.crew_qualifications_df = pd.read_sql_query("SELECT * FROM crew_qualifications", db_manager.conn)
            self.historical_rosters_df = pd.read_sql_query("SELECT * FROM historical_rosters", db_manager.conn)
            self.disruption_data_df = pd.read_sql_query("SELECT * FROM disruption_data", db_manager.conn)
            
            # Convert date columns to datetime64[ns] consistently
            date_columns = ['Date', 'Leave_Start', 'Leave_End', 'Maintenance_Due_Date', 'Validity_Start', 'Validity_End']
            
            for df_name in ['crew_df', 'flights_df', 'aircraft_df', 'crew_qualifications_df', 'historical_rosters_df', 'disruption_data_df']:
                df = getattr(self, df_name)
                for col in date_columns:
                    if col in df.columns:
                        # Convert to datetime64[ns] and ensure timezone-naive
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
            
            # Convert time columns
            if 'Scheduled_Departure_UTC' in self.flights_df.columns:
                self.flights_df['Scheduled_Departure_UTC'] = pd.to_datetime(self.flights_df['Scheduled_Departure_UTC'], format='%H:%M', errors='coerce').dt.time
            if 'Scheduled_Arrival_UTC' in self.flights_df.columns:
                self.flights_df['Scheduled_Arrival_UTC'] = pd.to_datetime(self.flights_df['Scheduled_Arrival_UTC'], format='%H:%M', errors='coerce').dt.time
            
            # Convert time columns for historical rosters
            time_columns = ['Duty_Start', 'Duty_End']
            for col in time_columns:
                if col in self.historical_rosters_df.columns:
                    self.historical_rosters_df[col] = pd.to_datetime(self.historical_rosters_df[col], format='%H:%M', errors='coerce').dt.time
            
            # Convert time columns for airports
            curfew_columns = ['Curfew_Start', 'Curfew_End']
            for col in curfew_columns:
                if col in self.airports_df.columns:
                    self.airports_df[col] = pd.to_datetime(self.airports_df[col], format='%H:%M', errors='coerce').dt.time
            
            logger.info("All data loaded successfully from SQLite")
            logger.info(f"Crew DataFrame date columns: {self.crew_df[['Leave_Start', 'Leave_End']].dtypes}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def get_flights_for_date_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get flights for a specific date range with robust date comparison
        """
        if self.flights_df is None or self.flights_df.empty:
            logger.warning("Flights DataFrame is None or empty")
            return pd.DataFrame()
        
        # Ensure Date column is properly formatted as datetime
        if not pd.api.types.is_datetime64_any_dtype(self.flights_df['Date']):
            try:
                self.flights_df['Date'] = pd.to_datetime(self.flights_df['Date'], errors='coerce')
                # Drop rows with invalid dates
                self.flights_df = self.flights_df.dropna(subset=['Date'])
            except Exception as e:
                logger.error(f"Error converting Date column to datetime: {e}")
                return pd.DataFrame()
        
        # Convert input dates to pandas Timestamp for consistent comparison
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Filter flights for the date range
        mask = (self.flights_df['Date'] >= start_dt) & (self.flights_df['Date'] <= end_dt)
        
        result = self.flights_df.loc[mask].copy()
        logger.info(f"Returning {len(result)} flights for date range {start_date} to {end_date}")
        
        return result    


# Create a global data loader instance
data_loader = DataLoader()