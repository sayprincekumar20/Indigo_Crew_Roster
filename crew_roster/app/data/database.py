# app/data/database.py
import sqlite3
import pandas as pd
import os
from pathlib import Path
from app.core.config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = settings.db_path):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize the database and create tables"""
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS crew_members (
                Crew_ID TEXT PRIMARY KEY,
                Name TEXT,
                Base TEXT,
                Rank TEXT,
                Qualification TEXT,
                Aircraft_Type_License TEXT,
                Leave_Start DATE,
                Leave_End DATE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flights (
                Date DATE,
                Flight_Number TEXT,
                Origin TEXT,
                Destination TEXT,
                Scheduled_Departure_UTC TIME,
                Scheduled_Arrival_UTC TIME,
                Aircraft_Type TEXT,
                Duration_HH_MM TEXT,
                PRIMARY KEY (Date, Flight_Number)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS aircraft (
                Aircraft_Type TEXT,
                Registration TEXT,
                Seating_Capacity INTEGER,
                Base TEXT,
                Maintenance_Due_Date DATE,
                Status TEXT,
                PRIMARY KEY (Registration)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS airports (
                Airport_Code TEXT PRIMARY KEY,
                Airport_Name TEXT,
                City TEXT,
                Time_Zone TEXT,
                Curfew_Start TIME,
                Curfew_End TIME
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS crew_preferences (
                Crew_ID TEXT,
                Preference_Type TEXT,
                Preference_Detail TEXT,
                Priority TEXT,
                FOREIGN KEY (Crew_ID) REFERENCES crew_members (Crew_ID)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS crew_qualifications (
                Crew_ID TEXT,
                License_Type TEXT,
                Aircraft_Type TEXT,
                Validity_Start DATE,
                Validity_End DATE,
                Check_Status TEXT,
                FOREIGN KEY (Crew_ID) REFERENCES crew_members (Crew_ID)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS historical_rosters (
                Date DATE,
                Crew_ID TEXT,
                Duty_Type TEXT,
                Flights TEXT,
                Duty_Start TIME,
                Duty_End TIME,
                Overtime_Minutes INTEGER,
                Standby_Used TEXT,
                FOREIGN KEY (Crew_ID) REFERENCES crew_members (Crew_ID)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS disruption_data (
                Date DATE,
                Disruption_Type TEXT,
                Affected_Flight TEXT,
                Affected_Crew TEXT,
                Original_Plan TEXT,
                New_Status TEXT,
                Reason TEXT,
                Impact_Duration_Minutes INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS generated_rosters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                start_date DATE,
                end_date DATE,
                fitness_score FLOAT,
                violation_count INTEGER,
                roster_data TEXT
            )
            """
        ]
        
        cursor = self.conn.cursor()
        for table in tables:
            cursor.execute(table)
        self.conn.commit()
    
    def import_csv_to_table(self, csv_path: str, table_name: str):
        """Import CSV data into SQLite table"""
        try:
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            logger.info(f"Imported {len(df)} records into {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error importing {csv_path} to {table_name}: {e}")
            return False
    
    def import_all_data(self, data_path: str):
        """Import all CSV data into database"""
        csv_mapping = {
            'crew_information.csv': 'crew_members',
            'flight_schedule.csv': 'flights',
            'aircraft_information.csv': 'aircraft',
            'airport_information.csv': 'airports',
            'crew_preferences.csv': 'crew_preferences',
            'crew_qualifications.csv': 'crew_qualifications',
            'historical_rosters.csv': 'historical_rosters',
            'disruption_data.csv': 'disruption_data'
        }
        
        for csv_file, table_name in csv_mapping.items():
            csv_path = os.path.join(data_path, csv_file)
            if os.path.exists(csv_path):
                self.import_csv_to_table(csv_path, table_name)
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a SQL query"""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor
    
    def fetch_all(self, query: str, params: tuple = None):
        """Fetch all results from a query"""
        cursor = self.execute_query(query, params)
        return cursor.fetchall()
    
    def fetch_one(self, query: str, params: tuple = None):
        """Fetch one result from a query"""
        cursor = self.execute_query(query, params)
        return cursor.fetchone()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Global database instance
db_manager = DatabaseManager()