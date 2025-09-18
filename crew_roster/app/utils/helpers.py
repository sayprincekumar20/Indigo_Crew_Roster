# app/utils/helpers.py
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta ,date
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from dateutil import parser

logger = logging.getLogger(__name__)


def parse_time(time_str: str) -> time:
    """
    Parse time string to datetime.time object
    Supports various time formats including ISO format
    """
    if isinstance(time_str, time):
        return time_str
        
    if pd.isna(time_str) or time_str is None:
        return None
        
    try:
        if isinstance(time_str, str):
            # Handle ISO format (e.g., "2023-10-01T08:00:00")
            if 'T' in time_str:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                return dt.time()
            
            # Handle HH:MM:SS format
            elif ':' in time_str:
                parts = time_str.split(':')
                if len(parts) >= 2:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2]) if len(parts) > 2 else 0
                    return time(hour=hours, minute=minutes, second=seconds)
            
            # Handle other formats using dateutil
            else:
                dt = parser.parse(time_str)
                return dt.time()
                
        elif isinstance(time_str, datetime):
            return time_str.time()
        elif isinstance(time_str, pd.Timestamp):
            return time_str.time()
        else:
            return None
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Could not parse time string: {time_str}, error: {e}")
        return None
    
def parse_duration(duration_str: str) -> timedelta:
    """
    Parse duration string (HH:MM) to timedelta
    """
    if pd.isna(duration_str) or duration_str is None:
        return timedelta(0)
        
    try:
        if isinstance(duration_str, str) and ':' in duration_str:
            hours, minutes = map(int, duration_str.split(':'))
            return timedelta(hours=hours, minutes=minutes)
        elif isinstance(duration_str, (int, float)):
            return timedelta(hours=duration_str)
        else:
            return timedelta(0)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse duration: {duration_str}, error: {e}")
        return timedelta(0)



def calculate_duty_hours(start_time, end_time) -> float:
    """
    Calculate duty hours between two time/datetime/string objects
    Handles overnight duties and multiple input types with robust parsing
    """
    if not start_time or not end_time:
        return 0.0
    
    # Parse both inputs to datetime objects
    start_dt = parse_datetime_from_string(start_time)
    end_dt = parse_datetime_from_string(end_time)
    
    if not start_dt or not end_dt:
        logger.warning(f"Could not parse datetime values: start={start_time}, end={end_time}")
        return 0.0
    
    # Calculate duration
    duration = end_dt - start_dt
    return round(duration.total_seconds() / 3600.0, 2)

def is_night_duty(start_time: time, end_time: time) -> bool:
    """
    Check if duty occurs during night hours (10 PM to 6 AM)
    """
    if not start_time or not end_time:
        return False
        
    # Convert to minutes since midnight for easier comparison
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    
    # Night time boundaries (10 PM to 6 AM)
    night_start = 22 * 60  # 10:00 PM
    night_end = 6 * 60     # 6:00 AM
    
    # Handle overnight duties
    if end_minutes < start_minutes:  # Duty spans midnight
        return (start_minutes >= night_start or end_minutes <= night_end)
    else:
        return (start_minutes >= night_start and start_minutes < 24 * 60) or \
               (end_minutes > 0 and end_minutes <= night_end)

def get_aircraft_type_crew(crew_df: pd.DataFrame, aircraft_type: str) -> List[str]:
    """
    Get list of crew members qualified for a specific aircraft type
    """
    qualified_crew = []
    
    for _, crew in crew_df.iterrows():
        if pd.notna(crew.get('Aircraft_Type_License')):
            licensed_types = [t.strip() for t in str(crew['Aircraft_Type_License']).split(',')]
            if aircraft_type in licensed_types:
                qualified_crew.append(crew['Crew_ID'])
    
    return qualified_crew

def get_crew_preferences(crew_id: str, preferences_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get all preferences for a specific crew member
    """
    crew_prefs = preferences_df[preferences_df['Crew_ID'] == crew_id]
    preferences = {}
    
    for _, pref in crew_prefs.iterrows():
        pref_type = pref['Preference_Type']
        pref_detail = pref['Preference_Detail']
        priority = pref['Priority']
        
        if pref_type not in preferences:
            preferences[pref_type] = []
        
        preferences[pref_type].append({
            'detail': pref_detail,
            'priority': priority
        })
    
    return preferences


def calculate_fairness_metrics(roster_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate fairness metrics for the roster
    """
    if roster_df.empty:
        return {
            'total_assignments': 0,
            'crew_utilization': 0.0,
            'fairness_score': 0.0,
            'max_duty_hours': 0.0,
            'min_duty_hours': 0.0,
            'avg_duty_hours': 0.0,
            'std_dev_duty_hours': 0.0
        }
    
    # Calculate duty hours per crew
    crew_hours = {}
    for _, row in roster_df.iterrows():
        crew_id = row['Crew_ID']
        
        # Get duty start and end times
        duty_start = row.get('Duty_Start')
        duty_end = row.get('Duty_End')
        
        # Calculate duty hours (function now handles strings, datetime, and time objects)
        duty_hours = calculate_duty_hours(duty_start, duty_end)
        
        if crew_id not in crew_hours:
            crew_hours[crew_id] = 0.0
        crew_hours[crew_id] += duty_hours
    
    hours = list(crew_hours.values())
    
    if not hours:
        return {
            'total_assignments': len(roster_df),
            'crew_utilization': 0.0,
            'fairness_score': 0.0,
            'max_duty_hours': 0.0,
            'min_duty_hours': 0.0,
            'avg_duty_hours': 0.0,
            'std_dev_duty_hours': 0.0
        }
    
    # Calculate metrics
    max_hours = max(hours)
    min_hours = min(hours)
    avg_hours = np.mean(hours)
    std_dev = np.std(hours)
    
    # Fairness score (0-100, higher is better)
    if max_hours > 0:
        fairness_score = 100 * (1 - (std_dev / max_hours))
    else:
        fairness_score = 100.0
    
    return {
        'total_assignments': len(roster_df),
        'crew_utilization': len(crew_hours) / len(roster_df['Crew_ID'].unique()) * 100 if len(roster_df) > 0 else 0,
        'fairness_score': fairness_score,
        'max_duty_hours': max_hours,
        'min_duty_hours': min_hours,
        'avg_duty_hours': avg_hours,
        'std_dev_duty_hours': std_dev
  }




def format_violations_for_display(violations: List[str]) -> List[Dict[str, str]]:
    """
    Format violations for better display in API responses
    """
    formatted = []
    
    for violation in violations:
        if "RAG Validation" in violation:
            # RAG violations have specific format
            parts = violation.split(": ", 1)
            if len(parts) == 2:
                formatted.append({
                    'type': 'RAG',
                    'category': parts[0].replace("RAG Validation - ", ""),
                    'message': parts[1]
                })
        else:
            # Standard violations
            formatted.append({
                'type': 'Standard',
                'category': 'DGCA Compliance',
                'message': violation
            })
    
    return formatted


def save_roster_to_json(roster_data: List[Dict], filename: str) -> bool:
    """
    Save roster data to JSON file
    """
    try:
        with open(filename, 'w') as f:
            json.dump(roster_data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving roster to JSON: {e}")
        return False

def load_roster_from_json(filename: str) -> List[Dict]:
    """
    Load roster data from JSON file
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading roster from JSON: {e}")
        return []
    
    
def convert_dates_to_strings(data: List[Dict]) -> List[Dict]:
    """
    Convert all date and datetime objects in a list of dictionaries to strings
    for JSON serialization
    """
    serializable_data = []
    
    for item in data:
        serializable_item = {}
        for key, value in item.items():
            if isinstance(value, (date, datetime)):
                serializable_item[key] = value.isoformat()
            elif isinstance(value, pd.Timestamp):
                serializable_item[key] = value.isoformat()
            else:
                serializable_item[key] = value
        serializable_data.append(serializable_item)
    
    return serializable_data

# app/utils/helpers.py (add this function)
def parse_datetime_from_string(datetime_str):
    """Parse datetime from string, handling various formats"""
    if datetime_str is None:
        return None
    
    if isinstance(datetime_str, (datetime, pd.Timestamp)):
        return datetime_str
    
    if isinstance(datetime_str, str):
        try:
            # Try ISO format first
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try other common formats
                return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
                except ValueError:
                    return None
    
    return None
def preprocess_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess datetime columns in a DataFrame to ensure proper datetime types
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    # List of potential datetime columns
    datetime_columns = ['Duty_Start', 'Duty_End', 'Date']
    
    for col in datetime_columns:
        if col in df_copy.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            
            # Convert to timezone-naive if it's timezone-aware
            if hasattr(df_copy[col].dtype, 'tz') and df_copy[col].dtype.tz is not None:
                df_copy[col] = df_copy[col].dt.tz_convert(None)
    
    return df_copy
def safe_datetime_subtraction(dt1, dt2):
    """
    Safely subtract two datetime objects, handling various input types
    Returns timedelta or None if operation fails
    """
    try:
        # Parse both inputs to datetime objects
        dt1_parsed = parse_datetime_from_string(dt1)
        dt2_parsed = parse_datetime_from_string(dt2)
        
        if dt1_parsed and dt2_parsed:
            return dt1_parsed - dt2_parsed
        else:
            return None
    except Exception as e:
        logger.warning(f"Datetime subtraction failed: {e}")
        return None
    
    
def validate_datetime_data(df: pd.DataFrame) -> bool:
    """
    Validate that datetime columns contain proper datetime values
    Returns True if valid, False otherwise
    """
    if df is None or df.empty:
        return True
    
    datetime_columns = ['Duty_Start', 'Duty_End']
    
    for col in datetime_columns:
        if col in df.columns:
            # Check if column contains strings that should be datetime
            if df[col].dtype == 'object':
                # Sample some values to check
                sample_values = df[col].head(5).tolist()
                for val in sample_values:
                    if val and isinstance(val, str):
                        parsed = parse_datetime_from_string(val)
                        if not parsed:
                            logger.warning(f"Invalid datetime format in column {col}: {val}")
                            return False
    
    return True


# app/utils/disruption_helpers.py
import pandas as pd
from typing import Dict, List

# app/utils/disruption_helpers.py
from app.data.database import db_manager
import json

async def get_current_roster():
    """Get current roster from database"""
    try:
        # This should return the actual roster data, not a coroutine
        result = db_manager.execute_query("SELECT roster_data FROM generated_rosters ORDER BY created_at DESC LIMIT 1")
        row = result.fetchone()
        if row:
            return json.loads(row[0])
        return None
    except Exception as e:
        logger.error(f"Error getting current roster: {e}")
        return None

async def validate_replacement(replacement: Dict) -> Dict:
    """Validate a replacement suggestion"""
    issues = []
    
    # Check if original crew exists (accept both original_crew and disrupted_crew_id)
    original_crew = replacement.get('original_crew') or replacement.get('disrupted_crew_id')
    if not original_crew:
        issues.append("Original crew ID is required")
    
    # Check if replacements are provided
    replacements = replacement.get('replacements', [])
    if not replacements:
        issues.append("No replacements specified")
    
    # Validate each replacement
    for rep in replacements:
        flight_number = rep.get('flight_number')
        new_crew = rep.get('new_crew')
        
        if not flight_number or flight_number.strip() == '':
            issues.append("Flight number is required for replacement")
        if not new_crew or new_crew.strip() == '':
            issues.append("New crew ID is required for replacement")
    
    return {"valid": len(issues) == 0, "issues": issues}

async def apply_replacement_to_roster(replacement: Dict):
    """Apply replacement to roster"""
    try:
        # Get current roster
        current_roster = await get_current_roster()
        if not current_roster:
            return {"id": None, "error": "No current roster found"}
        
        # Validate roster structure
        if not isinstance(current_roster, list):
            return {"id": None, "error": "Invalid roster format"}
        
        # Get original crew ID
        original_crew = replacement.get('original_crew', '').strip()
        replacements = replacement.get('replacements', [])
        
        # Apply each replacement
        for rep in replacements:
            flight_number = rep.get('flight_number', '').strip()
            new_crew = rep.get('new_crew', '').strip()
            
            # Find the flight in the roster
            flight_found = False
            for flight in current_roster:
                # Ensure flight has required properties
                if not isinstance(flight, dict) or 'Flight_Number' not in flight:
                    continue
                    
                if flight.get('Flight_Number') == flight_number:
                    flight_found = True
                    
                    # Ensure Crew_Members exists and is a list
                    if 'Crew_Members' not in flight or not isinstance(flight['Crew_Members'], list):
                        return {"id": None, "error": f"Invalid crew data for flight {flight_number}"}
                    
                    # Find and replace the crew member
                    crew_updated = False
                    for crew_member in flight.get('Crew_Members', []):
                        if not isinstance(crew_member, dict) or 'Crew_ID' not in crew_member:
                            continue
                            
                        if crew_member.get('Crew_ID') == original_crew:
                            crew_member['Crew_ID'] = new_crew
                            crew_updated = True
                            break
                    
                    if not crew_updated:
                        return {"id": None, "error": f"Crew member {original_crew} not found in flight {flight_number}"}
                    break
            
            if not flight_found:
                return {"id": None, "error": f"Flight {flight_number} not found in roster"}
        
        return {"id": "updated_roster", "roster": current_roster}
        
    except Exception as e:
        logger.error(f"Error applying replacement: {e}")
        return {"id": None, "error": str(e)}

async def save_updated_roster(roster: Dict):
    """Save updated roster"""
    try:
        # Convert to JSON string
        roster_json = json.dumps(roster)
        
        # Save to database (update the latest roster or create new entry)
        result = db_manager.execute_query(
            """INSERT INTO generated_rosters 
            (roster_data, created_at) 
            VALUES (?, datetime('now'))""",
            (roster_json,)
        )
        db_manager.conn.commit()
        
        return True
    except Exception as e:
        logger.error(f"Error saving updated roster: {e}")
        return False