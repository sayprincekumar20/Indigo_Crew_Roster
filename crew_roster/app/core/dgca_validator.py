# app/core/dgca_validator.py
from datetime import datetime, time, timedelta
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, time, timedelta
import pandas as pd
from typing import List, Dict, Tuple
from .rag_validator import RAGValidator
from app.utils.helpers import parse_time,parse_datetime_from_string,safe_datetime_subtraction, calculate_duty_hours, is_night_duty
import logging
from app.utils.constants import MinimumCrewRequirements


logger = logging.getLogger(__name__)

class DGCAValidator:
    def __init__(self):
        self.rules = self._initialize_rules()
        self.rag_validator = RAGValidator()
    
   
    
    # app/core/dgca_validator.py (update the _check_daily_limits method)
    def _check_daily_limits(self, schedule: pd.DataFrame, crew_info: pd.Series) -> List[str]:
        """Check daily flight time and FDP limits with proper date handling"""
        violations = []
        
        for date in schedule['Date'].unique():
            daily_schedule = schedule[schedule['Date'] == date]
            
            # Calculate total flight time and FDP
            total_flight_time = self._calculate_flight_time(daily_schedule)
            total_fdp = self._calculate_fdp(daily_schedule)
            
            # Check flight time limit
            if total_flight_time > self.rules["max_daily_flight_time"]:
                violations.append(
                    f"Crew {crew_info['Crew_ID']} exceeded daily flight time limit on {date}: "
                    f"{total_flight_time:.2f}h > {self.rules['max_daily_flight_time']}h"
                )
            
            # Check FDP limit based on reporting time
            if not daily_schedule.empty:
                first_duty = daily_schedule.iloc[0]
                reporting_time_str = first_duty.get('Duty_Start')
                
                # Convert string to datetime if needed
                if isinstance(reporting_time_str, str):
                    try:
                        reporting_time = datetime.fromisoformat(reporting_time_str)
                    except (ValueError, TypeError):
                        reporting_time = None
                else:
                    reporting_time = reporting_time_str
                
                if reporting_time:
                    fdp_limit = self._get_fdp_limit(reporting_time)
                    
                    if total_fdp > fdp_limit:
                        violations.append(
                            f"Crew {crew_info['Crew_ID']} exceeded FDP limit on {date}: "
                            f"{total_fdp:.2f}h > {fdp_limit}h (reporting at {reporting_time})"
                        )
        
        return violations
        
   
    
    def _calculate_fdp(self, schedule: pd.DataFrame) -> float:
        """Calculate Flight Duty Period with proper datetime handling for string dates"""
        if schedule.empty:
            return 0.0
        
        # Get all duty start and end times
        duty_starts = []
        duty_ends = []
        
        for _, row in schedule.iterrows():
            duty_start = row.get('Duty_Start')
            duty_end = row.get('Duty_End')
            
            # Convert string dates to datetime objects if needed
            if isinstance(duty_start, str):
                try:
                    duty_start = datetime.fromisoformat(duty_start)
                except (ValueError, TypeError):
                    duty_start = None

            if isinstance(duty_end, str):
                try:  
                    duty_end = datetime.fromisoformat(duty_end)
                except (ValueError, TypeError):
                    duty_end = None   

            if duty_start and duty_end:
                duty_starts.append(duty_start)
                duty_ends.append(duty_end)
        
        if not duty_starts or not duty_ends:
            return 0.0
        
        # Find the earliest start and latest end
        start = min(duty_starts)
        end = max(duty_ends)
        
        # Calculate FDP in hours
        return (end - start).total_seconds() / 3600.0
    
    def _initialize_rules(self) -> Dict:
        """Initialize DGCA rules from the PDF content"""
        return {
            "max_daily_flight_time": 10,  # hours
            "max_daily_fdp": {
                "0000-0559": 11.5,  # FDP for reporting between 0000-0559
                "0600-1159": 13,
                "1200-2359": 14
            },
            "max_weekly_flight_time": 35,  # hours in 7 days
            "max_weekly_duty_time": 60,    # hours in 7 days
            "min_rest_period": 12,         # hours
            "weekly_rest": 48,             # hours including 2 local nights
            "max_landings": {
                11: 6,
                11.5: 5,
                12: 4,
                12.5: 3,
                13: 2,
                13.5: 1
            }
        }


    def _check_minimum_crew_requirements(self, roster: pd.DataFrame,crew_data:pd.DataFrame) -> List[str]:
        """Check if each flight has the minimum required crew"""
        violations = []
        
        # Group assignments by flight
        for flight_number in roster['Flight_Number'].unique():
            flight_assignments = roster[roster['Flight_Number'] == flight_number]
            
            if flight_assignments.empty:
                continue
                
            # Get the aircraft type for this flight
            aircraft_type = flight_assignments.iloc[0]['Aircraft_Type']
            
            # Get minimum crew requirements
           # requirements = MinimumCrewRequirements.get_requirements(aircraft_type)
             # Get minimum crew requirements
            requirements = self._get_minimum_crew_requirements(aircraft_type)
            # Count cockpit and cabin crew
            cockpit_crew = 0
            cabin_crew = 0
            
            for _, assignment in flight_assignments.iterrows():
                crew_info = crew_data[crew_data['Crew_ID'] == assignment['Crew_ID']].iloc[0]
                
                if crew_info['Rank'] in ['Captain', 'First Officer']:
                    cockpit_crew += 1
                else:
                    cabin_crew += 1
            
            # Check cockpit crew requirements
            if cockpit_crew < requirements['cockpit']['min']:
                violations.append(
                    f"Flight {flight_number} has insufficient cockpit crew: "
                    f"{cockpit_crew} < {requirements['cockpit']['min']} required"
                )
            
            # Check cabin crew requirements
            if cabin_crew < requirements['cabin']['min']:
                violations.append(
                    f"Flight {flight_number} has insufficient cabin crew: "
                    f"{cabin_crew} < {requirements['cabin']['min']} required"
                )
        
        return violations 
    # app/core/dgca_validator.py (update the validate_roster method)
    def validate_roster(self, roster: List[Dict], crew_data: pd.DataFrame) -> List[str]:
        """Validate a roster against DGCA rules with minimal RAG usage"""
        violations = []
        
        # Convert the grouped roster to a flat format for validation
        flat_roster = self._flatten_grouped_roster(roster)
        
        # Check minimum crew requirements FIRST
        violations.extend(self._check_minimum_crew_requirements(flat_roster, crew_data))
        
        # Check each crew member's schedule
        for crew_id in flat_roster['Crew_ID'].unique():
            crew_schedule = flat_roster[flat_roster['Crew_ID'] == crew_id]
            crew_info = crew_data[crew_data['Crew_ID'] == crew_id].iloc[0]
            
            # Check standard limits (fast, local validation) - ALWAYS do this
            violations.extend(self._check_daily_limits(crew_schedule, crew_info))
            violations.extend(self._check_weekly_limits(crew_schedule, crew_info))
            violations.extend(self._check_rest_periods(crew_schedule))
            violations.extend(self._check_qualifications(crew_schedule, crew_info))
            
            # Use RAG ONLY for very complex cases and only if we have violations
            # to minimize API calls
            if (hasattr(self.rag_validator, 'rag_available') and 
                self.rag_validator.rag_available and
                len(crew_schedule) > 4 and  # Only validate crews with complex schedules
                len(violations) > 0):       # Only validate if we already found issues
                
                # Convert to dict for RAG validation
                roster_dict = flat_roster.to_dict('records')
                rag_violations = self.rag_validator.validate_complex_scenario(roster_dict, crew_id)
                for rag_violation in rag_violations:
                    violations.append(
                        f"RAG Validation - {rag_violation['type']}: {rag_violation['message']}"
                    )
        
        return violations

# Add this new method to convert grouped roster to flat format
    def _flatten_grouped_roster(self, grouped_roster: List[Dict]) -> pd.DataFrame:
        """Convert grouped roster (with Crew_Members array) to flat DataFrame"""
        flat_data = []
        
        for flight in grouped_roster:
            for crew_member in flight['Crew_Members']:
                flat_data.append({
                    'Date': flight['Date'],
                    'Flight_Number': flight['Flight_Number'],
                    'Crew_ID': crew_member['Crew_ID'],
                    'Duty_Start': flight['Duty_Start'],
                    'Duty_End': flight['Duty_End'],
                    'Aircraft_Type': flight['Aircraft_Type'],
                    'Origin': flight['Origin'],
                    'Destination': flight['Destination'],
                    'Duration': flight['Duration']
                })
        
        return pd.DataFrame(flat_data)       
  
   
    
    def _check_weekly_limits(self, schedule: pd.DataFrame, crew_info: pd.Series) -> List[str]:
        """Check weekly flight time and duty time limits"""
        violations = []
        
        # Group by week and check limits
        schedule = schedule.sort_values('Date')
        for i in range(len(schedule) - 6):
            week_schedule = schedule.iloc[i:i+7]
            
            total_flight_time = self._calculate_flight_time(week_schedule)
            total_duty_time = self._calculate_duty_time(week_schedule)
            
            if total_flight_time > self.rules["max_weekly_flight_time"]:
                violations.append(
                    f"Crew {crew_info['Crew_ID']} exceeded weekly flight time limit: "
                    f"{total_flight_time}h > {self.rules['max_weekly_flight_time']}h"
                )
            
            if total_duty_time > self.rules["max_weekly_duty_time"]:
                violations.append(
                    f"Crew {crew_info['Crew_ID']} exceeded weekly duty time limit: "
                    f"{total_duty_time}h > {self.rules['max_weekly_duty_time']}h"
                )
        
        return violations
   # app/core/dgca_validator.py (update the _check_rest_periods method)
    def _check_rest_periods(self, schedule: pd.DataFrame) -> List[str]:
        """Check minimum rest periods between duties with string date handling"""
        violations = []
        if schedule.empty:
            return violations
        
        # Sort by duty end time
        schedule = schedule.sort_values('Duty_End')
        
        for i in range(1, len(schedule)):
            prev_end = schedule.iloc[i-1]['Duty_End']
            curr_start = schedule.iloc[i]['Duty_Start']
            
            # Convert string dates to datetime objects if needed
            if isinstance(prev_end, str):
                try:
                    prev_end = datetime.fromisoformat(prev_end)
                except (ValueError, TypeError):
                    continue
            if isinstance(curr_start, str):
                try:
                    curr_start = datetime.fromisoformat(curr_start)
                except (ValueError, TypeError):
                    continue
            
            if not isinstance(prev_end, datetime) or not isinstance(curr_start, datetime):
                continue
            
            rest_hours = (curr_start - prev_end).total_seconds() / 3600
            
            if rest_hours < self.rules["min_rest_period"]:
                violations.append(
                    f"Insufficient rest for crew {schedule.iloc[i]['Crew_ID']} between "
                    f"{prev_end} and {curr_start}: {rest_hours}h < {self.rules['min_rest_period']}h"
                )
        
        return violations
        
    
    def _check_qualifications(self, schedule: pd.DataFrame, crew_info: pd.Series) -> List[str]:
        """Check if crew is qualified for assigned aircraft"""
        violations = []
        crew_qualifications = crew_info['Aircraft_Type_License'].split(', ')
        
        for _, duty in schedule.iterrows():
            if duty['Aircraft_Type'] not in crew_qualifications:
                violations.append(
                    f"Crew {crew_info['Crew_ID']} not qualified for {duty['Aircraft_Type']} "
                    f"on flight {duty['Flight_Number']}"
                )
        
        return violations
    
    def _calculate_flight_time(self, schedule: pd.DataFrame) -> float:
        """Calculate total flight time from schedule"""
        total_minutes = 0
        for _, row in schedule.iterrows():
            if 'Duration_HH_MM' in row:
                hours, minutes = map(int, row['Duration_HH_MM'].split(':'))
                total_minutes += hours * 60 + minutes
        return total_minutes / 60

    

    def _calculate_duty_time(self, schedule: pd.DataFrame) -> float:
        """Calculate total duty time with proper string to datetime conversion"""
        total_hours = 0.0
        
        for _, row in schedule.iterrows():
            start = row.get('Duty_Start')
            end = row.get('Duty_End')
            
            # Skip if either start or end is missing
            if pd.isna(start) or pd.isna(end) or start is None or end is None:
                continue
            
            # Convert string dates to datetime objects if needed
            if isinstance(start, str):
                try:
                    start = datetime.fromisoformat(start)
                except (ValueError, TypeError):
                    continue  # Skip this row if conversion fails
                    
            if isinstance(end, str):
                try:
                    end = datetime.fromisoformat(end)
                except (ValueError, TypeError):
                    continue  # Skip this row if conversion fails
            
            # Ensure we have datetime objects before performing operations
            if isinstance(start, datetime) and isinstance(end, datetime):
                try:
                    # Calculate duty hours
                    duty_duration = (end - start).total_seconds() / 3600
                    total_hours += duty_duration
                except (TypeError, AttributeError):
                    # Handle cases where datetime operations fail
                    logger.warning(f"Could not calculate duration: {start} to {end}")
                    continue
        
        return total_hours
    
    def _get_fdp_limit(self, reporting_time: datetime) -> float:
        """Get FDP limit based on reporting time"""
        hour = reporting_time.hour
        
        if 0 <= hour < 6:
            return self.rules["max_daily_fdp"]["0000-0559"]
        elif 6 <= hour < 12:
            return self.rules["max_daily_fdp"]["0600-1159"]
        else:
            return self.rules["max_daily_fdp"]["1200-2359"]
        
    # app/core/dgca_validator.py (add this method as a fallback)
    def _get_minimum_crew_requirements(self, aircraft_type):
        """Get minimum crew requirements with fallback if MinimumCrewRequirements is not available"""
        try:
            return MinimumCrewRequirements.get_requirements(aircraft_type)
        except NameError:
            # Fallback if MinimumCrewRequirements is not defined
            if aircraft_type == "A320neo":
                return {
                    "cockpit": {"min": 2, "max": 3},
                    "cabin": {"min": 4, "recommended": 5}
                }
            elif aircraft_type == "A321neo":
                return {
                    "cockpit": {"min": 2, "max": 3},
                    "cabin": {"min": 5, "recommended": 6}
                }
            else:
                return {
                    "cockpit": {"min": 2, "max": 3},
                    "cabin": {"min": 4, "recommended": 5}
                }
        