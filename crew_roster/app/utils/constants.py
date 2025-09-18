# app/utils/constants.py
from datetime import time
from enum import Enum

# DGCA Rule Constants
class DGCARules:
    MAX_DAILY_FLIGHT_TIME = 10.0  # hours
    MAX_WEEKLY_FLIGHT_TIME = 35.0  # hours in 7 days
    MAX_WEEKLY_DUTY_TIME = 60.0    # hours in 7 days
    MIN_REST_PERIOD = 12.0         # hours
    WEEKLY_REST = 48.0             # hours including 2 local nights
    
    # FDP limits based on reporting time
    FDP_LIMITS = {
        "0000-0559": 11.5,  # 11 hours 30 minutes
        "0600-1159": 13.0,   # 13 hours
        "1200-2359": 14.0    # 14 hours
    }
    
    # Maximum landings based on FDP
    MAX_LANDINGS = {
        11.0: 6,
        11.5: 5,
        12.0: 4,
        12.5: 3,
        13.0: 2,
        13.5: 1
    }
    
    # Night duty hours (10 PM to 6 AM)
    NIGHT_START = time(22, 0)  # 10:00 PM
    NIGHT_END = time(6, 0)     # 6:00 AM

# Aircraft Types
class AircraftTypes:
    A320NEO = "A320neo"
    A321NEO = "A321neo"
    ALL_TYPES = [A320NEO, A321NEO]

# Crew Ranks
class CrewRanks(Enum):
    CAPTAIN = "Captain"
    FIRST_OFFICER = "First Officer"
    PURSUER = "Purser"
    FLIGHT_ATTENDANT = "FA"
    SENIOR_FA = "Senior"
    JUNIOR_FA = "Junior"

# Crew Bases
class CrewBases(Enum):
    DELHI = "DEL"
    MUMBAI = "BOM"
    BANGALORE = "BLR"
    CHENNAI = "MAA"
    KOLKATA = "CCU"
    HYDERABAD = "HYD"
    GOA = "GOI"
    AHMEDABAD = "AMD"
    PUNE = "PNQ"
    COCHIN = "COK"

# Preference Types
class PreferenceTypes(Enum):
    DAY_OFF = "Day_Off"
    PREFERRED_SECTOR = "Preferred_Sector"
    AVOID_LATE_NIGHT = "Avoid_Late_Night"
    PREFERRED_AIRCRAFT = "Preferred_Aircraft"
    CONSECUTIVE_DAYS_OFF = "Consecutive_Days_Off"
    MAX_DUTY_DAYS = "Max_Duty_Days"
    AVOID_EARLY_MORNING = "Avoid_Early_Morning"
    PREFERRED_BASE = "Preferred_Base"

# Priority Levels
class PriorityLevels(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

# Duty Types
class DutyTypes(Enum):
    FLIGHT = "Flight"
    STANDBY = "Standby"
    TRAINING = "Training"
    ADMIN = "Administrative"
    POSITIONING = "Positioning"

# Disruption Types
class DisruptionTypes(Enum):
    DELAY = "Delay"
    SICKNESS = "Sickness"
    CANCELLATION = "Cancellation"
    TECHNICAL = "Technical"
    WEATHER = "Weather"
    ATC_RESTRICTIONS = "ATC Restrictions"
    MEDICAL = "Medical Unavailable"

# API Response Codes
class ResponseCodes(Enum):
    SUCCESS = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_ERROR = 500

# Genetic Algorithm Parameters
class GAParameters:
    POPULATION_SIZE = 100
    GENERATIONS = 50
    CROSSOVER_PROBABILITY = 0.7
    MUTATION_PROBABILITY = 0.2
    TOURNAMENT_SIZE = 3
    ELITISM_COUNT = 2

# Fitness Weights
class FitnessWeights:
    VIOLATION_PENALTY = 1000.0
    PREFERENCE_WEIGHT = 100.0
    FAIRNESS_WEIGHT = 50.0
    UTILIZATION_WEIGHT = 30.0
    COST_WEIGHT = 20.0

# Time Zones (IST = UTC+5:30)
class TimeZones:
    UTC = "UTC"
    IST = "IST"
    DELHI = "Asia/Kolkata"
    MUMBAI = "Asia/Kolkata"
    SINGAPORE = "Asia/Singapore"
    DUBAI = "Asia/Dubai"
    LONDON = "Europe/London"
    NEW_YORK = "America/New_York"

# Airport Curfew Information
AIRPORT_CURFEWS = {
    "BOM": {"start": "00:00", "end": "06:00"},
    "CCU": {"start": "00:30", "end": "06:00"},
    "GOI": {"start": "00:00", "end": "06:00"},
    "PNQ": {"start": "00:00", "end": "06:00"},
    "COK": {"start": "00:00", "end": "06:00"}
}

# Default Values
DEFAULT_VALUES = {
    "MAX_FLIGHT_DURATION": 8.0,  # hours
    "MIN_TURNAROUND_TIME": 0.5,  # hours
    "MAX_CONSECUTIVE_NIGHTS": 2,
    "MIN_CREW_REST": 10.0,       # hours
    "MAX_DUTY_DAY_EXTENSION": 2.0,  # hours
    "STANDBY_UTILIZATION_THRESHOLD": 0.3  # 30%
}

# Error Messages
ERROR_MESSAGES = {
    "DATA_LOAD_ERROR": "Failed to load required data",
    "VALIDATION_ERROR": "Roster validation failed",
    "OPTIMIZATION_ERROR": "Genetic algorithm optimization failed",
    "NO_FLIGHTS_FOUND": "No flights found for the specified date range",
    "INSUFFICIENT_CREW": "Insufficient crew available for scheduling",
    "DGCA_VIOLATION": "DGCA rules violation detected",
    "RAG_VALIDATION_ERROR": "RAG-based validation error"
}

# Success Messages
SUCCESS_MESSAGES = {
    "ROSTER_GENERATED": "Roster generated successfully",
    "VALIDATION_PASSED": "All validations passed",
    "OPTIMIZATION_COMPLETE": "Optimization completed successfully",
    "DATA_LOADED": "Data loaded successfully"
}



# Rank hierarchy for replacement considerations
RANK_HIERARCHY = {
    'Captain': ['Senior Captain', 'Captain', 'Check Captain'],
    'Senior First Officer': ['Captain', 'Senior First Officer', 'First Officer'],
    'First Officer': ['Senior First Officer', 'First Officer', 'Junior First Officer'],
    'Purser': ['Senior Purser', 'Purser', 'Senior Cabin Crew'],
    'Senior Cabin Crew': ['Purser', 'Senior Cabin Crew', 'Cabin Crew'],
    'Cabin Crew': ['Senior Cabin Crew', 'Cabin Crew', 'Trainee Cabin Crew']
}

# Base positioning times (hours)
BASE_POSITIONING_TIMES = {
    ('DEL', 'BOM'): 2.0,
    ('DEL', 'BLR'): 2.5,
    ('DEL', 'MAA'): 2.5,
    ('BOM', 'BLR'): 1.5,
    ('BOM', 'MAA'): 2.0,
    ('BLR', 'MAA'): 1.0,
}

# app/utils/constants.py - Update MinimumCrewRequirements class

class MinimumCrewRequirements:
    A320NEO = {
        "cockpit": {
            "min": 2,  # Captain + First Officer
            "ranks": ["Captain", "Senior First Officer", "First Officer", "Instructor/Check Pilot", "Trainee First Officer"]
        },
        "cabin": {
            "min": 4,
            "ranks": ["Purser", "SCCM (Lead Cabin Crew)", "Senior Cabin Crew", "Junior Cabin Crew", "Trainee Cabin Crew"]
        }
    }
    
    A321NEO = {
        "cockpit": {
            "min": 2,  # Captain + First Officer  
            "ranks": ["Captain", "Senior First Officer", "First Officer", "Instructor/Check Pilot", "Trainee First Officer"]
        },
        "cabin": {
            "min": 5,
            "ranks": ["Purser", "SCCM (Lead Cabin Crew)", "Senior Cabin Crew", "Junior Cabin Crew", "Trainee Cabin Crew"]
        }
    }
    
    @classmethod
    def get_requirements(cls, aircraft_type):
        if aircraft_type == "A320neo":
            return cls.A320NEO
        elif aircraft_type == "A321neo":
            return cls.A321NEO
        else:
            # Default to A320neo requirements for unknown types
            return cls.A320NEO