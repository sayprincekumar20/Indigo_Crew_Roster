# app/api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import date, time, datetime


class CrewMemberAssignment(BaseModel):
    Crew_ID: str
    Crew_Rank: str 

class RosterItem(BaseModel):
    Date: str  
    Flight_Number: str
    Duty_Start: Optional[str] = None  
    Duty_End: Optional[str] = None    
    Aircraft_Type: str
    Origin: str
    Destination: str
    Duration: str  
    Crew_Members: List[CrewMemberAssignment]  # Remove the top-level Crew_ID field   
     

class CrewMember(BaseModel):
    Crew_ID: str
    Name: str
    Base: str
    Rank: str
    Qualification: str
    Aircraft_Type_License: str
    Leave_Start: Optional[date] = None
    Leave_End: Optional[date] = None

class Flight(BaseModel):
    Date: date
    Flight_Number: str
    Origin: str
    Destination: str
    Scheduled_Departure_UTC: time
    Scheduled_Arrival_UTC: time
    Aircraft_Type: str
    Duration_HH_MM: str

class RosterRequest(BaseModel):
    start_date: date
    end_date: date
    optimization_weights: Optional[Dict[str, float]] = None

class RosterResponse(BaseModel):
    roster: List[RosterItem]  # Now uses RosterItem which has string dates
    fitness_score: float
    violations: List[Dict[str, str]]
    optimization_metrics: Dict[str, float]

# app/api/models.py - Add these models
class DisruptionRequest(BaseModel):
    crew_id: Optional[str] = None
    flight_number: Optional[str] = None
    start_date: date
    end_date: date
    reason: str
    disruption_type: str  # sickness, technical, weather, etc.

class ChatQuery(BaseModel):
    message: str
    context: Optional[Dict] = None

class ReplacementCandidate(BaseModel):
    crew_id: str
    score: float
    reasons: List[str]
    warnings: List[str]
    details: Dict

class ReplacementSuggestion(BaseModel):
    flight_number: str
    original_crew: str
    recommended_crew: str
    confidence: float
    explanation: str
    validation_checks: Dict

class SuggestionApply(BaseModel):
    original_crew: str
    replacements: List[Dict[str, str]]