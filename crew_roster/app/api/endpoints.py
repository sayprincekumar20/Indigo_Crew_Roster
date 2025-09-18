# app/api/endpoints.py
from django.conf import settings
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional,Dict, Any,Union
import pandas as pd
from datetime import datetime, date
import json
import logging
from sqlite3 import IntegrityError

from app.data.load_data import data_loader
from app.data.database import db_manager
from app.core.genetic_algorithm import CrewRosteringGA
from app.core.dgca_validator import DGCAValidator
from app.api.models import RosterRequest, RosterResponse, DisruptionRequest, ChatQuery
from app.utils.helpers import apply_replacement_to_roster, calculate_fairness_metrics, format_violations_for_display,convert_dates_to_strings, get_current_roster, save_updated_roster, validate_replacement
from app.utils.constants import ERROR_MESSAGES, SUCCESS_MESSAGES
from app.core.llm_chatbot import CrewDisruptionChatbot

logger = logging.getLogger(__name__)
router = APIRouter()

# Load data once at startup
@router.on_event("startup")
async def startup_event():
    """Initialize application data on startup"""
    try:
        logger.info("Starting up Crew Rostering API...")
        if not data_loader.load_all_data():
            raise Exception(ERROR_MESSAGES["DATA_LOAD_ERROR"])
        logger.info(SUCCESS_MESSAGES["DATA_LOADED"])
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "database": "connected" if db_manager.conn else "disconnected",
        "data_loaded": data_loader.flights_df is not None
    }

@router.post("/generate-roster", response_model=RosterResponse)
async def generate_roster(request: RosterRequest):
    """Generate an optimized crew roster using genetic algorithm"""
    try:
        logger.info(f"Generating roster for {request.start_date} to {request.end_date}")
        
        # Use the new method to get flights for date range
        flights_subset = data_loader.get_flights_for_date_range(request.start_date, request.end_date)
        
        logger.info(f"Found {len(flights_subset)} flights in date range")
        
        if flights_subset.empty:
            logger.warning(ERROR_MESSAGES["NO_FLIGHTS_FOUND"])
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES["NO_FLIGHTS_FOUND"])
        
        # Check if we have available crew
        if data_loader.crew_df is None or data_loader.crew_df.empty:
            logger.error(ERROR_MESSAGES["INSUFFICIENT_CREW"])
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES["INSUFFICIENT_CREW"])
        
        # Log some info about the flights
        logger.info(f"Flights date range: {flights_subset['Date'].min()} to {flights_subset['Date'].max()}")
        
        # Initialize validator and GA
        validator = DGCAValidator()
        ga = CrewRosteringGA(
            flights_subset,
            data_loader.crew_df,
            data_loader.crew_preferences_df,
            validator
        )
        
        # Run optimization
        logger.info("Starting genetic algorithm optimization")
        result = ga.optimize_roster()
        logger.info(f"Optimization completed with {len(result['violations'])} violations")
        
        # Convert dates to strings for JSON serialization
        # serializable_roster = convert_dates_to_strings(result['roster'])
        serializable_roster = result['roster']
        # Calculate optimization metrics - need to flatten for fairness calculation
        flat_roster = validator._flatten_grouped_roster(result['roster'])
        fairness_metrics = calculate_fairness_metrics(flat_roster)
        
        # # Calculate optimization metrics
        # roster_df = pd.DataFrame(result['roster'])
        # fairness_metrics = calculate_fairness_metrics(roster_df)
        
        optimization_metrics = {
            "total_assignments": len(flat_roster), 
            #"total_assignments": len(result['roster']),
            "crew_utilization": fairness_metrics['crew_utilization'],
            "violation_count": len(result['violations']),
            "fitness_score": sum(result['fitness']),
            "fairness_score": fairness_metrics['fairness_score'],
            "max_duty_hours": fairness_metrics['max_duty_hours'],
            "min_duty_hours": fairness_metrics['min_duty_hours'],
            "avg_duty_hours": fairness_metrics['avg_duty_hours'],
            "std_dev_duty_hours": fairness_metrics['std_dev_duty_hours']
        }
        
        # Format violations for better display
        formatted_violations = format_violations_for_display(result['violations'])
        
        # Store roster in database
        roster_json = json.dumps(result['roster'])
        try:
            db_manager.execute_query(
                """INSERT INTO generated_rosters 
                (start_date, end_date, fitness_score, violation_count, roster_data) 
                VALUES (?, ?, ?, ?, ?)""",
                (request.start_date.isoformat(), request.end_date.isoformat(), 
                 sum(result['fitness']), len(result['violations']), roster_json)
            )
            db_manager.conn.commit()
            logger.info("Roster saved to database successfully")
        except IntegrityError as e:
            logger.warning(f"Database integrity error: {e}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
        
        return RosterResponse(
           # roster=result['roster'],
            roster=serializable_roster,
            fitness_score=result['fitness'][0],
            violations=formatted_violations,
            optimization_metrics=optimization_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating roster: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{ERROR_MESSAGES['OPTIMIZATION_ERROR']}: {str(e)}")

@router.get("/rag-validate")
async def rag_validate_scenario(scenario: str):
    """Validate a specific scenario using RAG"""
    try:
        validator = DGCAValidator()
        result = validator.rag_validator.validate_with_rag(scenario)
        return result
    except Exception as e:
        logger.error(f"RAG validation error: {e}")
        raise HTTPException(status_code=500, detail=f"{ERROR_MESSAGES['RAG_VALIDATION_ERROR']}: {str(e)}")

@router.get("/rosters/history")
async def get_roster_history(limit: int = 10, offset: int = 0):
    """Get history of generated rosters"""
    try:
        result = db_manager.fetch_all(
            """SELECT id, created_at, start_date, end_date, fitness_score, violation_count 
            FROM generated_rosters ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (limit, offset)
        )
        
        total_count = db_manager.fetch_one("SELECT COUNT(*) FROM generated_rosters")[0]
        
        return {
            "rosters": [{
                "id": row[0],
                "created_at": row[1],
                "start_date": row[2],
                "end_date": row[3],
                "fitness_score": row[4],
                "violation_count": row[5]
            } for row in result],
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error fetching roster history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching roster history: {str(e)}")

@router.get("/rosters/{roster_id}")
async def get_roster_details(roster_id: int):
    """Get details of a specific roster"""
    try:
        result = db_manager.fetch_one(
            "SELECT roster_data FROM generated_rosters WHERE id = ?",
            (roster_id,)
        )
        if result:
            return {
                "roster_id": roster_id,
                "roster_data": json.loads(result[0])
            }
        else:
            raise HTTPException(status_code=404, detail="Roster not found")
    except Exception as e:
        logger.error(f"Error fetching roster details: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching roster details: {str(e)}")

@router.get("/crew/{crew_id}/schedule")
async def get_crew_schedule(
    crew_id: str, 
    start_date: Optional[date] = None, 
    end_date: Optional[date] = None
):
    """Get schedule for a specific crew member"""
    try:
        # Check if crew exists
        crew_exists = db_manager.fetch_one(
            "SELECT COUNT(*) FROM crew_members WHERE Crew_ID = ?",
            (crew_id,)
        )
        
        if not crew_exists or crew_exists[0] == 0:
            raise HTTPException(status_code=404, detail="Crew member not found")
        
        # Build query based on date range
        query = """SELECT gr.id, gr.start_date, gr.end_date, gr.violation_count
                   FROM generated_rosters gr
                   WHERE gr.roster_data LIKE ?"""
        params = [f'%"Crew_ID": "{crew_id}"%']
        
        if start_date and end_date:
            query += " AND gr.start_date >= ? AND gr.end_date <= ?"
            params.extend([start_date.isoformat(), end_date.isoformat()])
        
        query += " ORDER BY gr.created_at DESC LIMIT 10"
        
        result = db_manager.fetch_all(query, tuple(params))
        
        return {
            "crew_id": crew_id,
            "schedules": [{
                "roster_id": row[0],
                "start_date": row[1],
                "end_date": row[2],
                "violation_count": row[3]
            } for row in result]
        }
    except Exception as e:
        logger.error(f"Error fetching crew schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching crew schedule: {str(e)}")

@router.get("/flights")
async def get_flights(
    date: Optional[date] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    aircraft_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get flights with filtering options"""
    try:
        query = "SELECT * FROM flights WHERE 1=1"
        params = []
        
        if date:
            query += " AND Date = ?"
            params.append(date.isoformat())
        
        if origin:
            query += " AND Origin = ?"
            params.append(origin)
            
        if destination:
            query += " AND Destination = ?"
            params.append(destination)
            
        if aircraft_type:
            query += " AND Aircraft_Type = ?"
            params.append(aircraft_type)
        
        query += " ORDER BY Date, Scheduled_Departure_UTC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = db_manager.fetch_all(query, tuple(params))
        columns = ['Date', 'Flight_Number', 'Origin', 'Destination', 
                  'Scheduled_Departure_UTC', 'Scheduled_Arrival_UTC', 
                  'Aircraft_Type', 'Duration_HH_MM']
        
        return {
            "flights": [dict(zip(columns, row)) for row in result],
            "total": len(result),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error fetching flights: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching flights: {str(e)}")

@router.get("/crew")
async def get_crew_members(
    base: Optional[str] = None,
    rank: Optional[str] = None,
    aircraft_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get crew members with filtering options"""
    try:
        query = "SELECT * FROM crew_members WHERE 1=1"
        params = []
        
        if base:
            query += " AND Base = ?"
            params.append(base)
            
        if rank:
            query += " AND Rank = ?"
            params.append(rank)
            
        if aircraft_type:
            query += " AND Aircraft_Type_License LIKE ?"
            params.append(f'%{aircraft_type}%')
        
        query += " ORDER BY Crew_ID LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = db_manager.fetch_all(query, tuple(params))
        columns = ['Crew_ID', 'Name', 'Base', 'Rank', 'Qualification', 
                  'Aircraft_Type_License', 'Leave_Start', 'Leave_End']
        
        return {
            "crew_members": [dict(zip(columns, row)) for row in result],
            "total": len(result),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error fetching crew members: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching crew members: {str(e)}")

@router.get("/debug/date-issues")
async def debug_date_issues():
    """Debug endpoint to identify date comparison issues"""
    try:
        debug_info = {}
        
        # Check if data is loaded
        debug_info['data_loaded'] = data_loader.flights_df is not None
        
        if data_loader.flights_df is not None:
            # Check the data types
            debug_info['flights_df_dtypes'] = str(data_loader.flights_df.dtypes.to_dict())
            
            # Check Date column specifically
            if 'Date' in data_loader.flights_df.columns:
                debug_info['date_column_type'] = str(data_loader.flights_df['Date'].dtype)
                debug_info['date_column_sample'] = data_loader.flights_df['Date'].head(3).tolist()
                debug_info['date_column_sample_types'] = [str(type(x)) for x in data_loader.flights_df['Date'].head(3)]
            
            # Test date comparison
            test_date = date(2023, 10, 1)
            debug_info['test_date'] = str(test_date)
            debug_info['test_date_type'] = str(type(test_date))
            
            # Try the comparison that's failing
            try:
                mask = data_loader.flights_df['Date'] >= test_date
                debug_info['comparison_works'] = True
                debug_info['comparison_result_count'] = int(mask.sum())
            except Exception as e:
                debug_info['comparison_works'] = False
                debug_info['comparison_error'] = str(e)
                
            # Try with proper conversion
            try:
                test_date_dt = pd.to_datetime(test_date)
                mask = data_loader.flights_df['Date'] >= test_date_dt
                debug_info['converted_comparison_works'] = True
                debug_info['converted_comparison_result_count'] = int(mask.sum())
            except Exception as e:
                debug_info['converted_comparison_works'] = False
                debug_info['converted_comparison_error'] = str(e)
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug error: {e}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@router.get("/debug/data-status")
async def debug_data_status():
    """Debug endpoint to check data loading status"""
    try:
        status = {
            "crew_df_loaded": data_loader.crew_df is not None,
            "flights_df_loaded": data_loader.flights_df is not None,
            "aircraft_df_loaded": data_loader.aircraft_df is not None,
            "airports_df_loaded": data_loader.airports_df is not None,
            "crew_preferences_df_loaded": data_loader.crew_preferences_df is not None,
            "crew_qualifications_df_loaded": data_loader.crew_qualifications_df is not None,
            "historical_rosters_df_loaded": data_loader.historical_rosters_df is not None,
            "disruption_data_df_loaded": data_loader.disruption_data_df is not None,
        }
        
        if data_loader.crew_df is not None:
            status["crew_count"] = len(data_loader.crew_df)
            
        if data_loader.flights_df is not None:
            status["flights_count"] = len(data_loader.flights_df)
            status["flights_date_range"] = {
                "min": data_loader.flights_df['Date'].min().strftime('%Y-%m-%d') if not data_loader.flights_df.empty else None,
                "max": data_loader.flights_df['Date'].max().strftime('%Y-%m-%d') if not data_loader.flights_df.empty else None
            }
        
        # Check database connection
        status["database_connected"] = db_manager.conn is not None
        
        return status
        
    except Exception as e:
        logger.error(f"Data status debug error: {e}")
        raise HTTPException(status_code=500, detail=f"Data status debug error: {str(e)}")

@router.get("/test-date-comparison")
async def test_date_comparison():
    """Test endpoint to verify date comparison is working"""
    try:
        # Test with a specific date range
        test_start = date(2023, 10, 1)
        test_end = date(2023, 10, 3)
        
        flights = data_loader.get_flights_for_date_range(test_start, test_end)
        
        return {
            "success": True,
            "start_date": test_start.isoformat(),
            "end_date": test_end.isoformat(),
            "flights_count": len(flights),
            "flights_dates": flights['Date'].unique().tolist() if not flights.empty else []
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.delete("/rosters/{roster_id}")
async def delete_roster(roster_id: int):
    """Delete a specific roster from history"""
    try:
        result = db_manager.execute_query(
            "DELETE FROM generated_rosters WHERE id = ?",
            (roster_id,)
        )
        db_manager.conn.commit()
        
        if result.rowcount > 0:
            return {"success": True, "message": f"Roster {roster_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Roster not found")
            
    except Exception as e:
        logger.error(f"Error deleting roster: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting roster: {str(e)}")

@router.get("/preferences/{crew_id}")
async def get_crew_preferences(crew_id: str):
    """Get preferences for a specific crew member"""
    try:
        result = db_manager.fetch_all(
            "SELECT Preference_Type, Preference_Detail, Priority FROM crew_preferences WHERE Crew_ID = ?",
            (crew_id,)
        )
        
        return {
            "crew_id": crew_id,
            "preferences": [{
                "type": row[0],
                "detail": row[1],
                "priority": row[2]
            } for row in result]
        }
    except Exception as e:
        logger.error(f"Error fetching preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching preferences: {str(e)}")
    

    # app/api/endpoints.py - Add these endpoints
@router.post("/disruption/analyze")
async def analyze_disruption(request: DisruptionRequest):
    """Analyze a disruption and find replacement options"""
    try:
        # Load current roster
        current_roster = await get_current_roster()  # You'll need to implement this
        
        # Initialize chatbot
        chatbot = CrewDisruptionChatbot()
        
        # Handle disruption
        result = await chatbot.handle_disruption_request(request.dict(), current_roster)
        
        if result['success']:
            return {
                "status": "success",
                "analysis": result['summary'],
                "affected_flights": len(result['affected_flights']),
                "detailed_analysis": result['replacement_options']
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Error analyzing disruption: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing disruption: {str(e)}")

@router.post("/disruption/chat")
async def disruption_chat(query: ChatQuery):
    """Chat interface for disruption handling"""
    try:
        chatbot = CrewDisruptionChatbot()

        if not chatbot.llm_available:
            raise HTTPException(status_code=400, detail="LLM service not available. Please configure Groq API.")
        
        # Simple chat response (you can enhance this)
        prompt = f"""
        You are an airline crew operations expert. Help with this crew disruption question:
        
        QUESTION: {query.message}
        
        CONTEXT: {json.dumps(query.context or {}, indent=2)}
        
        Provide helpful, professional advice about crew disruptions, replacements, and DGCA compliance.
        """
        
        response = chatbot.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=chatbot.model_name,
            temperature=0.4
        )
        
        return {
            "response": response.choices[0].message.content,
            "suggested_actions": ["analyze_disruption", "view_roster", "check_compliance"]
        }
        
    except Exception as e:
        logger.error(f"Error in disruption chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/disruption/apply-replacement")
async def apply_disruption_replacement(replacement: Dict):
    """Apply a crew replacement for disruption"""
    try:
        # Validate replacement
        validation_result = await validate_replacement(replacement)
        
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=f"Invalid replacement: {validation_result['issues']}")
        
        # Apply replacement to roster
        updated_roster = await apply_replacement_to_roster(replacement)
        
        # Save updated roster
        await save_updated_roster(updated_roster)
        
        return {
            "status": "success",
            "message": "Replacement applied successfully",
            "updated_roster_id": updated_roster['id']
        }
        
    except Exception as e:
        logger.error(f"Error applying replacement: {e}")
        raise HTTPException(status_code=500, detail=f"Error applying replacement: {str(e)}")