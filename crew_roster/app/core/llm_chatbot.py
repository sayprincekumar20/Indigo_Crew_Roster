# # app/core/llm_chatbot.py
# import json
# from typing import Dict, List, Optional
# import logging
# import pandas as pd
# import os
# from dotenv import load_dotenv
# from groq import Groq
# from app.data.load_data import data_loader
# from app.utils.helpers import parse_time, calculate_duty_hours

# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()

# class CrewDisruptionChatbot:
#     def __init__(self):
#         self.llm_available = False
#         self.llm = None
#         self.model_name = os.getenv("GROQ_MODEL", "gemma2-9b-it")
        
#         try:
#             # Get API key directly from environment
#             api_key = os.getenv("GROQ_API_KEY")
            
#             if not api_key:
#                 logger.warning("Groq API key not configured. LLM features disabled.")
#                 return
                
#             self.llm = Groq(api_key=api_key)
#             self.llm_available = True
#             logger.info(f"Groq API initialized successfully with model: {self.model_name}")
            
#         except Exception as e:
#             logger.error(f"Failed to initialize Groq API: {e}")
#             self.llm_available = False
        
#     async def handle_disruption_request(self, disruption_request: Dict, roster_data: List[Dict]) -> Dict:
#         """Handle a disruption request using LLM"""
#         if not self.llm_available:
#             return {
#                 'success': False,
#                 'error': 'Groq API not configured. Please set GROQ_API_KEY environment variable.',
#                 'fallback_options': self._get_fallback_options(disruption_request, roster_data)
#             }
        
#         try:
#             # Step 1: Parse and understand the request
#             structured_request = await self._parse_disruption_request(disruption_request)
            
#             # Step 2: Find affected flights
#             affected_flights = self._find_affected_flights(roster_data, structured_request)
            
#             # Step 3: Find replacement candidates
#             replacement_options = []
#             for flight in affected_flights:
#                 candidates = self._find_replacement_candidates(flight, structured_request, roster_data)
#                 if candidates:  # Only analyze if we have candidates
#                     llm_analysis = await self._analyze_candidates_with_llm(flight, candidates, structured_request)
#                     replacement_options.append({
#                         'flight': flight,
#                         'candidates': candidates,
#                         'llm_analysis': llm_analysis
#                     })
            
#             # Step 4: Generate summary response
#             summary = await self._generate_summary_response(structured_request, replacement_options)
            
#             return {
#                 'success': True,
#                 'summary': summary,
#                 'affected_flights': affected_flights,
#                 'replacement_options': replacement_options,
#                 'structured_request': structured_request
#             }
            
#         except Exception as e:
#             logger.error(f"Error in disruption chatbot: {e}")
#             return {'success': False, 'error': str(e)}
    
#     def _get_fallback_options(self, disruption_request: Dict, roster_data: List[Dict]) -> Dict:
#         """Provide basic fallback options without LLM"""
#         affected_crew = disruption_request.get('crew_id')
#         affected_flights = []
        
#         if affected_crew:
#             for flight in roster_data:
#                 for crew_member in flight['Crew_Members']:
#                     if crew_member['Crew_ID'] == affected_crew:
#                         affected_flights.append({
#                             'flight_number': flight['Flight_Number'],
#                             'date': flight['Date'],
#                             'origin': flight['Origin'],
#                             'destination': flight['Destination'],
#                             'aircraft_type': flight['Aircraft_Type']
#                         })
#                         break
        
#         return {
#             'affected_crew': affected_crew,
#             'affected_flights': affected_flights,
#             'message': 'LLM not available. Basic disruption information provided.'
#         }
    
#     async def _parse_disruption_request(self, request: Dict) -> Dict:
#         """Use LLM to parse natural language disruption request"""
#         if not self.llm_available:
#             # Basic parsing without LLM
#             return {
#                 'crew_id': request.get('crew_id'),
#                 'flight_numbers': [],
#                 'start_date': request.get('start_date'),
#                 'end_date': request.get('end_date'),
#                 'reason': request.get('reason', 'unknown'),
#                 'disruption_type': request.get('disruption_type', 'unknown'),
#                 'severity': 'medium'
#             }
        
#         prompt = f"""
#         Analyze this crew disruption message and extract structured information:
        
#         MESSAGE: {json.dumps(request, indent=2)}
        
#         Extract the following information as JSON:
#         - crew_id (if mentioned, e.g., "FO002")
#         - flight_numbers (array of flight numbers if mentioned)
#         - start_date (date when disruption starts)
#         - end_date (date when disruption ends) 
#         - reason (sickness, technical, weather, etc.)
#         - disruption_type
#         - severity (high, medium, low)
        
#         Return ONLY valid JSON format.
#         """
        
#         response = self.llm.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model=self.model_name,
#             temperature=0.1,
#             response_format={"type": "json_object"}
#         )
        
#         return json.loads(response.choices[0].message.content)
    
#     def _find_affected_flights(self, roster_data: List[Dict], request: Dict) -> List[Dict]:
#         """Find flights affected by the disruption"""
#         affected_flights = []
#         crew_id = request.get('crew_id')
        
#         if not crew_id:
#             return affected_flights
        
#         for flight in roster_data:
#             for crew_member in flight['Crew_Members']:
#                 if (crew_member['Crew_ID'] == crew_id and
#                     self._is_date_in_range(flight['Date'], request['start_date'], request['end_date'])):
#                     affected_flights.append(flight)
#                     break
        
#         return affected_flights
    
#     def _find_replacement_candidates(self, flight: Dict, request: Dict, roster_data: List[Dict]) -> List[Dict]:
#         """Find suitable replacement candidates for a flight"""
#         candidates = []
#         original_crew_id = request.get('crew_id')
        
#         if not original_crew_id:
#             return candidates
        
#         try:
#             # Get original crew info
#             original_crew_df = data_loader.crew_df[data_loader.crew_df['Crew_ID'] == original_crew_id]
#             if original_crew_df.empty:
#                 return candidates
                
#             original_crew = original_crew_df.iloc[0]
            
#             # Find potential replacements
#             for _, crew in data_loader.crew_df.iterrows():
#                 if (crew['Crew_ID'] != original_crew_id and 
#                     self._is_suitable_replacement(crew, original_crew, flight, roster_data)):
                    
#                     candidate = {
#                         'crew_id': crew['Crew_ID'],
#                         'name': crew['Name'],
#                         'rank': crew['Rank'],
#                         'base': crew['Base'],
#                         'score': self._calculate_suitability_score(crew, original_crew, flight, roster_data),
#                         'qualifications': crew['Aircraft_Type_License'],
#                         'reasons': self._get_replacement_reasons(crew, original_crew, flight),
#                         'warnings': self._get_replacement_warnings(crew, flight, roster_data)
#                     }
#                     candidates.append(candidate)
            
#             # Sort by score (descending)
#             return sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]  # Top 5 candidates
            
#         except Exception as e:
#             logger.error(f"Error finding replacement candidates: {e}")
#             return []
    
#     def _is_suitable_replacement(self, candidate_crew, original_crew, flight, roster_data) -> bool:
#         """Check if a crew member is suitable for replacement"""
#         try:
#             # Same rank
#             if candidate_crew['Rank'] != original_crew['Rank']:
#                 return False
            
#             # Aircraft qualification
#             aircraft_types = str(candidate_crew['Aircraft_Type_License']).split(', ')
#             if flight['Aircraft_Type'] not in [at.strip() for at in aircraft_types]:
#                 return False
            
#             # Check availability
#             if not self._is_crew_available(candidate_crew['Crew_ID'], flight['Date'], roster_data):
#                 return False
            
#             return True
            
#         except Exception as e:
#             logger.error(f"Error in suitability check: {e}")
#             return False
    
#     def _calculate_suitability_score(self, candidate_crew, original_crew, flight, roster_data) -> float:
#         """Calculate suitability score (0-100)"""
#         try:
#             score = 60  # Base score
            
#             # Same base bonus
#             if candidate_crew['Base'] == flight['Origin']:
#                 score += 20
            
#             # Experience bonus (simplified)
#             if 'Senior' in candidate_crew['Rank']:
#                 score += 15
            
#             # Availability bonus
#             if self._is_crew_highly_available(candidate_crew['Crew_ID'], flight['Date'], roster_data):
#                 score += 10
            
#             # Check for duty time concerns
#             if self._would_exceed_duty_hours(candidate_crew['Crew_ID'], flight, roster_data):
#                 score -= 25
            
#             return max(0, min(100, score))
            
#         except Exception as e:
#             logger.error(f"Error calculating suitability score: {e}")
#             return 50  # Default score on error
    
#     def _is_crew_highly_available(self, crew_id: str, date: str, roster_data: List[Dict]) -> bool:
#         """Check if crew is highly available (simplified)"""
#         # Count assignments on this date
#         assignment_count = 0
#         for flight in roster_data:
#             if flight['Date'] == date:
#                 for crew in flight['Crew_Members']:
#                     if crew['Crew_ID'] == crew_id:
#                         assignment_count += 1
#         return assignment_count == 0  # Highly available if no assignments
    
#     async def _analyze_candidates_with_llm(self, flight: Dict, candidates: List[Dict], request: Dict) -> str:
#         """Use LLM to analyze and rank candidates"""
#         if not self.llm_available:
#             return "LLM analysis not available. Please configure Groq API."
        
#         prompt = f"""
#         Analyze these crew replacement options and provide intelligent recommendations:
        
#         FLIGHT: {flight['Flight_Number']} ({flight['Origin']}-{flight['Destination']})
#         AIRCRAFT: {flight['Aircraft_Type']}
#         DATE: {flight['Date']}
#         DISRUPTION: {request.get('reason', 'Unknown')}
        
#         CANDIDATES: {json.dumps(candidates, indent=2)}
        
#         Provide a concise analysis with:
#         1. Top recommendation with score
#         2. Key advantages of the best candidate
#         3. Potential concerns to consider
#         4. DGCA compliance summary
        
#         Format for easy reading with bullet points.
#         """
        
#         response = self.llm.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model=self.model_name,
#             temperature=0.3
#         )
        
#         return response.choices[0].message.content
    
#     async def _generate_summary_response(self, request: Dict, replacement_options: List[Dict]) -> str:
#         """Generate summary response using LLM"""
#         if not self.llm_available:
#             return "LLM summary not available. Please configure Groq API."
        
#         prompt = f"""
#         Generate a comprehensive summary of this crew disruption situation:
        
#         DISRUPTION REQUEST: {json.dumps(request, indent=2)}
        
#         REPLACEMENT OPTIONS FOUND: {len(replacement_options)} affected flights
        
#         Provide a professional summary including:
#         1. Overview of the disruption impact
#         2. Number of affected flights
#         3. Best overall recommendations
#         4. Key risks or constraints
#         5. Next steps suggested
        
#         Format for an airline operations manager.
#         """
        
#         response = self.llm.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model=self.model_name,
#             temperature=0.4
#         )
        
#         return response.choices[0].message.content
    
#     # Helper methods
#     def _is_date_in_range(self, date_str: str, start_date: str, end_date: str) -> bool:
#         """Check if date is within range"""
#         try:
#             date_obj = pd.to_datetime(date_str).date()
#             start_obj = pd.to_datetime(start_date).date()
#             end_obj = pd.to_datetime(end_date).date()
#             return start_obj <= date_obj <= end_obj
#         except:
#             return False
    
#     def _is_crew_available(self, crew_id: str, date: str, roster_data: List[Dict]) -> bool:
#         """Check if crew is available on given date (simplified)"""
#         for flight in roster_data:
#             if flight['Date'] == date:
#                 for crew in flight['Crew_Members']:
#                     if crew['Crew_ID'] == crew_id:
#                         return False  # Already assigned
#         return True
    
#     def _get_replacement_reasons(self, candidate_crew, original_crew, flight) -> List[str]:
#         """Get reasons why this candidate is suitable"""
#         reasons = []
        
#         try:
#             if candidate_crew['Base'] == flight['Origin']:
#                 reasons.append(f"Same base ({flight['Origin']})")
            
#             if candidate_crew['Rank'] == original_crew['Rank']:
#                 reasons.append(f"Same rank ({candidate_crew['Rank']})")
            
#             aircraft_types = str(candidate_crew['Aircraft_Type_License']).split(', ')
#             if flight['Aircraft_Type'] in [at.strip() for at in aircraft_types]:
#                 reasons.append(f"Qualified for {flight['Aircraft_Type']}")
#         except:
#             pass
        
#         return reasons
    
#     def _get_replacement_warnings(self, candidate_crew, flight, roster_data) -> List[str]:
#         """Get warnings about this replacement"""
#         warnings = []
        
#         try:
#             if self._would_exceed_duty_hours(candidate_crew['Crew_ID'], flight, roster_data):
#                 warnings.append("May exceed duty hour limits")
            
#             if candidate_crew['Base'] != flight['Origin']:
#                 warnings.append(f"Different base ({candidate_crew['Base']} vs {flight['Origin']})")
#         except:
#             pass
        
#         return warnings
    
#     def _would_exceed_duty_hours(self, crew_id: str, new_flight: Dict, roster_data: List[Dict]) -> bool:
#         """Check if adding this flight would exceed duty hours (simplified)"""
#         try:
#             existing_duty_hours = self._calculate_existing_duty_hours(crew_id, new_flight['Date'], roster_data)
#             new_flight_hours = self._estimate_flight_duration(new_flight)
#             return (existing_duty_hours + new_flight_hours) > 10  # DGCA limit
#         except:
#             return False
    
#     def _calculate_existing_duty_hours(self, crew_id: str, date: str, roster_data: List[Dict]) -> float:
#         """Calculate existing duty hours for crew on given date"""
#         total_hours = 0.0
#         for flight in roster_data:
#             if flight['Date'] == date:
#                 for crew in flight['Crew_Members']:
#                     if crew['Crew_ID'] == crew_id:
#                         total_hours += self._estimate_flight_duration(flight)
#         return total_hours
    
#     def _estimate_flight_duration(self, flight: Dict) -> float:
#         """Estimate flight duration in hours"""
#         try:
#             if 'Duration' in flight and flight['Duration']:
#                 if ':' in flight['Duration']:
#                     hours, minutes = map(int, flight['Duration'].split(':'))
#                     return hours + minutes / 60
#             return 2.0  # Default fallback
#         except:
#             return 2.0



# app/core/llm_chatbot.py
import json
from typing import Dict, List, Optional
import logging
import pandas as pd
from groq import Groq
from app.data.load_data import data_loader

logger = logging.getLogger(__name__)

class CrewDisruptionChatbot:
    def __init__(self):
        self.llm_available = False
        self.llm = None
        
        try:
            # HARDCODED API KEY - Replace with your actual key
            api_key = ""
            model_name = "gemma2-9b-it"
            
            if not api_key or api_key == "":
                logger.warning("Groq API key not configured. LLM features disabled.")
                return
                
            self.llm = Groq(api_key=api_key)
            self.model_name = model_name
            self.llm_available = True
            logger.info(f"Groq API initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq API: {e}")
            self.llm_available = False
        
    async def handle_disruption_request(self, disruption_request: Dict, roster_data: List[Dict]) -> Dict:
        """Handle a disruption request using LLM"""
        if not self.llm_available:
            return {
                'success': False,
                'error': 'Groq API not configured.',
                'fallback_options': self._get_fallback_options(disruption_request, roster_data)
            }
        
        try:
            # Step 1: Parse and understand the request
            structured_request = await self._parse_disruption_request(disruption_request)
            
            # Step 2: Find affected flights
            affected_flights = self._find_affected_flights(roster_data, structured_request)
            
            # Step 3: Find replacement candidates
            replacement_options = []
            for flight in affected_flights:
                candidates = self._find_replacement_candidates(flight, structured_request, roster_data)
                if candidates:  # Only analyze if we have candidates
                    llm_analysis = await self._analyze_candidates_with_llm(flight, candidates, structured_request)
                    replacement_options.append({
                        'flight': flight,
                        'candidates': candidates,
                        'llm_analysis': llm_analysis
                    })
            
            # Step 4: Generate summary response
            summary = await self._generate_summary_response(structured_request, replacement_options)
            
            return {
                'success': True,
                'summary': summary,
                'affected_flights': affected_flights,
                'replacement_options': replacement_options,
                'structured_request': structured_request
            }
            
        except Exception as e:
            logger.error(f"Error in disruption chatbot: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_fallback_options(self, disruption_request: Dict, roster_data: List[Dict]) -> Dict:
        """Provide basic fallback options without LLM"""
        affected_crew = disruption_request.get('crew_id')
        affected_flights = []
        
        if affected_crew:
            for flight in roster_data:
                for crew_member in flight['Crew_Members']:
                    if crew_member['Crew_ID'] == affected_crew:
                        affected_flights.append({
                            'flight_number': flight['Flight_Number'],
                            'date': flight['Date'],
                            'origin': flight['Origin'],
                            'destination': flight['Destination'],
                            'aircraft_type': flight['Aircraft_Type']
                        })
                        break
        
        return {
            'affected_crew': affected_crew,
            'affected_flights': affected_flights,
            'message': 'LLM not available. Basic disruption information provided.'
        }
    
    async def _parse_disruption_request(self, request: Dict) -> Dict:
        """Use LLM to parse natural language disruption request"""
        if not self.llm_available:
            # Basic parsing without LLM
            return {
                'crew_id': request.get('crew_id'),
                'flight_numbers': [],
                'start_date': request.get('start_date'),
                'end_date': request.get('end_date'),
                'reason': request.get('reason', 'unknown'),
                'disruption_type': request.get('disruption_type', 'unknown'),
                'severity': 'medium'
            }
        
        prompt = f"""
        Analyze this crew disruption message and extract structured information:
        
        MESSAGE: {json.dumps(request, indent=2)}
        
        Extract the following information as JSON:
        - crew_id (if mentioned, e.g., "FO002")
        - flight_numbers (array of flight numbers if mentioned)
        - start_date (date when disruption starts)
        - end_date (date when disruption ends) 
        - reason (sickness, technical, weather, etc.)
        - disruption_type
        - severity (high, medium, low)
        
        Return ONLY valid JSON format.
        """
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _find_affected_flights(self, roster_data: List[Dict], request: Dict) -> List[Dict]:
        """Find flights affected by the disruption"""
        affected_flights = []
        crew_id = request.get('crew_id')
        
        if not crew_id:
            return affected_flights
        
        for flight in roster_data:
            for crew_member in flight['Crew_Members']:
                if (crew_member['Crew_ID'] == crew_id and
                    self._is_date_in_range(flight['Date'], request['start_date'], request['end_date'])):
                    affected_flights.append(flight)
                    break
        
        return affected_flights
    
    def _find_replacement_candidates(self, flight: Dict, request: Dict, roster_data: List[Dict]) -> List[Dict]:
        """Find suitable replacement candidates for a flight"""
        candidates = []
        original_crew_id = request.get('crew_id')
        
        if not original_crew_id:
            return candidates
        
        try:
            # Get original crew info
            original_crew_df = data_loader.crew_df[data_loader.crew_df['Crew_ID'] == original_crew_id]
            if original_crew_df.empty:
                return candidates
                
            original_crew = original_crew_df.iloc[0]
            
            # Find potential replacements
            for _, crew in data_loader.crew_df.iterrows():
                if (crew['Crew_ID'] != original_crew_id and 
                    self._is_suitable_replacement(crew, original_crew, flight, roster_data)):
                    
                    candidate = {
                        'crew_id': crew['Crew_ID'],
                        'name': crew['Name'],
                        'rank': crew['Rank'],
                        'base': crew['Base'],
                        'score': self._calculate_suitability_score(crew, original_crew, flight, roster_data),
                        'qualifications': crew['Aircraft_Type_License'],
                        'reasons': self._get_replacement_reasons(crew, original_crew, flight),
                        'warnings': self._get_replacement_warnings(crew, flight, roster_data)
                    }
                    candidates.append(candidate)
            
            # Sort by score (descending)
            return sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]  # Top 5 candidates
            
        except Exception as e:
            logger.error(f"Error finding replacement candidates: {e}")
            return []
    
    def _is_suitable_replacement(self, candidate_crew, original_crew, flight, roster_data) -> bool:
        """Check if a crew member is suitable for replacement"""
        try:
            # Same rank
            if candidate_crew['Rank'] != original_crew['Rank']:
                return False
            
            # Aircraft qualification
            aircraft_types = str(candidate_crew['Aircraft_Type_License']).split(', ')
            if flight['Aircraft_Type'] not in [at.strip() for at in aircraft_types]:
                return False
            
            # Check availability
            if not self._is_crew_available(candidate_crew['Crew_ID'], flight['Date'], roster_data):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in suitability check: {e}")
            return False
    
    def _calculate_suitability_score(self, candidate_crew, original_crew, flight, roster_data) -> float:
        """Calculate suitability score (0-100)"""
        try:
            score = 60  # Base score
            
            # Same base bonus
            if candidate_crew['Base'] == flight['Origin']:
                score += 20
            
            # Experience bonus (simplified)
            if 'Senior' in candidate_crew['Rank']:
                score += 15
            
            # Availability bonus
            if self._is_crew_highly_available(candidate_crew['Crew_ID'], flight['Date'], roster_data):
                score += 10
            
            # Check for duty time concerns
            if self._would_exceed_duty_hours(candidate_crew['Crew_ID'], flight, roster_data):
                score -= 25
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating suitability score: {e}")
            return 50  # Default score on error
    
    def _is_crew_highly_available(self, crew_id: str, date: str, roster_data: List[Dict]) -> bool:
        """Check if crew is highly available (simplified)"""
        # Count assignments on this date
        assignment_count = 0
        for flight in roster_data:
            if flight['Date'] == date:
                for crew in flight['Crew_Members']:
                    if crew['Crew_ID'] == crew_id:
                        assignment_count += 1
        return assignment_count == 0  # Highly available if no assignments
    
    async def _analyze_candidates_with_llm(self, flight: Dict, candidates: List[Dict], request: Dict) -> str:
        """Use LLM to analyze and rank candidates"""
        if not self.llm_available:
            return "LLM analysis not available. Please configure Groq API."
        
        prompt = f"""
        Analyze these crew replacement options and provide intelligent recommendations:
        
        FLIGHT: {flight['Flight_Number']} ({flight['Origin']}-{flight['Destination']})
        AIRCRAFT: {flight['Aircraft_Type']}
        DATE: {flight['Date']}
        DISRUPTION: {request.get('reason', 'Unknown')}
        
        CANDIDATES: {json.dumps(candidates, indent=2)}
        
        Provide a concise analysis with:
        1. Top recommendation with score
        2. Key advantages of the best candidate
        3. Potential concerns to consider
        4. DGCA compliance summary
        
        Format for easy reading with bullet points.
        """
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    async def _generate_summary_response(self, request: Dict, replacement_options: List[Dict]) -> str:
        """Generate summary response using LLM"""
        if not self.llm_available:
            return "LLM summary not available. Please configure Groq API."
        
        prompt = f"""
        Generate a comprehensive summary of this crew disruption situation:
        
        DISRUPTION REQUEST: {json.dumps(request, indent=2)}
        
        REPLACEMENT OPTIONS FOUND: {len(replacement_options)} affected flights
        
        Provide a professional summary including:
        1. Overview of the disruption impact
        2. Number of affected flights
        3. Best overall recommendations
        4. Key risks or constraints
        5. Next steps suggested
        
        Format for an airline operations manager.
        """
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    
    # Helper methods
    def _is_date_in_range(self, date_str: str, start_date: str, end_date: str) -> bool:
        """Check if date is within range"""
        try:
            date_obj = pd.to_datetime(date_str).date()
            start_obj = pd.to_datetime(start_date).date()
            end_obj = pd.to_datetime(end_date).date()
            return start_obj <= date_obj <= end_obj
        except:
            return False
    
    def _is_crew_available(self, crew_id: str, date: str, roster_data: List[Dict]) -> bool:
        """Check if crew is available on given date (simplified)"""
        for flight in roster_data:
            if flight['Date'] == date:
                for crew in flight['Crew_Members']:
                    if crew['Crew_ID'] == crew_id:
                        return False  # Already assigned
        return True
    
    def _get_replacement_reasons(self, candidate_crew, original_crew, flight) -> List[str]:
        """Get reasons why this candidate is suitable"""
        reasons = []
        
        try:
            if candidate_crew['Base'] == flight['Origin']:
                reasons.append(f"Same base ({flight['Origin']})")
            
            if candidate_crew['Rank'] == original_crew['Rank']:
                reasons.append(f"Same rank ({candidate_crew['Rank']})")
            
            aircraft_types = str(candidate_crew['Aircraft_Type_License']).split(', ')
            if flight['Aircraft_Type'] in [at.strip() for at in aircraft_types]:
                reasons.append(f"Qualified for {flight['Aircraft_Type']}")
        except:
            pass
        
        return reasons
    
    def _get_replacement_warnings(self, candidate_crew, flight, roster_data) -> List[str]:
        """Get warnings about this replacement"""
        warnings = []
        
        try:
            if self._would_exceed_duty_hours(candidate_crew['Crew_ID'], flight, roster_data):
                warnings.append("May exceed duty hour limits")
            
            if candidate_crew['Base'] != flight['Origin']:
                warnings.append(f"Different base ({candidate_crew['Base']} vs {flight['Origin']})")
        except:
            pass
        
        return warnings
    
    def _would_exceed_duty_hours(self, crew_id: str, new_flight: Dict, roster_data: List[Dict]) -> bool:
        """Check if adding this flight would exceed duty hours (simplified)"""
        try:
            existing_duty_hours = self._calculate_existing_duty_hours(crew_id, new_flight['Date'], roster_data)
            new_flight_hours = self._estimate_flight_duration(new_flight)
            return (existing_duty_hours + new_flight_hours) > 10  # DGCA limit
        except:
            return False
    
    def _calculate_existing_duty_hours(self, crew_id: str, date: str, roster_data: List[Dict]) -> float:
        """Calculate existing duty hours for crew on given date"""
        total_hours = 0.0
        for flight in roster_data:
            if flight['Date'] == date:
                for crew in flight['Crew_Members']:
                    if crew['Crew_ID'] == crew_id:
                        total_hours += self._estimate_flight_duration(flight)
        return total_hours
    
    def _estimate_flight_duration(self, flight: Dict) -> float:
        """Estimate flight duration in hours"""
        try:
            if 'Duration' in flight and flight['Duration']:
                if ':' in flight['Duration']:
                    hours, minutes = map(int, flight['Duration'].split(':'))
                    return hours + minutes / 60
            return 2.0  # Default fallback
        except:
            return 2.0