# app/core/genetic_algorithm.py
import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime, timedelta, date
from .dgca_validator import DGCAValidator
from app.utils.helpers import parse_time, calculate_duty_hours, parse_datetime_from_string,preprocess_datetime_columns
import logging
from app.utils.constants import MinimumCrewRequirements
import traceback

logger = logging.getLogger(__name__)

class CrewRosteringGA:
    def __init__(self, flights_df, crew_df, preferences_df, validator):
        self.flights_df = flights_df
        self.crew_df = crew_df
        self.preferences_df = preferences_df
        self.validator = validator
        self.available_crew = self._get_available_crew()
        
        # Initialize DEAP
        self._setup_deap()
    
    def _get_available_crew(self) -> List[str]:
        """Get list of available crew members with proper date comparison"""
        try:
            current_date = datetime.now().date()
            
            # Convert current_date to pandas Timestamp for proper comparison
            current_date_ts = pd.Timestamp(current_date)
            
            # Create a mask for available crew
            # Crew is available if:
            # 1. Leave_Start is null OR
            # 2. Leave_Start is after current date OR  
            # 3. Leave_End is before current date
            available_mask = (
                self.crew_df['Leave_Start'].isna() |
                (self.crew_df['Leave_Start'] > current_date_ts) |
                (self.crew_df['Leave_End'] < current_date_ts)
            )
            
            available_crew = self.crew_df[available_mask]['Crew_ID'].tolist()
            logger.info(f"Found {len(available_crew)} available crew members")
            return available_crew
            
        except Exception as e:
            logger.error(f"Error getting available crew: {e}")
            # Return all crew as fallback
            return self.crew_df['Crew_ID'].tolist()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("individual", self._generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_individual)


# # Add these new methods to the CrewRosteringGA class
#     def _find_suitable_cockpit_crew(self, flight: pd.Series, min_count: int) -> List[str]:
#         """Find suitable cockpit crew (pilots) for a given flight"""
#         suitable_crew = []
        
#         for crew_id in self.available_crew:
#             crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
#             # Check if crew is a pilot (Captain or First Officer)
#             if crew_info['Rank'] in ['Captain', 'First Officer']:
#                 # Check aircraft qualification
#                 aircraft_types = str(crew_info['Aircraft_Type_License']).split(', ')
#                 if flight['Aircraft_Type'] in aircraft_types:
#                     # Check if crew is at the right base
#                     if crew_info['Base'] == flight['Origin']:
#                         suitable_crew.append(crew_id)
        
#         # Return at least min_count crew members if available
#         if len(suitable_crew) >= min_count:
#             return random.sample(suitable_crew, min_count)
#         else:
#             return None



    # def _find_suitable_cabin_crew(self, flight: pd.Series, min_count: int) -> List[str]:
    #     """Find suitable cabin crew for a given flight"""
    #     suitable_crew = []
        
    #     for crew_id in self.available_crew:
    #         crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
    #         # Check if crew is cabin crew (not pilot)
    #         if crew_info['Rank'] not in ['Captain', 'First Officer']:
    #             # Check aircraft qualification
    #             aircraft_types = str(crew_info['Aircraft_Type_License']).split(', ')
    #             if flight['Aircraft_Type'] in aircraft_types:
    #                 # Check if crew is at the right base
    #                 if crew_info['Base'] == flight['Origin']:
    #                     suitable_crew.append(crew_id)
        
    #     # Return at least min_count crew members if available
    #     if len(suitable_crew) >= min_count:
    #         return random.sample(suitable_crew, min_count)
    #     else:
    #         return None
        

    # app/core/genetic_algorithm.py - Update these methods

    def _find_suitable_cockpit_crew(self, flight: pd.Series, min_count: int) -> List[str]:
        """Find suitable cockpit crew with proper rank requirements"""
        suitable_crew = []
        requirements = self._get_minimum_crew_requirements(flight['Aircraft_Type'])
        cockpit_ranks = requirements['cockpit']['ranks']
        
        for crew_id in self.available_crew:
            crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
            # Check if crew is a pilot with required rank
            if crew_info['Rank'] in cockpit_ranks:
                # Check aircraft qualification
                aircraft_types = str(crew_info['Aircraft_Type_License']).split(', ')
                if flight['Aircraft_Type'] in aircraft_types:
                    # Check if crew is at the right base
                    if crew_info['Base'] == flight['Origin']:
                        suitable_crew.append(crew_id)
        
        # Return at least min_count crew members if available
        if len(suitable_crew) >= min_count:
            return random.sample(suitable_crew, min_count)
        else:
            return None

    def _find_suitable_cabin_crew(self, flight: pd.Series, min_count: int) -> List[str]:
        """Find suitable cabin crew with proper rank requirements"""
        suitable_crew = []
        requirements = self._get_minimum_crew_requirements(flight['Aircraft_Type'])
        cabin_ranks = requirements['cabin']['ranks']
        
        for crew_id in self.available_crew:
            crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
            # Check if crew is cabin crew with required rank
            if crew_info['Rank'] in cabin_ranks:
                # Check aircraft qualification
                aircraft_types = str(crew_info['Aircraft_Type_License']).split(', ')
                if flight['Aircraft_Type'] in aircraft_types:
                    # Check if crew is at the right base
                    if crew_info['Base'] == flight['Origin']:
                        suitable_crew.append(crew_id)
        
        # Return at least min_count crew members if available
        if len(suitable_crew) >= min_count:
            return random.sample(suitable_crew, min_count)
        else:
            return None
        
    def _find_available_cabin_crew(self, flight: pd.Series, min_count: int, flight_date: str, duty_hours: float) -> List[str]:
        """Find available cabin crew considering daily limits"""
        suitable_crew = []
        requirements = self._get_minimum_crew_requirements(flight['Aircraft_Type'])
        cabin_ranks = requirements['cabin']['ranks']
        
        for crew_id in self.available_crew:
            crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
            # Check if crew is cabin crew with required rank and available
            if (crew_info['Rank'] in cabin_ranks and 
                self._is_crew_available(crew_id, flight_date, duty_hours)):
                
                # Check aircraft qualification
                aircraft_types = str(crew_info['Aircraft_Type_License']).split(', ')
                if flight['Aircraft_Type'] in aircraft_types:
                    # Check if crew is at the right base
                    if crew_info['Base'] == flight['Origin']:
                        suitable_crew.append(crew_id)
        
        # Return at least min_count crew members if available
        if len(suitable_crew) >= min_count:
            return random.sample(suitable_crew, min_count)
        else:
            return None
    # app/core/genetic_algorithm.py - Add these methods

    def __init__(self, flights_df, crew_df, preferences_df, validator):
        self.flights_df = flights_df
        self.crew_df = crew_df
        self.preferences_df = preferences_df
        self.validator = validator
        self.available_crew = self._get_available_crew()
        self.crew_assignments = {}  # Track crew assignments by date
        self._initialize_crew_tracking()
        
        # Initialize DEAP
        self._setup_deap()

    def _initialize_crew_tracking(self):
        """Initialize crew assignment tracking"""
        self.crew_assignments = {}
        for crew_id in self.available_crew:
            self.crew_assignments[crew_id] = {
                'total_assignments': 0,
                'daily_assignments': {},
                'last_assignment_date': None
            }

    def _update_crew_tracking(self, crew_id: str, flight_date: str, duty_hours: float):
        """Update crew assignment tracking"""
        if crew_id not in self.crew_assignments:
            self.crew_assignments[crew_id] = {
                'total_assignments': 0,
                'daily_assignments': {},
                'last_assignment_date': None
            }
        
        self.crew_assignments[crew_id]['total_assignments'] += 1
        self.crew_assignments[crew_id]['last_assignment_date'] = flight_date
        
        if flight_date not in self.crew_assignments[crew_id]['daily_assignments']:
            self.crew_assignments[crew_id]['daily_assignments'][flight_date] = 0
        
        self.crew_assignments[crew_id]['daily_assignments'][flight_date] += duty_hours

    def _is_crew_available(self, crew_id: str, flight_date: str, duty_hours: float) -> bool:
        """Check if crew is available for assignment considering daily limits"""
        if crew_id not in self.crew_assignments:
            return True
        
        # Check daily duty hours limit (max 10 hours)
        daily_hours = self.crew_assignments[crew_id]['daily_assignments'].get(flight_date, 0)
        if daily_hours + duty_hours > 10:
            return False
        
        return True



    def _calculate_violation_penalty(self, roster) -> float:
        """Calculate penalty for DGCA violations with robust format handling"""
        try:
            # Check if roster is a DataFrame and convert to list of dicts if needed
            if isinstance(roster, pd.DataFrame):
                # Convert DataFrame to the expected grouped format
                grouped_roster = self._convert_dataframe_to_grouped(roster)
                violations = self.validator.validate_roster(grouped_roster, self.crew_df)
            elif isinstance(roster, list):
                # Already in the expected format
                violations = self.validator.validate_roster(roster, self.crew_df)
            else:
                logger.error(f"Unknown roster format: {type(roster)}")
                return 10000  # Large penalty for invalid format
            
            return len(violations) * 1000  # Heavy penalty for violations
        except Exception as e:
            logger.error(f"Error calculating violation penalty: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 10000  # Large penalty if validation fails completely
        

    def _convert_dataframe_to_grouped(self, roster_df: pd.DataFrame) -> List[Dict]:
        """Convert a flat DataFrame roster to the grouped format expected by the validator"""
        if roster_df.empty:
            return []
        
        grouped_roster = []
        
        # Group by flight
        for flight_key in roster_df[['Date', 'Flight_Number']].drop_duplicates().itertuples(index=False):
            date, flight_number = flight_key
            flight_assignments = roster_df[(roster_df['Date'] == date) & (roster_df['Flight_Number'] == flight_number)]
            
            # Get flight details from the first row
            first_row = flight_assignments.iloc[0]
            flight_data = {
                'Date': date,
                'Flight_Number': flight_number,
                'Duty_Start': first_row.get('Duty_Start'),
                'Duty_End': first_row.get('Duty_End'),
                'Aircraft_Type': first_row.get('Aircraft_Type'),
                'Origin': first_row.get('Origin'),
                'Destination': first_row.get('Destination'),
                'Duration': first_row.get('Duration'),
                'Crew_Members': []
            }
            
            # Add crew members
            for _, row in flight_assignments.iterrows():
                crew_info = self.crew_df[self.crew_df['Crew_ID'] == row['Crew_ID']].iloc[0]
                flight_data['Crew_Members'].append({
                    'Crew_ID': row['Crew_ID'],
                    'Crew_Rank': crew_info['Rank']
                })
            
            grouped_roster.append(flight_data)
        
        return grouped_roster


    def _find_suitable_crew(self, flight: pd.Series) -> List[str]:
        """Find crew suitable for a given flight"""
        suitable_crew = []
        
        for crew_id in self.available_crew:
            crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
            # Check aircraft qualification
            aircraft_types = str(crew_info['Aircraft_Type_License']).split(', ')
            if flight['Aircraft_Type'] in aircraft_types:
                # Check if crew is at the right base
                if crew_info['Base'] == flight['Origin']:
                    suitable_crew.append(crew_id)
        
        return suitable_crew
    
    def _mutate_individual(self, individual, indpb: float):
        """Mutate an individual with given probability"""
        for i in range(len(individual)):
            if random.random() < indpb:
                flight_number, _ = individual[i]
                flight = self.flights_df[self.flights_df['Flight_Number'] == flight_number].iloc[0]
                
                # Get minimum crew requirements
                requirements = self._get_minimum_crew_requirements(flight['Aircraft_Type'])
                
                # Find all suitable crew (both cockpit and cabin)
                cockpit_crew = self._find_suitable_cockpit_crew(flight, requirements['cockpit']['min'])
                cabin_crew = self._find_suitable_cabin_crew(flight, requirements['cabin']['min'])
                
                # Combine both lists
                suitable_crew = cockpit_crew + cabin_crew if cockpit_crew and cabin_crew else []
                
                if suitable_crew:
                    individual[i] = (flight_number, random.choice(suitable_crew))
        
        return individual,
    
 
    def _evaluate_individual(self, individual):
        """Evaluate the fitness of an individual"""
        # Convert individual to roster format
        roster = self._individual_to_roster(individual)
        
        # Calculate fitness components
        violation_penalty = self._calculate_violation_penalty(roster)
        preference_score = self._calculate_preference_score(roster)
        fairness_score = self._calculate_fairness_score(roster)
        over_assignment_penalty = self._calculate_over_assignment_penalty()
        
        # Return tuple with 3 values to match the 3 weights
        return (violation_penalty + over_assignment_penalty, 
                preference_score, 
                fairness_score)
    
    def _calculate_over_assignment_penalty(self) -> float:
        """Calculate penalty for over-assigning crew"""
        penalty = 0
        
        for crew_id, assignments in self.crew_assignments.items():
            # Penalize crew with too many assignments
            if assignments['total_assignments'] > 4:  # Max 4 assignments per crew
                penalty += (assignments['total_assignments'] - 4) * 500
            
            # Penalize crew with excessive daily hours
            for date, hours in assignments['daily_assignments'].items():
                if hours > 10:  # Max 10 hours per day
                    penalty += (hours - 10) * 1000
        
        return penalty
   
    def _individual_to_roster(self, individual) -> List[Dict]:
        """Convert individual to roster with crew members grouped by flight"""
        flight_assignments = {}
        
        for flight_number, crew_id in individual:
            if crew_id is None:
                continue
                
            # Find the flight in the DataFrame
            flight_row = self.flights_df[self.flights_df['Flight_Number'] == flight_number]
            if flight_row.empty:
                continue
                
            flight = flight_row.iloc[0]
            crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
            # Create flight key
            flight_key = f"{flight['Date']}_{flight_number}"
            
            if flight_key not in flight_assignments:
                # Get date and times with proper handling
                flight_date = flight['Date']
                
                # Ensure flight_date is a proper date object and convert to string for JSON
                if isinstance(flight_date, pd.Timestamp):
                    flight_date_str = flight_date.strftime('%Y-%m-%d')
                    flight_date_obj = flight_date.date()
                elif isinstance(flight_date, str):
                    flight_date_obj = pd.to_datetime(flight_date).date()
                    flight_date_str = flight_date
                else:
                    flight_date_str = str(flight_date)
                    flight_date_obj = flight_date
                
                # Handle time objects
                departure_time = parse_time(flight['Scheduled_Departure_UTC'])
                arrival_time = parse_time(flight['Scheduled_Arrival_UTC'])
                
                # Create datetime objects for duty start/end
                duty_start = None
                duty_end = None
                
                if departure_time:
                    duty_start = datetime.combine(flight_date_obj, departure_time)
                if arrival_time:
                    duty_end = datetime.combine(flight_date_obj, arrival_time)
                    # Handle overnight flights (arrival next day)
                    if duty_start and duty_end < duty_start:
                        duty_end += timedelta(days=1)
                
                # Convert datetime objects to strings for JSON serialization
                duty_start_str = duty_start.isoformat() if duty_start else None
                duty_end_str = duty_end.isoformat() if duty_end else None
                
                flight_assignments[flight_key] = {
                    'Date': flight_date_str,
                    'Flight_Number': flight_number,
                    'Duty_Start': duty_start_str,
                    'Duty_End': duty_end_str,
                    'Aircraft_Type': flight['Aircraft_Type'],
                    'Origin': flight['Origin'],
                    'Destination': flight['Destination'],
                    'Duration': flight['Duration_HH_MM'],
                    'Crew_Members': []  # Initialize empty crew list
                }
            
            # Add crew member to the flight
            flight_assignments[flight_key]['Crew_Members'].append({
                'Crew_ID': crew_id,
                'Crew_Rank': crew_info['Rank']
            })
        
        # Convert to list of dictionaries
        return list(flight_assignments.values())

    # app/core/dgca_validator.py (add a method to flatten the roster)
    def _flatten_roster_for_validation(self, roster: List[Dict]) -> pd.DataFrame:
        """Convert grouped roster to flat format for validation"""
        flat_data = []
        
        for flight in roster:
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

    def validate_roster(self, roster: List[Dict], crew_data: pd.DataFrame) -> List[str]:
        """Validate a roster against DGCA rules with minimal RAG usage"""
        violations = []
        
        # Check if roster is empty
        if not roster:
            return violations
        
        # Convert the grouped roster to a flat format for validation
        try:
            flat_roster = self._flatten_grouped_roster(roster)
            
            # Check if flat_roster is a DataFrame and has the required columns
            if not isinstance(flat_roster, pd.DataFrame) or 'Crew_ID' not in flat_roster.columns:
                logger.error("Invalid roster format for validation")
                return ["Invalid roster format - cannot validate"]
            
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
        
        except Exception as e:
            logger.error(f"Error validating roster: {e}")
            violations.append(f"Validation error: {str(e)}")
        
        return violations  

    def _flatten_grouped_roster_for_preferences(self, grouped_roster: List[Dict]) -> pd.DataFrame:
        """Convert grouped roster to flat DataFrame for preference checking"""
        flat_data = []
        
        for flight in grouped_roster:
            for crew_member in flight['Crew_Members']:
                flat_data.append({
                    'Date': flight['Date'],
                    'Flight_Number': flight['Flight_Number'],
                    'Crew_ID': crew_member['Crew_ID'],
                    'Origin': flight['Origin'],
                    'Destination': flight['Destination']
                })
        
        return pd.DataFrame(flat_data)


    def _calculate_preference_score(self, roster: List[Dict]) -> float:
        """Calculate score based on crew preferences with proper roster format handling"""
        score = 0
        total_preferences = len(self.preferences_df)
        
        if total_preferences == 0:
            return 0
        
        # Convert grouped roster to flat format for preference checking
        flat_roster = self._flatten_grouped_roster_for_preferences(roster)
        
        for _, pref in self.preferences_df.iterrows():
            crew_id = pref['Crew_ID']
            
            # Find crew assignments in the flat roster
            crew_assignments = flat_roster[flat_roster['Crew_ID'] == crew_id]
            
            if crew_assignments.empty:
                continue
                
            if pref['Preference_Type'] == 'Day_Off':
                # Check if crew has day off on preferred date
                try:
                    preferred_date = pd.to_datetime(pref['Preference_Detail']).date()
                    
                    # Check if crew has any assignments on this date
                    has_assignment_on_date = False
                    for _, assignment in crew_assignments.iterrows():
                        assignment_date = pd.to_datetime(assignment['Date']).date()
                        if assignment_date == preferred_date:
                            has_assignment_on_date = True
                            break
                    
                    if not has_assignment_on_date:
                        score += 1
                except Exception as e:
                    logger.warning(f"Error processing day off preference for {crew_id}: {e}")
            
            elif pref['Preference_Type'] == 'Preferred_Sector':
                # Check if crew is assigned to preferred sector
                try:
                    origin, dest = pref['Preference_Detail'].split('-')
                    for _, assignment in crew_assignments.iterrows():
                        if assignment['Origin'] == origin and assignment['Destination'] == dest:
                            score += 1
                            break
                except Exception as e:
                    logger.warning(f"Error processing sector preference for {crew_id}: {e}")
        
        # Normalize score
        return (total_preferences - score) / total_preferences * 100 if total_preferences > 0 else 0
        
        

    def _calculate_fairness_score(self, roster: List[Dict]) -> float:
        """Calculate fairness of workload distribution with grouped roster format"""
        if not roster:
            return 0
        
        # Calculate duty hours per crew
        crew_hours = {}
        
        for flight in roster:
            for crew_member in flight['Crew_Members']:
                crew_id = crew_member['Crew_ID']
                
                # Calculate duty hours for this flight
                duty_start = flight.get('Duty_Start')
                duty_end = flight.get('Duty_End')
                duty_hours = calculate_duty_hours(duty_start, duty_end)
                
                if crew_id not in crew_hours:
                    crew_hours[crew_id] = 0.0
                crew_hours[crew_id] += duty_hours
        
        hours = list(crew_hours.values())
        
        if not hours:
            return 0
        
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
        
        # Ensure score is within reasonable bounds
        fairness_score = max(0, min(100, fairness_score))
        
        return fairness_score
            
    
    # app/core/genetic_algorithm.py (add this method to CrewRosteringGA class)
    def optimize_roster(self, population_size=30, generations=20, 
                       crossover_prob=0.7, mutation_prob=0.2) -> Dict:
        """Run the genetic algorithm optimization with RAG cache management"""
        try:
            pop = self.toolbox.population(n=population_size)
            
            # Statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            
            # Run algorithm with generation-based cache clearing
            for gen in range(generations):
                # Clear RAG cache every few generations to prevent memory issues
                if gen % 5 == 0 and hasattr(self.validator, 'rag_validator'):
                    self.validator.rag_validator.clear_cache()
                
                # Select the next generation individuals
                offspring = self.toolbox.select(pop, len(pop))
                # Clone the selected individuals
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < crossover_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < mutation_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace population
                pop[:] = offspring
                
                # Gather all the fitnesses in one list and print the stats
                fits = [ind.fitness.values[0] for ind in pop]
                
                if gen % 10 == 0:
                    logger.info(f"Generation {gen}: Min fitness {min(fits):.2f}, Avg fitness {np.mean(fits):.2f}")
            
            # Get best individual
            best_individual = tools.selBest(pop, 1)[0]
            best_roster = self._individual_to_roster(best_individual)
            violations = self.validator.validate_roster(best_roster, self.crew_df)
            
            return {
                #roster': best_roster.to_dict('records'),
                'roster': best_roster,  # This is now a list, not a DataFrame                
                'fitness': best_individual.fitness.values,
                'violations': violations,
                'logbook': {"generations": generations}
            }
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            logger.error(f"Error in genetic algorithm optimization: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty result with high penalty
            return {
                'roster': [],
                'fitness': (10000, 10000, 10000),
                'violations': ["Optimization failed: " + str(e)],
                'logbook': {}
            }
    # app/core/genetic_algorithm.py (add this method as a fallback)
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

# # Update the _generate_individual method to use the fallback
#     def _generate_individual(self):
#         """Generate a random individual (roster solution) with proper crew composition"""
#         individual = []
        
#         for _, flight in self.flights_df.iterrows():
#             # Get minimum crew requirements for this aircraft type
#             requirements = self._get_minimum_crew_requirements(flight['Aircraft_Type'])
            
#             # Find suitable cockpit crew (pilots)
#             cockpit_crew = self._find_suitable_cockpit_crew(flight, requirements['cockpit']['min'])
            
#             # Find suitable cabin crew
#             cabin_crew = self._find_suitable_cabin_crew(flight, requirements['cabin']['min'])
            
#             if cockpit_crew and cabin_crew:
#                 # Assign all required crew members
#                 for crew_member in cockpit_crew + cabin_crew:
#                     individual.append((flight['Flight_Number'], crew_member))
#             else:
#                 # If no suitable crew, assign None (will be penalized in fitness)
#                 individual.append((flight['Flight_Number'], None))
        
#         return creator.Individual(individual)



    # app/core/genetic_algorithm.py - Update _generate_individual method

    def _generate_individual(self):
        """Generate a random individual with proper crew composition and availability"""
        individual = []
        self._initialize_crew_tracking()  # Reset tracking for each individual
        
        for _, flight in self.flights_df.iterrows():
            # Get minimum crew requirements for this aircraft type
            requirements = self._get_minimum_crew_requirements(flight['Aircraft_Type'])
            
            # Calculate duty hours for availability check
            duty_hours = self._calculate_flight_duty_hours(flight)
            flight_date = flight['Date'].strftime('%Y-%m-%d') if hasattr(flight['Date'], 'strftime') else str(flight['Date'])
            
            # Find suitable cockpit crew (pilots)
            cockpit_crew = self._find_available_cockpit_crew(flight, requirements['cockpit']['min'], flight_date, duty_hours)
            
            # Find suitable cabin crew
            cabin_crew = self._find_available_cabin_crew(flight, requirements['cabin']['min'], flight_date, duty_hours)
            
            if cockpit_crew and cabin_crew:
                # Assign all required crew members and update tracking
                for crew_member in cockpit_crew + cabin_crew:
                    individual.append((flight['Flight_Number'], crew_member))
                    self._update_crew_tracking(crew_member, flight_date, duty_hours)
            else:
                # If no suitable crew, assign None (will be penalized in fitness)
                individual.append((flight['Flight_Number'], None))
        
        return creator.Individual(individual)

    def _find_available_cockpit_crew(self, flight: pd.Series, min_count: int, flight_date: str, duty_hours: float) -> List[str]:
        """Find available cockpit crew considering daily limits"""
        suitable_crew = []
        requirements = self._get_minimum_crew_requirements(flight['Aircraft_Type'])
        cockpit_ranks = requirements['cockpit']['ranks']
        
        for crew_id in self.available_crew:
            crew_info = self.crew_df[self.crew_df['Crew_ID'] == crew_id].iloc[0]
            
            # Check if crew is a pilot with required rank and available
            if (crew_info['Rank'] in cockpit_ranks and 
                self._is_crew_available(crew_id, flight_date, duty_hours)):
                
                # Check aircraft qualification
                aircraft_types = str(crew_info['Aircraft_Type_License']).split(', ')
                if flight['Aircraft_Type'] in aircraft_types:
                    # Check if crew is at the right base
                    if crew_info['Base'] == flight['Origin']:
                        suitable_crew.append(crew_id)
        
        # Return at least min_count crew members if available
        if len(suitable_crew) >= min_count:
            return random.sample(suitable_crew, min_count)
        else:
            return None

    def _calculate_flight_duty_hours(self, flight: pd.Series) -> float:
        """Calculate duty hours for a flight"""
        try:
            departure = parse_time(flight['Scheduled_Departure_UTC'])
            arrival = parse_time(flight['Scheduled_Arrival_UTC'])
            
            if departure and arrival:
                # Handle overnight flights
                if arrival < departure:
                    arrival_hours = arrival.hour + 24
                else:
                    arrival_hours = arrival.hour
                    
                duty_hours = arrival_hours - departure.hour + (arrival.minute - departure.minute) / 60
                return max(duty_hours, 1.0)  # At least 1 hour
        except:
            pass
        
        return 2.0  # Default fallback        