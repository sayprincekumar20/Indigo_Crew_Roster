# app/core/rag_validator.py
import os
import pdfplumber
import time
import hashlib
from typing import List, Dict, Any
from threading import Lock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from groq import Groq
from app.core.config import settings
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class RAGValidator:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.vector_store = None
        self.rag_available = False
        self._cache = {}
        self._last_api_call = 0
        self._api_lock = Lock()
        self._min_call_interval = 1.0  # Minimum 1 second between API calls
        self._setup_rag_system()
    
    def _setup_rag_system(self):
        """Set up the RAG system with DGCA rules"""
        # Load and process DGCA PDF
        pdf_path = "app/data/sample_data/dgca_rules.pdf"
        if os.path.exists(pdf_path):
            try:
                self._process_dgca_pdf(pdf_path)
                self.rag_available = True
                logger.info("RAG system initialized with DGCA rules")
            except Exception as e:
                logger.error(f"Error processing DGCA PDF: {e}")
                self._create_default_rules()
                logger.info("RAG system using default rules as fallback")
        else:
            logger.warning("DGCA rules PDF not found. Using default rules.")
            self._create_default_rules()
    
    def _create_default_rules(self):
        """Create default DGCA rules as fallback"""
        try:
            default_rules = """
            DGCA Rule 1: Maximum daily flight time shall not exceed 10 hours.
            DGCA Rule 2: Maximum weekly flight time shall not exceed 35 hours.
            DGCA Rule 3: Minimum rest period between duties shall be 12 hours.
            DGCA Rule 4: Weekly rest shall be at least 48 hours including 2 local nights.
            DGCA Rule 5: Night duty (10 PM to 6 AM) requires special consideration and additional rest.
            DGCA Rule 6: Crew must be qualified for the aircraft type they are assigned to.
            DGCA Rule 7: Consecutive night duties should be limited to maximum of 2 nights.
            DGCA Rule 8: After night duty, minimum rest of 12 hours is required before next assignment.
            """
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(default_rules)]
            
            # Create embeddings and vector store with error handling
            try:
                embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=settings.chroma_db_path
                )
                self.rag_available = True
                logger.info("RAG system using default rules")
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                # Fallback: create in-memory store without embeddings
                self.vector_store = None
                self.default_documents = documents
                self.rag_available = False
                logger.warning("RAG system disabled due to embedding errors")
                
        except Exception as e:
            logger.error(f"Error creating default rules: {e}")
            self.rag_available = False
    
    def _process_dgca_pdf(self, pdf_path: str):
        """Process DGCA PDF and create vector store"""
        try:
            # Extract text from PDF
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=settings.chroma_db_path
            )
            
        except Exception as e:
            logger.error(f"Error processing DGCA PDF: {e}")
            raise
    
    def _get_scenario_hash(self, scenario_description: str) -> str:
        """Generate hash for scenario to use as cache key"""
        return hashlib.md5(scenario_description.encode()).hexdigest()
    
    def validate_with_rag(self, scenario_description: str) -> Dict[str, Any]:
        """
        Validate a crew scenario using RAG with Groq API with rate limiting and caching
        """
        # Check cache first
        scenario_hash = self._get_scenario_hash(scenario_description)
        if scenario_hash in self._cache:
            return self._cache[scenario_hash]
        
        if not self.rag_available:
            return self._get_fallback_response(scenario_description)
        
        # Apply rate limiting
        with self._api_lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_api_call
            if time_since_last_call < self._min_call_interval:
                time.sleep(self._min_call_interval - time_since_last_call)
            
            try:
                # Retrieve relevant DGCA rules
                context = ""
                if self.vector_store:
                    relevant_docs = self.vector_store.similarity_search(scenario_description, k=3)
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                elif hasattr(self, 'default_documents'):
                    # Use first few documents as context
                    context = "\n".join([doc.page_content for doc in self.default_documents[:2]])
                else:
                    context = "Default DGCA rules: Max 10h daily flight, 12h min rest, crew must be qualified"
                
                # Create prompt for Groq
                prompt = f"""
                You are an expert aviation compliance analyst for IndiGo Airlines. 
                Your task is to determine if a crew duty scenario violates DGCA rules.

                Use the following aviation rules to base your decision on:
                <context>
                {context}
                </context>

                Crew Duty Scenario to analyze:
                <scenario>
                {scenario_description}
                </scenario>

                First, analyze the scenario step-by-step against the provided rules.
                If there is a violation, state "VIOLATION: [clear explanation]". 
                If there is no violation, state "COMPLIANT: [brief confirmation]".
                If the provided context does not contain enough information, state "INCONCLUSIVE: [reason]".
                """
                
                # Call Groq API
                response = self.groq_client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=800  # Reduced to save tokens
                )
                
                result = response.choices[0].message.content
                
                # Parse the response
                if "VIOLATION:" in result:
                    result_obj = {
                        "status": "violation",
                        "message": result.replace("VIOLATION:", "").strip(),
                        "context": context
                    }
                elif "COMPLIANT:" in result:
                    result_obj = {
                        "status": "compliant",
                        "message": result.replace("COMPLIANT:", "").strip(),
                        "context": context
                    }
                else:
                    result_obj = {
                        "status": "inconclusive",
                        "message": result,
                        "context": context
                    }
                
                # Update last API call time
                self._last_api_call = time.time()
                
            except Exception as e:
                logger.error(f"Error in RAG validation: {e}")
                result_obj = self._get_fallback_response(scenario_description)
        
        # Cache the result
        self._cache[scenario_hash] = result_obj
        return result_obj
    
    def _get_fallback_response(self, scenario_description: str) -> Dict[str, Any]:
        """Get fallback response when API is unavailable"""
        # Simple rule-based fallback
        scenario_lower = scenario_description.lower()
        
        if "night" in scenario_lower and "consecutive" in scenario_lower:
            return {
                "status": "violation",
                "message": "Consecutive night duties may violate DGCA regulations. Maximum 2 consecutive night duties allowed with proper rest.",
                "context": "Fallback rules: DGCA limits consecutive night duties"
            }
        elif "extended" in scenario_lower or "long" in scenario_lower or "hours" in scenario_lower:
            return {
                "status": "violation", 
                "message": "Extended duty periods may exceed DGCA limits. Maximum duty period is typically 14 hours.",
                "context": "Fallback rules: DGCA duty time limitations"
            }
        elif "rest" in scenario_lower and ("insufficient" in scenario_lower or "short" in scenario_lower):
            return {
                "status": "violation",
                "message": "Insufficient rest between duties violates DGCA regulations. Minimum 12 hours rest required.",
                "context": "Fallback rules: DGCA minimum rest requirements"
            }
        else:
            return {
                "status": "inconclusive",
                "message": "Unable to validate with RAG system. Using fallback rules.",
                "context": "Fallback DGCA rules applied"
            }
    
    def validate_complex_scenario(self, roster_data: List[Dict], crew_id: str) -> List[Dict]:
        """
        Validate complex scenarios for a specific crew member using RAG
        with extremely reduced API calls - only validate the most critical cases    
        """
        violations = []
        
        if not self.rag_available:
            return violations
        
        # Get crew schedule
        crew_schedule = [r for r in roster_data if r['Crew_ID'] == crew_id]
        
        if len(crew_schedule) < 2:
            return violations  # No complex scenarios with single duty
        
        # Only validate patterns that are likely to be problematic
        
        # 1. Check for consecutive night duties
        night_duties = []
        for i, duty in enumerate(crew_schedule):
            if self._is_night_duty(duty):
                night_duties.append((i, duty))
        
        # Validate consecutive night duties
        if len(night_duties) >= 2:
            for j in range(len(night_duties) - 1):
                first_idx, first_duty = night_duties[j]
                second_idx, second_duty = night_duties[j + 1]
                
                # Only validate if duties are consecutive or close together
                if second_idx - first_idx <= 2:
                    scenario = f"""
                    Crew member {crew_id} has consecutive night duties:
                    - Duty 1: {first_duty['Date']} from {first_duty['Duty_Start']} to {first_duty['Duty_End']}
                    - Duty 2: {second_duty['Date']} from {second_duty['Duty_Start']} to {second_duty['Duty_End']}
                    Check DGCA compliance for consecutive night duties.
                    """
                    result = self.validate_with_rag(scenario)
                    if result['status'] == 'violation':
                        violations.append({
                            'crew_id': crew_id,
                            'date': first_duty['Date'],
                            'type': 'consecutive_night_duty',
                            'message': result['message'],
                            'context': result.get('context', '')
                        })
        
        # 2. Check for extended duty periods (limit to top 3 longest duties)
        duty_durations = []
        for duty in crew_schedule:
            if duty.get('Duty_Start') and duty.get('Duty_End'):
                try:
                    duty_hours = (duty['Duty_End'] - duty['Duty_Start']).total_seconds() / 3600
                    if duty_hours > 11:  # Flag duties longer than 11 hours
                        duty_durations.append((duty_hours, duty))
                except (TypeError, AttributeError):
                    pass
        
        # Sort by duration and take top 3
        duty_durations.sort(key=lambda x: x[0], reverse=True)
        for duty_hours, duty in duty_durations[:3]:
            scenario = f"""
            Crew member {crew_id} has extended duty: {duty_hours:.1f} hours on {duty['Date']}
            from {duty['Duty_Start']} to {duty['Duty_End']}. Check DGCA duty time limits.
            """
            result = self.validate_with_rag(scenario)
            if result['status'] == 'violation':
                violations.append({
                    'crew_id': crew_id,
                    'date': duty['Date'],
                    'type': 'extended_duty',
                    'message': result['message'],
                    'context': result.get('context', '')
                })
        
        # 3. Check for short rest periods between duties
        for i in range(len(crew_schedule) - 1):
            current_duty = crew_schedule[i]
            next_duty = crew_schedule[i + 1]
            
            if (current_duty.get('Duty_End') and next_duty.get('Duty_Start') and
                current_duty['Date'] == next_duty['Date']):
                try:
                    rest_hours = (next_duty['Duty_Start'] - current_duty['Duty_End']).total_seconds() / 3600
                    if rest_hours < 11:  # Flag rest periods shorter than 11 hours
                        scenario = f"""
                        Crew member {crew_id} has short rest: {rest_hours:.1f} hours between duties
                        on {current_duty['Date']}. Check DGCA minimum rest requirements.
                        """
                        result = self.validate_with_rag(scenario)
                        if result['status'] == 'violation':
                            violations.append({
                                'crew_id': crew_id,
                                'date': current_duty['Date'],
                                'type': 'short_rest',
                                'message': result['message'],
                                'context': result.get('context', '')
                            })
                except (TypeError, AttributeError):
                    pass
        
        return violations
    
    def _is_night_duty(self, duty: Dict) -> bool:
        """Check if duty occurs during night hours (10 PM to 6 AM)"""
        try:
            if not duty.get('Duty_Start') or not duty.get('Duty_End'):
                return False
                
            start_hour = duty['Duty_Start'].hour
            end_hour = duty['Duty_End'].hour
            
            # Duty spans night hours if it starts before 6 AM or ends after 10 PM
            return start_hour < 6 or end_hour >= 22
        except:
            return False
    
    def clear_cache(self):
        """Clear the validation cache"""
        self._cache.clear()
        logger.info("RAG validation cache cleared")