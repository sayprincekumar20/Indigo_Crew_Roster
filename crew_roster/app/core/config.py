# app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    app_name: str = "AI-Powered Crew Rostering API"
    data_path: str = "app/data/sample_data/"
    db_path: str = "crew_rostering.db"
    population_size: int = 20
    generations: int = 10
    crossover_prob: float = 0.6
    mutation_prob: float = 0.15
    
    # Groq API Configuration
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = "gemma2-9b-it"
    
    # RAG Configuration
    chroma_db_path: str = "chroma_db"
    embedding_model: str = "intfloat/e5-large-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()