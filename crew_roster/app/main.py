# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.core.config import settings
from app.data.database import db_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Crew Rostering API",
    description="API for optimizing crew schedules using genetic algorithms with RAG validation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        logger.info("Starting AI-Powered Crew Rostering API")
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    db_manager.close()
    logger.info("Application shutdown complete")

@app.get("/")
async def root():
    return {
        "message": "AI-Powered Crew Rostering API",
        "version": "1.0.0",
        "docs": "/docs",
        "health_check": "/api/v1/health"
    }

@app.get("/info")
async def info():
    return {
        "app_name": settings.app_name,
        "groq_model": settings.groq_model,
        "population_size": settings.population_size,
        "generations": settings.generations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")