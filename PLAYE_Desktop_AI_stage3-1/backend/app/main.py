"""
PLAYE PhotoLab - Cloud Backend
FastAPI server for heavy AI processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import logging
from typing import Optional

from app.config import settings
from app.api import routes
from app.db.database import create_tables

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="PLAYE PhotoLab Backend",
    description="Cloud AI processing for forensic video analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    logger.info("Starting PLAYE PhotoLab Backend...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ CUDA not available. Running on CPU (slow)")
    
    # Store device in app state
    app.state.device = device
    
    # Pre-load models (опционально)
    # await load_all_models()

    # Create database tables if they do not exist. In a production system you
    # would use migrations (Alembic) instead of directly creating tables.
    try:
        create_tables()
        logger.info("✅ Database tables checked/created successfully")
    except Exception as exc:
        logger.error(f"❌ Failed to create database tables: {exc}")
    
    logger.info("✅ Backend started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down backend...")
    # Cleanup models, close connections, etc.


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PLAYE PhotoLab Backend",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": str(app.state.device)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Для разработки
    )