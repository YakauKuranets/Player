"""
PLAYE PhotoLab - Cloud Backend
FastAPI server for heavy AI processing
"""

import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import logging
from typing import Optional
from starlette.responses import Response

from app.config import settings
from app.api import routes
from app.api.response import error_response
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


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Attach request_id to every request and expose it in response headers."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return HTTP errors with unified API schema."""
    return error_response(request, error=str(exc.detail), status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return fallback errors with unified API schema."""
    logger.exception("Unhandled backend exception")
    return error_response(request, error=str(exc), status_code=500)


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
