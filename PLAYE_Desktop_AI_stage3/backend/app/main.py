"""PLAYE PhotoLab backend app."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app.api.auth_routes import router as auth_router
from app.api.enterprise_routes import router as enterprise_router
from app.api.response import error_response
from app.api.routes import router as ai_router
from app.db.database import create_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PLAYE PhotoLab", version="3.0.0", description="Cloud AI processing for forensic video analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "app://*", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router, prefix="/api")
app.include_router(auth_router)
app.include_router(enterprise_router, prefix="/api")


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return error_response(request, error=str(exc.detail), status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled backend exception")
    return error_response(request, error=str(exc), status_code=500)


@app.on_event("startup")
async def startup_event():
    create_tables()


@app.get("/")
async def root():
    return {"service": "PLAYE PhotoLab Backend", "version": "3.0.0", "status": "running"}


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
