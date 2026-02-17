"""API package for PLAYE PhotoLab backend."""

from fastapi import APIRouter

# Create a router to include endpoints
router = APIRouter()

# Import API routes to register them (defined in routes.py)
from . import routes  # noqa: F401