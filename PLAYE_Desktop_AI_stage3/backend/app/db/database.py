"""
Database configuration and helper functions for PLAYE PhotoLab backend.

This module defines the SQLAlchemy engine and session factory used by the
backend. It also provides a convenience function to create all tables
declared in ``app.db.models``. The database URL is read from the global
settings defined in ``app.config.Settings``.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base


# Create the SQLAlchemy engine. ``echo`` can be set to True for verbose
# logging of SQL statements during development.
engine = create_engine(settings.DATABASE_URL, echo=False, future=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def create_tables() -> None:
    """Create all database tables defined in the Base metadata.

    This function should be called once at application startup to ensure
    that the database schema exists. In production environments you might
    prefer to use migrations (e.g. Alembic) instead of calling this
    function directly.
    """
    Base.metadata.create_all(bind=engine)