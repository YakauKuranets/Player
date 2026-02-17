"""
SQLAlchemy models for PLAYE PhotoLab backend.

This module defines the database schema for the cloud backend. The two
primary entities are ``Case`` and ``AIJob``. A ``Case`` represents a
collection of related media and analysis tasks, while an ``AIJob``
represents a single AI inference task executed on an image. Additional
models can be added here as needed (e.g. ``User``, ``ModelMetadata``).
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class Case(Base):
    """Database model representing a forensic case.

    A case groups together related images and the AI jobs applied to them.
    """

    __tablename__ = 'cases'

    id: int = Column(Integer, primary_key=True)
    name: str = Column(String, nullable=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to AIJob. A case may have many jobs.
    jobs = relationship('AIJob', back_populates='case', cascade='all, delete-orphan')


class AIJob(Base):
    """Database model representing an AI processing job.

    Each job corresponds to a single invocation of an AI model (e.g. face
    enhancement, upscaling, denoising, detection). Jobs may optionally be
    associated with a case. They track input and output file paths as well
    as the current status of the job.
    """

    __tablename__ = 'ai_jobs'

    id: int = Column(Integer, primary_key=True)
    case_id: int = Column(Integer, ForeignKey('cases.id'), nullable=True)
    task: str = Column(String, nullable=False)
    status: str = Column(String, default='pending')
    input_path: str = Column(String, nullable=True)
    output_path: str = Column(String, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to the case
    case = relationship('Case', back_populates='jobs')