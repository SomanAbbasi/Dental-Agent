import re
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class AppointmentStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AppointmentToken(BaseModel):
    """
    Represents a finalized booking slot.
    token_id format is strictly D-XXX (e.g. D-104).
    """
    token_id: str = Field(
        ...,
        description="Alphanumeric slot token e.g. D-104",
    )
    patient_name: str = Field(..., min_length=2, max_length=100)
    patient_phone: str = Field(...)
    symptoms: str = Field(..., min_length=3, max_length=500)
    time_window: str = Field(...)
    language: str = Field(...)
    status: AppointmentStatus = Field(default=AppointmentStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("token_id")
    @classmethod
    def validate_token_format(cls, v: str) -> str:
        pattern = re.compile(r"^D-\d{3,6}$")
        if not pattern.match(v):
            raise ValueError(
                f"token_id must match format D-XXX, got: {v}"
            )
        return v