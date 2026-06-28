
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User message",
    )
    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique conversation ID — use call SID or session ID",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    validation_status: str
    language: str
    token_id: Optional[str] = None
    is_complete: bool = False


class AppointmentRecord(BaseModel):
    token_id: str
    patient_name: str
    patient_phone: str
    symptoms: str
    time_window: str
    language: str
    status: str
    created_at: str


class AppointmentListResponse(BaseModel):
    total: int
    appointments: list[AppointmentRecord]
    stats: dict

