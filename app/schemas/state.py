from typing import Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from app.schemas.language import Language
from app.schemas.patient import PatientInfo
from app.schemas.appointment import AppointmentToken
from app.schemas.validation import ValidationStatus


class AgentState(TypedDict):
  

    # Full conversation history (auto-appended via reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Detected and locked language — set once by Language Gate, never changed
    language: Language

    # Patient info collected incrementally across turns
    patient_info: PatientInfo

    # Where we are in the booking lifecycle
    validation_status: ValidationStatus

    # Set only after Database Writer succeeds
    current_token: Optional[AppointmentToken]

    # How many times we've looped back asking for missing info
    # Prevents infinite loops — cap at settings.max_retry_attempts
    retry_count: int

    # Flag set by guardrail node — blocks DB write if True
    is_blocked: bool

    # Last retrieved RAG context (policy answer)
    rag_context: Optional[str]