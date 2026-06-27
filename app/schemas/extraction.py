from typing import Optional
from pydantic import BaseModel, Field


class ExtractedPatientData(BaseModel):
    """
     LLM must return from with_structured_output.
    If the LLM cannot extract a field, it returns null — never guesses.
    """
    name: Optional[str] = Field(
        default=None,
        description="Patient full name if mentioned, else null",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Phone number if mentioned, else null",
    )
    symptoms: Optional[str] = Field(
        default=None,
        description="Reason for visit or symptoms if mentioned, else null",
    )
    time_window: Optional[str] = Field(
        default=None,
        description="Requested appointment time if mentioned, else null",
    )
    detected_language: str = Field(
        default="unknown",
        description="Language detected: english, urdu, punjabi, saraiki, or unknown",
    )
    is_policy_question: bool = Field(
        default=False,
        description="True if the user is asking about clinic policies not booking",
    )
    is_confirmation: bool = Field(
        default=False,
        description="True if the user is confirming the readback (yes/haan/ha ji etc.)",
    )
    is_rejection: bool = Field(
        default=False,
        description="True if the user is rejecting the readback (no/nahi/galat etc.)",
    )


class GuardrailResult(BaseModel):
    """
    Result from the safety classifier.
    """
    is_safe: bool
    threat_type: str = Field(default="none")
    confidence: float = Field(ge=0.0, le=1.0)