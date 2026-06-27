from enum import Enum


class ValidationStatus(str, Enum):
    """
    Tracks where the conversation is in the booking lifecycle.
    The LangGraph router reads this to decide which node runs next.
    """
    NOT_STARTED = "not_started"
    COLLECTING = "collecting"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"       # user said info is wrong
    POLICY_QUERY = "policy_query"  # user asked a clinic policy question
    GUARDRAIL_BLOCKED = "guardrail_blocked"  # clinical/injection attempt
    COMPLETE = "complete"