

from langchain_core.messages import AIMessage
from app.schemas.state import AgentState
from app.schemas.validation import ValidationStatus
from app.guardrails.classifier import classify_message
from app.config.logger import get_logger

logger = get_logger(__name__)

BLOCKED_RESPONSE_ENGLISH = (
    "I'm an appointment assistant only and cannot provide medical advice. "
    "Please discuss any medical concerns directly with the doctor during your visit."
)

BLOCKED_RESPONSE_URDU = (
    "میں صرف appointment book کرنے میں مدد کر سکتا ہوں۔ "
    "طبی مشورے کے لیے براہ کرم ڈاکٹر سے ملاقات کے دوران بات کریں۔"
)

BLOCKED_RESPONSE_PUNJABI = (
    "میں صرف appointment book کرن وچ مدد کر سکدا ہاں۔ "
    "دوائی یا علاج بارے ڈاکٹر نال گل کرو۔"
)


def guardrail_node(state: AgentState) -> dict:
    """
    Final safety gate before any data is written.
    two-layer classifier on the last user message.

    If blocked:
    - Sets is_blocked = True
    - Sets status to GUARDRAIL_BLOCKED
    - Returns a canned response in the caller's language
    - Graph will not proceed to database write

    If safe:
    - Sets is_blocked = False
    - Lets the graph continue to database write
    """
    messages = state.get("messages", [])
    language = state.get("language", "english")

    # Get last human message
    last_human = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            last_human = msg.content
            break

    if not last_human:
        return {"is_blocked": False}

    result = classify_message(last_human)

    if not result.is_safe:
        logger.warning(
            "guardrail_node_blocked",
            threat_type=result.threat_type,
            confidence=result.confidence,
            language=language,
        )

        # Pick response in caller's language
        if "urdu" in str(language).lower():
            response = BLOCKED_RESPONSE_URDU
        elif "punjabi" in str(language).lower():
            response = BLOCKED_RESPONSE_PUNJABI
        else:
            response = BLOCKED_RESPONSE_ENGLISH

        return {
            "is_blocked": True,
            "validation_status": ValidationStatus.GUARDRAIL_BLOCKED,
            "messages": [AIMessage(content=response)],
        }

    return {"is_blocked": False}


