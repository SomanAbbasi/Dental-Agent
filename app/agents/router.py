

from app.schemas.state import AgentState
from app.schemas.validation import ValidationStatus
from app.rag.retriever import is_policy_question
from app.config.logger import get_logger

logger = get_logger(__name__)

ABSOLUTE_MAX_RETRIES = 8


def _already_answered_last_human(messages: list) -> bool:
    """True if the latest AI message comes after the latest human message."""
    last_human_idx = -1
    last_ai_idx = -1
    for i, msg in enumerate(messages):
        if msg.__class__.__name__ == "HumanMessage":
            last_human_idx = i
        elif msg.__class__.__name__ == "AIMessage":
            last_ai_idx = i
    return last_human_idx >= 0 and last_ai_idx > last_human_idx


def should_continue(state: AgentState) -> str:
    """
   
    Routing logic:
    1. If blocked by guardrail → end
    2. If complete → end
    3. If confirmed and not blocked → db_writer
    4. If policy question detected → rag_policy
    5. If collecting or awaiting confirmation → info_extractor
    6. Fallback → info_extractor
    """
    validation_status = state.get("validation_status", ValidationStatus.NOT_STARTED)
    is_blocked = state.get("is_blocked", False)
    retry_count = state.get("retry_count", 0)
    messages = state.get("messages", [])

    logger.debug(
        "router_inspecting_state",
        status=validation_status,
        is_blocked=is_blocked,
        retry_count=retry_count,
    )

    # Safety: absolute loop cap
    if retry_count >= ABSOLUTE_MAX_RETRIES:
        logger.error(
            "absolute_max_retries_hit",
            retry_count=retry_count,
        )
        return "end"

    # Guardrail blocked — stop everything
    if is_blocked or validation_status == ValidationStatus.GUARDRAIL_BLOCKED:
        logger.info("router_ending_guardrail_blocked")
        return "end"

    # Booking complete
    if validation_status == ValidationStatus.COMPLETE:
        logger.info("router_ending_complete")
        return "end"

    # Confirmed — run final guardrail check then write to DB
    if validation_status == ValidationStatus.CONFIRMED:
        logger.info("router_to_guardrail")
        return "guardrail"

    # Check if last human message is a policy question
    last_human = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            last_human = msg.content
            break

    if last_human and is_policy_question(last_human) and not _already_answered_last_human(messages):
        logger.info("router_to_rag_policy", query=last_human[:50])
        return "rag_policy"

    # Default — keep collecting information
    logger.info(
        "router_to_info_extractor",
        status=validation_status,
    )
    return "info_extractor"