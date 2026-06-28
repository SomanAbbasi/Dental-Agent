
from langchain_core.messages import AIMessage
from app.schemas.state import AgentState
from app.schemas.patient import PatientInfo
from app.schemas.validation import ValidationStatus
from app.rag.retriever import retrieve_policy
from app.config.prompts import SYSTEM_PROMPT_RAG
from app.config.settings import get_settings
from app.config.llm import get_llm
from app.config.logger import get_logger

logger = get_logger(__name__)


def _preserve_validation_status(state: AgentState) -> ValidationStatus:
    status = state.get("validation_status", ValidationStatus.COLLECTING)
    info = state.get("patient_info", PatientInfo())
    if status == ValidationStatus.AWAITING_CONFIRMATION:
        return ValidationStatus.AWAITING_CONFIRMATION
    if info.is_complete():
        return ValidationStatus.AWAITING_CONFIRMATION
    return ValidationStatus.COLLECTING


def _booking_reminder_suffix(state: AgentState) -> str:
    info = state.get("patient_info", PatientInfo())
    if _preserve_validation_status(state) != ValidationStatus.AWAITING_CONFIRMATION:
        return ""
    if not info.is_complete():
        return ""
    return (
        f"\n\nYour pending appointment is for {info.time_window} "
        f"regarding {info.symptoms}. Please reply yes to confirm or no to change."
    )


def rag_policy_node(state: AgentState) -> dict:
    """
    RAG Policy Searcher node.

    Called when the router detects a policy question.
    Retrieves relevant clinic policy chunks and uses the LLM
    to generate a grounded answer — never free-form.

    If no relevant context is found, returns a canned response
    directing the caller to contact the clinic directly.
    """
    settings = get_settings()
    llm = get_llm()

    # Get the last user message as the query
    last_message = ""
    for msg in reversed(state["messages"]):
        if msg.__class__.__name__ == "HumanMessage":
            last_message = msg.content
            break

    if not last_message:
        logger.warning("rag_node_called_with_no_human_message")
        return {
            "rag_context": None,
            "validation_status": _preserve_validation_status(state),
        }

    logger.info("rag_node_retrieving", query=last_message[:80])
    preserved_status = _preserve_validation_status(state)
    reminder = _booking_reminder_suffix(state)

    context = retrieve_policy(last_message)

    if context is None:
        logger.info("rag_no_context_found", query=last_message[:80])
        no_info_response = (
            "I don't have specific information about that. "
            "Please call the clinic directly at "
            f"{settings.clinic_phone} for accurate details."
            f"{reminder}"
        )
        return {
            "messages": [AIMessage(content=no_info_response)],
            "rag_context": None,
            "validation_status": preserved_status,
        }

    # Build a grounded answer using only the retrieved context
    prompt = SYSTEM_PROMPT_RAG.format(
        clinic_name=settings.clinic_name,
        clinic_phone=settings.clinic_phone,
        context=context,
    )

    response = llm.invoke(prompt + f"\n\nQuestion: {last_message}")
    answer = response.content.strip() + reminder

    logger.info("rag_node_answered", query=last_message[:80])

    return {
        "messages": [AIMessage(content=answer)],
        "rag_context": context,
        "validation_status": preserved_status,
    }