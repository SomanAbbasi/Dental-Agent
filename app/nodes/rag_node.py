
from langchain_core.messages import AIMessage
from app.schemas.state import AgentState
from app.schemas.validation import ValidationStatus
from app.rag.retriever import retrieve_policy
from app.config.prompts import SYSTEM_PROMPT_RAG
from app.config.settings import get_settings
from app.config.llm import get_llm
from app.config.logger import get_logger

logger = get_logger(__name__)


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
            "validation_status": ValidationStatus.COLLECTING,
        }

    logger.info("rag_node_retrieving", query=last_message[:80])

    context = retrieve_policy(last_message)

    if context is None:
        logger.info("rag_no_context_found", query=last_message[:80])
        no_info_response = (
            "I don't have specific information about that. "
            "Please call the clinic directly at "
            f"{settings.clinic_phone} for accurate details."
        )
        return {
            "messages": [AIMessage(content=no_info_response)],
            "rag_context": None,
            "validation_status": ValidationStatus.COLLECTING,
        }

    # Build a grounded answer using only the retrieved context
    prompt = SYSTEM_PROMPT_RAG.format(
        clinic_name=settings.clinic_name,
        clinic_phone=settings.clinic_phone,
        context=context,
    )

    response = llm.invoke(prompt + f"\n\nQuestion: {last_message}")

    logger.info("rag_node_answered", query=last_message[:80])

    return {
        "messages": [AIMessage(content=response.content)],
        "rag_context": context,
        "validation_status": ValidationStatus.COLLECTING,
    }