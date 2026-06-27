from pydantic import ValidationError
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from app.schemas.state import AgentState
from app.schemas.patient import PatientInfo
from app.schemas.validation import ValidationStatus
from app.schemas.extraction import ExtractedPatientData
from app.nodes.extractor import extract_patient_data
from app.config.settings import get_settings
from app.config.prompts import SYSTEM_PROMPT_MAIN
from app.config.llm import get_llm
from app.config.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 5


def info_extractor_node(state: AgentState) -> dict:
    """
 
    Each turn it:
    1. Extracts any new data from the conversation
    2. Merges it into existing PatientInfo
    3. Asks for the next missing field naturally
    4. If all fields collected, moves to AWAITING_CONFIRMATION
    """
    settings = get_settings()
    llm = get_llm()
    messages = state.get("messages", [])
    current_info = state.get("patient_info", PatientInfo())
    retry_count = state.get("retry_count", 0)
    language = state.get("language")

    # Infinite loop protection
    if retry_count >= MAX_RETRIES:
        logger.warning(
            "max_retries_reached",
            retry_count=retry_count,
        )
        fallback = (
            "I'm having trouble collecting your information. "
            "Please call us directly at "
            f"{settings.clinic_phone} to book your appointment."
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "validation_status": ValidationStatus.COMPLETE,
            "retry_count": retry_count,
        }

    # Extract structured data from conversation history
    try:
        extracted: ExtractedPatientData = extract_patient_data(messages)
    except Exception as e:
        logger.error("extraction_failed", error=str(e))
        extracted = ExtractedPatientData()

    # Check if user is confirming or rejecting readback
    if state.get("validation_status") == ValidationStatus.AWAITING_CONFIRMATION:
        if extracted.is_confirmation:
            logger.info("user_confirmed_details")
            return {
                "validation_status": ValidationStatus.CONFIRMED,
                "patient_info": current_info,
            }
        if extracted.is_rejection:
            logger.info("user_rejected_details")
            return {
                "validation_status": ValidationStatus.COLLECTING,
                "patient_info": PatientInfo(),  # reset and recollect
                "retry_count": retry_count + 1,
                "messages": [AIMessage(content=(
                    "No problem, let's start again. "
                    "Could you please tell me your full name?"
                ))],
            }

    # Merge newly extracted data into existing PatientInfo
    try:
        merged = PatientInfo(
            name=extracted.name or current_info.name,
            phone=extracted.phone or current_info.phone,
            symptoms=extracted.symptoms or current_info.symptoms,
            time_window=extracted.time_window or current_info.time_window,
        )
    except ValidationError as e:
        logger.warning("merge_validation_error", error=str(e))
        merged = current_info

    logger.info(
        "info_merge_complete",
        missing=merged.missing_fields(),
        complete=merged.is_complete(),
    )

    # All fields collected — generate readback for confirmation
    if merged.is_complete():
        readback = _generate_readback(merged, language, settings, llm)
        return {
            "patient_info": merged,
            "validation_status": ValidationStatus.AWAITING_CONFIRMATION,
            "messages": [AIMessage(content=readback)],
            "retry_count": retry_count,
        }

    # Still missing fields — ask for next one naturally
    response = _ask_for_next_field(
        merged, messages, language, settings, llm
    )

    return {
        "patient_info": merged,
        "validation_status": ValidationStatus.COLLECTING,
        "messages": [AIMessage(content=response)],
        "retry_count": retry_count + 1,
    }


def _generate_readback(
    info: PatientInfo,
    language,
    settings,
    llm,
) -> str:
    """Generates a confirmation readback in the caller's language."""
    prompt = f"""
You are the receptionist for {settings.clinic_name}.
Generate a confirmation readback message in {language} language.
Read back these details and ask if everything is correct:

Name: {info.name}
Phone: {info.phone}
Reason for visit: {info.symptoms}
Requested time: {info.time_window}

Keep it warm and clear. End with a yes/no question asking for confirmation.
Respond ONLY with the readback message.
""".strip()

    response = llm.invoke(prompt)
    return response.content.strip()


def _ask_for_next_field(
    info: PatientInfo,
    messages: list,
    language,
    settings,
    llm,
) -> str:
    """Asks for the next missing field naturally in the caller's language."""
    missing = info.missing_fields()
    field_map = {
        "name": "their full name",
        "phone": "their phone number",
        "symptoms": "the reason for their visit or their symptoms",
        "time_window": "their preferred appointment date and time",
    }
    next_field = field_map.get(missing[0], missing[0])

    history = "\n".join(
        f"{m.__class__.__name__}: {m.content}"
        for m in messages[-6:]
    )

    prompt = f"""
You are the receptionist for {settings.clinic_name}.
You are speaking in {language} language.
Continue this conversation naturally and ask for {next_field}.
Do not ask for more than one thing at a time.
Do not repeat information already collected.
Keep it brief — one or two sentences maximum.

Conversation so far:
{history}

Respond ONLY with your next message to the patient.
""".strip()

    response = llm.invoke(prompt)
    return response.content.strip()