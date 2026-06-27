

from langchain_core.messages import AIMessage
from app.schemas.state import AgentState
from app.schemas.validation import ValidationStatus
from app.config.logger import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


def db_writer_node(state: AgentState) -> dict:
 
    settings = get_settings()
    patient_info = state.get("patient_info")
    language = state.get("language", "english")

    if patient_info is None or not patient_info.is_complete():
        logger.error("db_writer_called_with_incomplete_info")
        return {
            "validation_status": ValidationStatus.REJECTED,
        }

    # STUB — Phase 6 replaces this with real slot assignment
    stub_token = "D-001"

    logger.info(
        "db_writer_stub_booking",
        token=stub_token,
        patient_name=patient_info.name,
        phone=patient_info.phone,
        time_window=patient_info.time_window,
    )

    if "urdu" in str(language).lower():
        confirmation = (
            f"آپ کی appointment book ہو گئی ہے۔ "
            f"آپ کا token number ہے: {stub_token}۔ "
            f"آپ کو جلد ہی SMS موصول ہوگا۔ شکریہ!"
        )
    elif "punjabi" in str(language).lower():
        confirmation = (
            f"Appointment book ho gayi hai! "
            f"Tera token number hai: {stub_token}۔ "
            f"SMS aa javega. Shukriya!"
        )
    else:
        confirmation = (
            f"Your appointment has been successfully booked! "
            f"Your token number is {stub_token}. "
            f"You will receive an SMS confirmation shortly. Thank you!"
        )

    return {
        "validation_status": ValidationStatus.COMPLETE,
        "messages": [AIMessage(content=confirmation)],
        "current_token": None,  # Phase 6 sets this to real AppointmentToken
    }