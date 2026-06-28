from langchain_core.messages import AIMessage
from app.schemas.state import AgentState
from app.schemas.validation import ValidationStatus
from app.schemas.appointment import AppointmentToken
from app.database.token_manager import slot_db, TokenAssignmentError
from app.config.logger import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


def db_writer_node(state: AgentState) -> dict:
   
    settings = get_settings()
    patient_info = state.get("patient_info")
    language = state.get("language", "english")

    # Final completeness check — belt and suspenders
    if patient_info is None or not patient_info.is_complete():
        logger.error(
            "db_writer_called_with_incomplete_info",
            missing=patient_info.missing_fields() if patient_info else "all",
        )
        return {
            "validation_status": ValidationStatus.REJECTED,
            "messages": [AIMessage(content=(
                "I'm sorry, some of your information is missing. "
                f"Please call us directly at {settings.clinic_phone}."
            ))],
        }

    try:
        token: AppointmentToken = slot_db.assign_token(
            patient_info=patient_info,
            language=str(language),
        )

        logger.info(
            "appointment_booked",
            token_id=token.token_id,
            patient=patient_info.name,
        )

        confirmation = _build_confirmation(token, language, settings)

        return {
            "validation_status": ValidationStatus.COMPLETE,
            "current_token": token,
            "messages": [AIMessage(content=confirmation)],
        }

    except TokenAssignmentError as e:
        logger.error("db_writer_token_assignment_failed", error=str(e))
        return {
            "validation_status": ValidationStatus.REJECTED,
            "messages": [AIMessage(content=(
                "I'm sorry, there was a problem booking your appointment. "
                f"Please call us directly at {settings.clinic_phone}."
            ))],
        }


def _build_confirmation(
    token: AppointmentToken,
    language,
    settings,
) -> str:
    """Builds a confirmation message in the caller's language."""
    lang_str = str(language).lower()

    if "urdu" in lang_str:
        return (
            f"آپ کی appointment کامیابی سے book ہو گئی ہے!\n"
            f"آپ کا Token Number: {token.token_id}\n"
            f"نام: {token.patient_name}\n"
            f"وقت: {token.time_window}\n"
            f"آپ کو جلد ہی SMS موصول ہوگا۔ "
            f"{settings.clinic_name} میں آپ کا شکریہ!"
        )
    elif "punjabi" in lang_str:
        return (
            f"Appointment book ho gayi hai!\n"
            f"Tera Token Number: {token.token_id}\n"
            f"Naam: {token.patient_name}\n"
            f"Waqt: {token.time_window}\n"
            f"SMS aa javega. {settings.clinic_name} da shukriya!"
        )
    elif "saraiki" in lang_str:
        return (
            f"Appointment book tھe gayi!\n"
            f"Tera Token: {token.token_id}\n"
            f"Naam: {token.patient_name}\n"
            f"Waqt: {token.time_window}\n"
            f"Mehrbani {settings.clinic_name}!"
        )
    else:
        return (
            f"Your appointment has been successfully booked!\n"
            f"Token Number: {token.token_id}\n"
            f"Name: {token.patient_name}\n"
            f"Time: {token.time_window}\n"
            f"You will receive an SMS confirmation shortly.\n"
            f"Thank you for choosing {settings.clinic_name}!"
        )