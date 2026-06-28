from pydantic import ValidationError
from langchain_core.messages import AIMessage
from app.schemas.state import AgentState
from app.schemas.patient import PatientInfo
from app.schemas.validation import ValidationStatus
from app.schemas.extraction import ExtractedPatientData
from app.nodes.extractor import extract_patient_data
from app.nodes.rag_node import rag_policy_node
from app.rag.retriever import (
    is_policy_question,
    is_booking_clarification,
    is_identity_or_role_question,
    is_general_inquiry,
    is_booking_status_question,
    is_off_topic_question,
)
from app.utils.availability import (
    get_availability_response,
    validate_time_window,
    is_availability_question,
)
from app.config.settings import get_settings
from app.config.llm import get_llm
from app.config.logger import get_logger
from app.guardrails.classifier import classify_message

logger = get_logger(__name__)

MAX_RETRIES = 8

CONFIRM_WORDS = {
    "yes", "yeah", "yep", "y", "correct", "right", "confirm", "confirmed",
    "ok", "okay", "haan", "ha", "ha ji", "ji", "theek hai", "theek", "sahi",
}
REJECT_WORDS = {
    "no", "nope", "n", "wrong", "incorrect", "change", "nahi", "galat",
    "not correct", "cancel",
}

GUARDRAIL_RESPONSE = (
    "I'm here to help with dental appointments only. "
    "Please keep our conversation respectful and avoid inappropriate requests."
)

OFF_TOPIC_RESPONSE = (
    "I can only help with BrightSmile Dental Clinic appointments, services, "
    "hours, and policies. I don't have information on that topic. "
    "How can I help with your dental visit?"
)


def info_extractor_node(state: AgentState) -> dict:
    """
    Core collection node.

    Turn 1: Agent has just greeted the caller — ask for name only.
    Turn 2+: Extract what the caller said, merge, ask for next field.
    """
    settings = get_settings()
    llm = get_llm()
    messages = state.get("messages", [])
    current_info = state.get("patient_info", PatientInfo())
    retry_count = state.get("retry_count", 0)
    language = state.get("language", "english")

    # Infinite loop protection
    if retry_count >= MAX_RETRIES:
        logger.warning("max_retries_reached", retry_count=retry_count)
        fallback = (
            "I'm having trouble collecting your information. "
            f"Please call us directly at {settings.clinic_phone}."
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "validation_status": ValidationStatus.COMPLETE,
        }

    # Count how many human messages exist
    human_messages = [
        m for m in messages
        if m.__class__.__name__ == "HumanMessage"
    ]

    # Answer side questions before forcing the booking flow
    last_human = human_messages[-1].content if human_messages else ""
    preserved_status = _preserve_validation_status(state, current_info)

    guardrail_result = classify_message(last_human) if last_human else None
    if guardrail_result and not guardrail_result.is_safe:
        logger.warning(
            "info_extractor_guardrail_blocked",
            threat_type=guardrail_result.threat_type,
        )
        return {
            "patient_info": current_info,
            "validation_status": preserved_status,
            "messages": [AIMessage(content=GUARDRAIL_RESPONSE)],
            "retry_count": retry_count + 1,
            "is_blocked": guardrail_result.threat_type in {
                "prompt_injection", "jailbreak", "system_extraction",
            },
        }

    if last_human and is_off_topic_question(last_human):
        logger.info("off_topic_response", query=last_human[:50])
        return {
            "patient_info": current_info,
            "validation_status": preserved_status,
            "messages": [AIMessage(content=OFF_TOPIC_RESPONSE)],
            "retry_count": retry_count,
        }

    if last_human and current_info.name and is_booking_status_question(last_human):
        logger.info("booking_status_response", query=last_human[:50])
        return {
            "patient_info": current_info,
            "validation_status": preserved_status,
            "messages": [AIMessage(content=_booking_details_reply(current_info, preserved_status))],
            "retry_count": retry_count,
        }

    if last_human and is_identity_or_role_question(last_human):
        logger.info("identity_inquiry_response", query=last_human[:50])
        reply = _identity_reply(settings) + _confirmation_suffix(preserved_status, current_info)
        return {
            "patient_info": current_info,
            "validation_status": preserved_status,
            "messages": [AIMessage(content=reply)],
            "retry_count": retry_count,
        }

    if last_human and is_booking_clarification(last_human, messages):
        logger.info("booking_clarification_response", query=last_human[:50])
        return {
            "patient_info": current_info,
            "validation_status": preserved_status,
            "messages": [AIMessage(content=_booking_clarification_reply(messages))],
            "retry_count": retry_count,
        }

    if last_human and is_availability_question(last_human):
        avail_reply = get_availability_response(current_info.time_window, last_human)
        if avail_reply:
            logger.info("availability_response", query=last_human[:50])
            cleared_info = current_info
            if current_info.time_window and validate_time_window(current_info.time_window):
                cleared_info = PatientInfo(
                    name=current_info.name,
                    phone=current_info.phone,
                    symptoms=current_info.symptoms,
                    time_window=None,
                )
            return {
                "patient_info": cleared_info,
                "validation_status": preserved_status,
                "messages": [AIMessage(content=avail_reply)],
                "retry_count": retry_count,
            }

    if last_human and is_policy_question(last_human):
        logger.info("info_extractor_delegating_to_rag", query=last_human[:50])
        return rag_policy_node(state)

    if last_human and is_general_inquiry(last_human):
        logger.info("general_inquiry_response", query=last_human[:50])
        return rag_policy_node(state)

    # First human message — agent just greeted, now ask for name
    if len(human_messages) == 1:
        logger.info("first_turn_asking_for_name")
        response = _ask_for_next_field(
            PatientInfo(), messages, language, settings, llm
        )
        return {
            "patient_info": PatientInfo(),
            "validation_status": ValidationStatus.COLLECTING,
            "messages": [AIMessage(content=response)],
            "retry_count": 0,
        }

    # Turn 2+ — extract from conversation
    try:
        extracted: ExtractedPatientData = extract_patient_data(messages)
    except Exception as e:
        logger.error("extraction_failed", error=str(e))
        extracted = ExtractedPatientData()

    # Check if user is confirming or rejecting readback
    if state.get("validation_status") == ValidationStatus.AWAITING_CONFIRMATION:
        if _is_simple_confirmation(last_human) or extracted.is_confirmation:
            logger.info("user_confirmed_details")
            return {
                "validation_status": ValidationStatus.CONFIRMED,
                "patient_info": current_info,
            }
        if _is_simple_rejection(last_human) or extracted.is_rejection:
            logger.info("user_rejected_details")
            return {
                "validation_status": ValidationStatus.COLLECTING,
                "patient_info": PatientInfo(),
                "retry_count": 0,
                "messages": [AIMessage(content=(
                    "No problem, let's start again. "
                    "Could you please tell me your full name?"
                ))],
            }
        # Still awaiting — don't regenerate a new LLM readback
        if current_info.is_complete():
            logger.info("awaiting_confirmation_reminder")
            return {
                "patient_info": current_info,
                "validation_status": ValidationStatus.AWAITING_CONFIRMATION,
                "messages": [AIMessage(content=_pending_confirmation_reminder(current_info))],
                "retry_count": retry_count,
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

    if merged.time_window:
        slot_rejection = validate_time_window(merged.time_window)
        if slot_rejection:
            logger.info("invalid_time_window_rejected", time_window=merged.time_window)
            merged = PatientInfo(
                name=merged.name,
                phone=merged.phone,
                symptoms=merged.symptoms,
                time_window=None,
            )
            return {
                "patient_info": merged,
                "validation_status": ValidationStatus.COLLECTING,
                "messages": [AIMessage(content=slot_rejection)],
                "retry_count": retry_count,
            }

    # All fields collected — generate readback for confirmation (once)
    if merged.is_complete():
        if state.get("validation_status") == ValidationStatus.AWAITING_CONFIRMATION:
            return {
                "patient_info": merged,
                "validation_status": ValidationStatus.AWAITING_CONFIRMATION,
                "messages": [AIMessage(content=_pending_confirmation_reminder(merged))],
                "retry_count": retry_count,
            }
        readback = _generate_readback(merged, language, settings, llm)
        return {
            "patient_info": merged,
            "validation_status": ValidationStatus.AWAITING_CONFIRMATION,
            "messages": [AIMessage(content=readback)],
            "retry_count": retry_count,
        }

    # Still missing fields — ask for next one
    response = _ask_for_next_field(
        merged, messages, language, settings, llm
    )

    return {
        "patient_info": merged,
        "validation_status": ValidationStatus.COLLECTING,
        "messages": [AIMessage(content=response)],
        "retry_count": retry_count,
    }


def _is_simple_confirmation(text: str) -> bool:
    normalized = text.lower().strip().rstrip(".!")
    return normalized in CONFIRM_WORDS


def _is_simple_rejection(text: str) -> bool:
    normalized = text.lower().strip().rstrip(".!")
    return normalized in REJECT_WORDS


def _pending_confirmation_reminder(info: PatientInfo) -> str:
    return (
        "Here are your appointment details:\n"
        f"Name: {info.name}\n"
        f"Phone: {info.phone}\n"
        f"Reason for visit: {info.symptoms}\n"
        f"Requested time: {info.time_window}\n\n"
        "Please reply yes to confirm or no to make changes."
    )


def _generate_readback(info, language, settings, llm) -> str:
    from langchain_core.messages import HumanMessage
    prompt = f"""
You are the receptionist for {settings.clinic_name}.
Generate a confirmation readback in {language} language.
Read back these details and ask if everything is correct:

Name: {info.name}
Phone: {info.phone}
Reason for visit: {info.symptoms}
Requested time: {info.time_window}

Keep it warm and clear. End with a yes/no confirmation question.
Respond ONLY with the readback message, nothing else.
""".strip()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def _ask_for_next_field(info, messages, language, settings, llm) -> str:
    from langchain_core.messages import HumanMessage
    missing = info.missing_fields()

    field_questions = {
        "name": "Could you please tell me your full name?",
        "phone": "What is your phone number?",
        "symptoms": "What is the reason for your visit or what symptoms are you experiencing?",
        "time_window": "What date and time would you prefer for your appointment?",
    }

    if not missing:
        return "I have all your details. Let me confirm them with you."

    next_field = missing[0]
    question = field_questions[next_field]

    # Translate if not English
    if str(language).lower() not in ["english", "unknown"]:
        prompt = f"""
Translate this question to {language} language naturally:
"{question}"
Respond ONLY with the translated question, nothing else.
""".strip()
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    return question


def _booking_clarification_reply(messages: list) -> str:
    """Explain why a booking detail is needed, then repeat the last question."""
    last_ai = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            last_ai = msg.content
            break

    last_ai_lower = last_ai.lower()
    if "name" in last_ai_lower:
        return (
            "We need your full name to register your appointment. "
            "Could you please tell me your full name?"
        )
    if "phone" in last_ai_lower:
        return (
            "We need your phone number to confirm your booking and contact you "
            "about your appointment. What is your phone number?"
        )
    if "symptom" in last_ai_lower or "reason for visit" in last_ai_lower:
        return (
            "Knowing your reason for visit helps us schedule the right type of "
            "appointment. What is the reason for your visit?"
        )
    if "time" in last_ai_lower or "date" in last_ai_lower:
        return (
            "We need your preferred date and time so we can check availability. "
            "What date and time would you prefer for your appointment?"
        )
    return (
        "I'm collecting a few details to complete your booking. "
        f"{last_ai}"
    )


def _preserve_validation_status(state: AgentState, info: PatientInfo) -> ValidationStatus:
    status = state.get("validation_status", ValidationStatus.COLLECTING)
    if status == ValidationStatus.AWAITING_CONFIRMATION:
        return ValidationStatus.AWAITING_CONFIRMATION
    if info.is_complete():
        return ValidationStatus.AWAITING_CONFIRMATION
    return ValidationStatus.COLLECTING


def _confirmation_suffix(status: ValidationStatus, info: PatientInfo) -> str:
    if status != ValidationStatus.AWAITING_CONFIRMATION or not info.is_complete():
        return ""
    return (
        f"\n\nYour pending appointment is for {info.time_window} "
        f"regarding {info.symptoms}. Please reply yes to confirm or no to change."
    )


def _booking_details_reply(info: PatientInfo, status: ValidationStatus) -> str:
    lines = []
    if info.name:
        lines.append(f"Name: {info.name}")
    if info.phone:
        lines.append(f"Phone: {info.phone}")
    if info.symptoms:
        lines.append(f"Reason for visit: {info.symptoms}")
    if info.time_window:
        lines.append(f"Requested appointment: {info.time_window}")

    details = "\n".join(lines)
    if status == ValidationStatus.AWAITING_CONFIRMATION:
        return (
            f"Here are your appointment details:\n{details}\n\n"
            "Please reply yes to confirm or no to make changes."
        )
    return (
        f"Here is what I have recorded so far:\n{details}\n\n"
        "Would you like to continue with your booking?"
    )


def _identity_reply(settings) -> str:
    return (
        f"I'm the AI receptionist for {settings.clinic_name}. "
        "I can help you book a dental appointment and answer questions "
        "about our clinic, services, hours, and policies. "
        "I use secure AI language technology — I don't perform clinical procedures. "
        "When you visit the clinic, our dental team will take care of your treatment. "
        f"If you need to speak with staff directly, please call {settings.clinic_phone}. "
        "How can I help you today?"
    )