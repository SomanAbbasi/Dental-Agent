from langchain_core.messages import AIMessage, SystemMessage
from app.schemas.state import AgentState
from app.schemas.language import Language
from app.schemas.validation import ValidationStatus
from app.config.settings import get_settings
from app.config.prompts import SYSTEM_PROMPT_MAIN
from app.config.llm import get_llm
from app.config.logger import get_logger

logger = get_logger(__name__)


def language_gate_node(state: AgentState) -> dict:

    settings = get_settings()
    llm = get_llm()

    current_language = state.get("language", Language.UNKNOWN)
    messages = state.get("messages", [])

    # Language already locked — just pass through
    if current_language != Language.UNKNOWN:
        logger.debug(
            "language_already_locked",
            language=current_language,
        )
        return {"language": current_language}

    # First turn — detect language from the first user message
    last_human_content = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            last_human_content = msg.content
            break

    if not last_human_content:
        # No human message yet — send greeting in all languages
        greeting = (
            "Welcome to BrightSmile Dental Clinic! / "
            "BrightSmile Dental Clinic mein khush aamdeed! / "
            "Assalam o Alaikum! Appointment book karni hai?"
        )
        return {
            "language": Language.UNKNOWN,
            "messages": [AIMessage(content=greeting)],
            "validation_status": ValidationStatus.NOT_STARTED,
            "retry_count": 0,
            "is_blocked": False,
            "rag_context": None,
        }

    # Ask LLM to detect language and generate appropriate greeting
    detection_prompt = f"""
Detect the language of this message: "{last_human_content}"
Options: english, urdu, punjabi, saraiki

Then respond with a warm greeting for {settings.clinic_name} in that detected language.
Ask how you can help them today.
Keep it to 2 sentences maximum.

Important: respond ONLY with the greeting message, nothing else.
Start your response with the detected language code in brackets like [urdu] or [english],
then a space, then the greeting.
Example: [urdu] آپ کا BrightSmile میں خیر مقدم ہے! آپ کی کیا مدد کر سکتا ہوں؟
""".strip()

    response = llm.invoke(detection_prompt)
    raw = response.content.strip()

    # Parse language from response
    detected = Language.UNKNOWN
    greeting_text = raw

    for lang in Language:
        if lang == Language.UNKNOWN:
            continue
        if raw.lower().startswith(f"[{lang.value}]"):
            detected = lang
            greeting_text = raw[len(f"[{lang.value}]") :].strip()
            break

    logger.info(
        "language_detected",
        detected=detected,
        message_preview=last_human_content[:50],
    )

    return {
        "language": detected,
        "messages": [AIMessage(content=greeting_text)],
        "validation_status": ValidationStatus.COLLECTING,
        "retry_count": 0,
        "is_blocked": False,
        "rag_context": None,
    }
