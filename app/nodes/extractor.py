from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import BaseMessage, HumanMessage
from app.config.llm import get_llm
from app.config.settings import get_settings
from app.config.logger import get_logger
from app.schemas.extraction import ExtractedPatientData

logger = get_logger(__name__)

EXTRACTION_PROMPT = """
From the conversation history below, extract any patient booking information.
Return null for any field not yet mentioned.
Do not invent or guess any values.
Detect the language from the messages.

For time_window:
- Preserve the user's intent exactly, including relative phrases like "tomorrow",
  "next Monday", "one week from now", "in 3 days", or "day after tomorrow".
- Include any mentioned clock time such as "10 AM" or "4:30 PM".
- Do not convert relative dates yourself — return the user's wording.

Conversation:
{history}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
def extract_patient_data(
    messages: list[BaseMessage],
) -> ExtractedPatientData:
    """
    Uses LangChain with_structured_output to force the LLM
    to return a valid ExtractedPatientData object.
    If it cannot, it raises — never returns garbage.
    """
    settings = get_settings()
    llm = get_llm()

    # function_calling works with llama-3.3 on Groq; json_mode raises BadRequestError
    structured_llm = llm.with_structured_output(
        ExtractedPatientData,
        method="function_calling",
    )

    history_text = "\n".join(
        f"{msg.__class__.__name__}: {msg.content}"
        for msg in messages[-10:]  # last 10 messages only, keep context window small
    )

    prompt = EXTRACTION_PROMPT.format(history=history_text)

    logger.debug("extracting_patient_data", history_length=len(messages))

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    logger.info(
        "extraction_complete",
        has_name=result.name is not None,
        has_phone=result.phone is not None,
        has_symptoms=result.symptoms is not None,
        has_time=result.time_window is not None,
        language=result.detected_language,
    )

    return result