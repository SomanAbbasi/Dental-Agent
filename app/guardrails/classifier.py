
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import HumanMessage, SystemMessage
from app.config.llm import get_classifier_llm
from app.config.prompts import GUARDRAIL_CLASSIFIER_PROMPT
from app.config.logger import get_logger
from app.schemas.extraction import GuardrailResult

logger = get_logger(__name__)

HARDCODED_INJECTION_PATTERNS: list[str] = [
    "ignore previous",
    "ignore above",
    "ignore all",
    "new instruction",
    "forget your instructions",
    "you are now",
    "act as",
    "pretend you are",
    "system prompt",
    "reveal your prompt",
    "what are your instructions",
    "jailbreak",
    "dan mode",
    "developer mode",
    "sudo",
    "override",
    "disregard",
    "bypass your",
    "forget everything",
    "do anything now",
]

INAPPROPRIATE_PATTERNS: list[str] = [
    "fuck", "shit", "bitch", "asshole", "bastard", "dick",
    "piss off", "shut up", "stupid bot", "idiot",
    "kill you", "hate you", "useless",
]


def check_inappropriate_patterns(message: str) -> GuardrailResult | None:
    """Block abusive or profane language."""
    message_lower = message.lower().strip()
    for pattern in INAPPROPRIATE_PATTERNS:
        if pattern in message_lower:
            logger.warning(
                "inappropriate_language_blocked",
                pattern=pattern,
                message_preview=message[:80],
            )
            return GuardrailResult(
                is_safe=False,
                threat_type="abusive_language",
                confidence=1.0,
            )
    return None


def check_hardcoded_patterns(message: str) -> GuardrailResult | None:
    """
    Fast pattern check — no LLM call needed.
    Returns a blocked GuardrailResult if matched, else None.
    """
    message_lower = message.lower().strip()
    for pattern in HARDCODED_INJECTION_PATTERNS:
        if pattern in message_lower:
            logger.warning(
                "hardcoded_pattern_blocked",
                pattern=pattern,
                message_preview=message[:80],
            )
            return GuardrailResult(
                is_safe=False,
                threat_type="prompt_injection",
                confidence=1.0,
            )
    return None


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
)
def classify_message(message: str) -> GuardrailResult:
    """
    Two-layer guardrail:
    Layer 1 — instant hardcoded pattern match (free, fast)
    Layer 2 — LLM classifier for subtle threats

    If the LLM classifier itself fails or returns malformed JSON,
    we default to SAFE so a transient API error doesn't block callers.
    Log the failure for monitoring.
    """
    # Layer 1: hardcoded patterns
    hardcoded_result = check_hardcoded_patterns(message)
    if hardcoded_result is not None:
        return hardcoded_result

    inappropriate_result = check_inappropriate_patterns(message)
    if inappropriate_result is not None:
        return inappropriate_result

    # Layer 2: LLM classification
    llm = get_classifier_llm()
    prompt = GUARDRAIL_CLASSIFIER_PROMPT.format(message=message)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if model returns them
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        parsed = json.loads(raw)
        result = GuardrailResult(**parsed)

        if not result.is_safe:
            logger.warning(
                "llm_classifier_blocked",
                threat_type=result.threat_type,
                confidence=result.confidence,
                message_preview=message[:80],
            )

        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.error(
            "classifier_parse_failed",
            error=str(e),
            raw_response=raw if "raw" in dir() else "no response",
        )
        # Fail open — don't block callers on classifier errors
        return GuardrailResult(
            is_safe=True,
            threat_type="none",
            confidence=0.0,
        )