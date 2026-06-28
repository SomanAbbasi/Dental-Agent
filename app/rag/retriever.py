from functools import lru_cache
from langchain_community.vectorstores import FAISS
from app.rag.builder import build_faiss_index
from app.config.logger import get_logger
from app.utils.availability import is_availability_question

logger = get_logger(__name__)


SIMILARITY_THRESHOLD = 0.35

# Maximum number of chunks to retrieve per query
TOP_K = 3


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
   
    return build_faiss_index()


def retrieve_policy(query: str) -> str | None:
    """
    Retrieves relevant policy chunks for a given query.

    Returns:
        A formatted string of relevant policy text if similarity
        is above threshold, else None.

    Returning None signals to the agent that no relevant policy
    was found — it should then tell the caller to contact the clinic
    directly rather than making something up.
    """
    if not query or len(query.strip()) < 3:
        logger.warning("retrieve_called_with_empty_query")
        return None

    vectorstore = get_vectorstore()

    logger.debug("retrieving_policy", query=query[:80])

    
    results = vectorstore.similarity_search_with_score(
        query,
        k=TOP_K,
    )

    if not results:
        logger.info("no_results_returned", query=query[:80])
        return None

    # Filter by similarity threshold
    relevant = [
        (doc, score)
        for doc, score in results
        if score >= SIMILARITY_THRESHOLD
    ]

    if not relevant:
        logger.info(
            "all_results_below_threshold",
            query=query[:80],
            best_score=results[0][1] if results else 0,
            threshold=SIMILARITY_THRESHOLD,
        )
        return None

    # Format the retrieved chunks into a single context string
    context_parts = []
    for i, (doc, score) in enumerate(relevant, 1):
        source = doc.metadata.get("source", "policy")
        context_parts.append(
            f"[Source {i} — relevance {score:.2f}]\n{doc.page_content.strip()}"
        )
        logger.debug(
            "chunk_retrieved",
            chunk_index=i,
            score=round(score, 3),
            source=source,
        )

    return "\n\n".join(context_parts)


IDENTITY_PHRASES = [
    "who are you", "what are you", "who is this", "what is this",
    "are you a bot", "are you real", "are you human", "are you ai",
    "are you a robot", "real person", "speak with", "talk to",
    "speak to", "human assistant", "dental assistant", "receptionist",
    "someone else", "live agent", "actual person", "introduce yourself",
    "what can you help", "what can you do", "how can you help",
    "who am i speaking", "who do i speak",
    "about yourself", "about you", "tell me about yourself",
    "which tools", "what tools", "what model", "what ai",
    "how do you work", "what technology",
]

BOOKING_STATUS_PHRASES = [
    "my appointment", "my meeting", "which date", "what date", "when is my",
    "what time is my", "appointment date", "meeting date", "scheduled for",
    "booked for", "my booking", "which day", "on which date", "what day",
    "tell me the date", "tell me date", "also tell me date", "the date i have",
    "i book", "i booked", "just book", "have meeting", "have appointment",
    "appointment details", "meeting details", "my schedule",
    "book a meeting", "booked a meeting", "book a appointment",
]

BOOKING_INTENT_PHRASES = [
    "book an", "book a ", "booking", "appointment", "schedule",
    "my name is", "call me ", "this is ",
]


def is_identity_or_role_question(query: str) -> bool:
    """Detect questions about the agent itself or requests to speak to a person."""
    q = query.lower().strip()
    if any(phrase in q for phrase in IDENTITY_PHRASES):
        return True

    identity_patterns = [
        "with whom", "to whom", "speaking with", "speaking to",
        "talking with", "talking to", "am i speaking", "am i talking",
        "who am i", "whom am i", "who is this", "who's this",
        "are you there", "anyone there", "human there",
    ]
    return any(pattern in q for pattern in identity_patterns)


def is_booking_status_question(query: str) -> bool:
    """Detect when the user asks about their own in-progress or pending booking."""
    q = query.lower().strip()
    return any(phrase in q for phrase in BOOKING_STATUS_PHRASES)


def is_policy_question(query: str) -> bool:
    """
    Detect clinic information questions (policies, services, hours, etc.).
    Runs before vector search to route to the RAG node.
    """
    q = query.lower().strip()
    if is_identity_or_role_question(q) or is_booking_status_question(q):
        return False

    policy_keywords = [
        # English — policies
        "cancel", "cancellation", "miss", "missed", "no show", "no-show",
        "late", "reschedule", "payment", "pay", "hours", "hour", "open",
        "close", "closed", "emergency", "parking", "children", "minor",
        "record", "refund", "feedback", "token", "lost token", "how long",
        # English — hours & schedule
        "timing", "timings", "schedule", "opening", "working hours",
        "clinic hour", "clinic hours", "when are you", "when do you",
        "what time", "open on", "close on", "sunday", "weekend",
        # English — services, staff & general clinic info
        "service", "services", "treatment", "treatments", "offer", "offers",
        "provide", "available", "what do you", "what can you", "do you do",
        "do you have", "tell me about", "tell me", "what kind of", "types of",
        "dental care", "procedure", "procedures", "checkup", "check-up",
        "cleaning", "whitening", "extraction", "filling", "root canal",
        "braces", "orthodont", "implant", "cosmetic", "specialist",
        "clinic info", "clinic detail", "information about",
        "doctor", "dentist", "experience", "qualification", "qualified",
        "doctor name", "dentist name", "name of doctor", "name of dentist",
        "who is the doctor", "who is the dentist", "which doctor",
        # Urdu / Punjabi transliterations
        "band", "khula", "waqt", "payment", "paisa", "paisay",
        "appointment cancel", "emergency", "service", "ilaj", "dant", "daant",
        "kitne baje", "khulna", "band hona",
    ]
    return any(kw in q for kw in policy_keywords)


OFF_TOPIC_KEYWORDS = [
    "latest news", "breaking news", "news today", "weather",
    "politics", "election", "football", "cricket score",
    "stock market", "bitcoin", "crypto", "recipe",
    "movie", "tell me a joke", "write a poem",
]


def is_off_topic_question(query: str) -> bool:
    """Questions unrelated to dental clinic booking or policies."""
    q = query.lower().strip()
    if is_identity_or_role_question(q) or is_policy_question(q):
        return False
    if is_booking_status_question(q) or is_availability_question(q):
        return False
    return any(kw in q for kw in OFF_TOPIC_KEYWORDS)


def is_booking_clarification(query: str, messages: list) -> bool:
    """Detect when the user questions why booking details are needed."""
    q = query.lower().strip()
    if q not in {
        "why", "why?", "what for", "what for?", "how come", "how come?",
        "reason", "reason?", "for what", "for what?",
    } and not q.startswith("why "):
        return False

    skipped_current = False
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            if not skipped_current and msg.content.lower().strip() == q:
                skipped_current = True
                continue
        if msg.__class__.__name__ == "AIMessage":
            last_ai = msg.content.lower()
            booking_prompts = [
                "your name", "full name", "phone number", "phone",
                "reason for visit", "symptoms", "appointment time",
                "date and time", "preferred",
            ]
            return any(p in last_ai for p in booking_prompts)
    return False


def is_general_inquiry(query: str) -> bool:
    """
    Detect non-booking questions the agent should answer before collecting data.
    Excludes messages that look like the user is providing booking details.
    """
    q = query.lower().strip()
    if not q:
        return False
    if any(phrase in q for phrase in BOOKING_INTENT_PHRASES):
        return False
    if is_booking_status_question(q):
        return True
    if is_identity_or_role_question(q):
        return True
    if is_policy_question(q):
        return True
    if is_booking_clarification(q, []):
        return True

    question_starters = (
        "who ", "what ", "where ", "when ", "why ", "how ", "which ",
        "can ", "could ", "would ", "is ", "are ", "do ", "does ",
        "tell me ", "i want to know", "i would like to know",
    )
    if "?" in q:
        return True
    return q.startswith(question_starters)