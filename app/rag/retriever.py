from functools import lru_cache
from langchain_community.vectorstores import FAISS
from app.rag.builder import build_faiss_index
from app.config.logger import get_logger

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


def is_policy_question(query: str) -> bool:
    """
    Fast keyword check to detect if a message is asking about clinic policy.
    This runs BEFORE the vector search to avoid unnecessary embedding calls.
    """
    policy_keywords = [
        # English
        "cancel", "cancellation", "miss", "missed", "no show", "no-show",
        "late", "reschedule", "payment", "pay", "hours", "open", "close",
        "closed", "emergency", "parking", "children", "minor", "record",
        "refund", "feedback", "token", "lost token", "how long",
        # Urdu / Punjabi transliterations
        "band", "khula", "waqt", "payment", "paisa", "paisay",
        "cancel", "appointment cancel", "time", "emergency",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in policy_keywords)