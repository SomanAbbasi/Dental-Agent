

from functools import lru_cache
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.schemas.state import AgentState
from app.nodes.language_gate import language_gate_node
from app.nodes.info_extractor import info_extractor_node
from app.nodes.rag_node import rag_policy_node
from app.nodes.validator import validator_node
from app.nodes.guardrail_node import guardrail_node
from app.nodes.db_writer import db_writer_node
from app.agents.router import should_continue
from app.config.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def build_graph():
    """

    Graph topology:
        START → language_gate → [router] → info_extractor (loop)
                                         → rag_policy → info_extractor
                                         → guardrail → db_writer
                                         → end
    """
    logger.info("building_langgraph")

    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("language_gate", language_gate_node)
    builder.add_node("info_extractor", info_extractor_node)
    builder.add_node("rag_policy", rag_policy_node)
    builder.add_node("validator", validator_node)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("db_writer", db_writer_node)

    # Entry point
    builder.set_entry_point("language_gate")

    # After language gate — always route via should_continue
    builder.add_conditional_edges(
        "language_gate",
        should_continue,
        {
            "info_extractor": "info_extractor",
            "rag_policy": "rag_policy",
            "guardrail": "guardrail",
            "end": END,
        },
    )

    # After info extractor — route again
    builder.add_conditional_edges(
        "info_extractor",
        should_continue,
        {
            "info_extractor": "info_extractor",
            "rag_policy": "rag_policy",
            "guardrail": "guardrail",
            "end": END,
        },
    )

    # After RAG — always go back to info extractor to continue booking
    builder.add_edge("rag_policy", "info_extractor")

    # After validator — route
    builder.add_conditional_edges(
        "validator",
        should_continue,
        {
            "info_extractor": "info_extractor",
            "guardrail": "guardrail",
            "end": END,
        },
    )

    # After guardrail — if safe go to db_writer, else end
    builder.add_conditional_edges(
        "guardrail",
        lambda s: "db_writer" if not s.get("is_blocked") else "end",
        {
            "db_writer": "db_writer",
            "end": END,
        },
    )

    # db_writer always ends the conversation
    builder.add_edge("db_writer", END)

    # Compile with in-memory checkpointer for multi-turn state
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("langgraph_compiled_successfully")
    return graph