
from functools import lru_cache
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.schemas.state import AgentState
from app.nodes.language_gate import language_gate_node
from app.nodes.info_extractor import info_extractor_node
from app.nodes.rag_node import rag_policy_node
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

    builder.add_node("language_gate", language_gate_node)
    builder.add_node("info_extractor", info_extractor_node)
    builder.add_node("rag_policy", rag_policy_node)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("db_writer", db_writer_node)

    builder.set_entry_point("language_gate")

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

    builder.add_edge("rag_policy", "info_extractor")

    builder.add_conditional_edges(
        "guardrail",
        lambda s: "db_writer" if not s.get("is_blocked") else "end",
        {
            "db_writer": "db_writer",
            "end": END,
        },
    )

    builder.add_edge("db_writer", END)

    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_after=["language_gate", "info_extractor", "rag_policy"],
    )

    logger.info("langgraph_compiled_successfully")
    return graph
