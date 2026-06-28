
import asyncio
import uuid
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage
from app.api.schemas import ChatRequest, ChatResponse
from app.agents.graph import build_graph
from app.schemas.patient import PatientInfo
from app.schemas.language import Language
from app.schemas.validation import ValidationStatus
from app.config.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _run_graph(message: str, thread_id: str) -> dict:
 
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state to check if conversation already started
    try:
        current_state = graph.get_state(config)
        existing_values = current_state.values if current_state else {}
        pending_nodes = current_state.next if current_state else ()
    except Exception:
        existing_values = {}
        pending_nodes = ()

    # If graph paused at an interrupt, merge the new message and resume
    if pending_nodes:
        graph.update_state(
            config,
            {"messages": [HumanMessage(content=message)]},
        )
        final_state = graph.invoke(None, config=config)
        return final_state

    # Build input — only pass the new human message
    # LangGraph merges it with existing message history via add_messages reducer
    graph_input = {
        "messages": [HumanMessage(content=message)],
    }

    # If first turn, initialize all state fields
    if not existing_values:
        graph_input.update({
            "language": Language.UNKNOWN,
            "patient_info": PatientInfo(),
            "validation_status": ValidationStatus.NOT_STARTED,
            "current_token": None,
            "retry_count": 0,
            "is_blocked": False,
            "rag_context": None,
        })

    final_state = graph.invoke(graph_input, config=config)
    return final_state


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Single-turn chat endpoint.
    Send a message, get a reply.
    The thread_id links messages into a conversation.
    """
    logger.info(
        "chat_request",
        thread_id=request.thread_id,
        message_preview=request.message[:50],
    )

    try:
        final_state = await asyncio.get_event_loop().run_in_executor(
            None,
            _run_graph,
            request.message,
            request.thread_id,
        )
    except Exception as e:
        logger.error("graph_execution_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}",
        )

    # Extract the last AI message as the reply
    messages = final_state.get("messages", [])
    ai_messages = [
        m for m in messages
        if m.__class__.__name__ == "AIMessage"
    ]
    reply = ai_messages[-1].content if ai_messages else "I'm sorry, please try again."

    validation_status = final_state.get(
        "validation_status", ValidationStatus.NOT_STARTED
    )
    language = final_state.get("language", Language.UNKNOWN)
    current_token = final_state.get("current_token")
    patient_info = final_state.get("patient_info")

    return ChatResponse(
        reply=reply,
        thread_id=request.thread_id,
        validation_status=str(validation_status),
        language=str(language),
        token_id=current_token.token_id if current_token else None,
        is_complete=str(validation_status) == ValidationStatus.COMPLETE,
    )


@router.websocket("/ws/{thread_id}")
async def websocket_chat(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for streaming multi-turn conversations.
    Each message sent over the socket gets a reply over the same socket.
    thread_id in the URL keeps the conversation state linked.
    """
    await websocket.accept()
    logger.info("websocket_connected", thread_id=thread_id)

    try:
        while True:
            message = await websocket.receive_text()

            if not message.strip():
                continue

            logger.debug(
                "websocket_message",
                thread_id=thread_id,
                preview=message[:50],
            )

            try:
                final_state = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _run_graph,
                    message,
                    thread_id,
                )

                messages = final_state.get("messages", [])
                ai_messages = [
                    m for m in messages
                    if m.__class__.__name__ == "AIMessage"
                ]
                reply = (
                    ai_messages[-1].content
                    if ai_messages
                    else "I'm sorry, please try again."
                )

                validation_status = final_state.get(
                    "validation_status", ValidationStatus.NOT_STARTED
                )
                current_token = final_state.get("current_token")

                await websocket.send_json({
                    "reply": reply,
                    "validation_status": str(validation_status),
                    "language": str(final_state.get("language", "unknown")),
                    "token_id": current_token.token_id if current_token else None,
                    "is_complete": str(validation_status) == ValidationStatus.COMPLETE,
                })

                # Close gracefully when booking is complete
                if str(validation_status) == ValidationStatus.COMPLETE:
                    logger.info(
                        "websocket_booking_complete",
                        thread_id=thread_id,
                    )
                    await websocket.close()
                    break

            except Exception as e:
                logger.error(
                    "websocket_graph_error",
                    error=str(e),
                    thread_id=thread_id,
                )
                await websocket.send_json({
                    "reply": "I'm sorry, something went wrong. Please try again.",
                    "validation_status": "error",
                    "language": "unknown",
                    "token_id": None,
                    "is_complete": False,
                })

    except WebSocketDisconnect:
        logger.info("websocket_disconnected", thread_id=thread_id)
