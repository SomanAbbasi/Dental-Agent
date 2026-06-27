

import pytest
from app.agents.router import should_continue
from app.schemas.validation import ValidationStatus
from app.schemas.patient import PatientInfo
from langchain_core.messages import HumanMessage, AIMessage


def make_state(
    status=ValidationStatus.COLLECTING,
    is_blocked=False,
    retry_count=0,
    last_human="hello I need an appointment",
):
    return {
        "messages": [HumanMessage(content=last_human)],
        "language": "english",
        "patient_info": PatientInfo(),
        "validation_status": status,
        "current_token": None,
        "retry_count": retry_count,
        "is_blocked": is_blocked,
        "rag_context": None,
    }


class TestRouter:

    def test_blocked_state_routes_to_end(self):
        state = make_state(is_blocked=True)
        assert should_continue(state) == "end"

    def test_guardrail_blocked_status_routes_to_end(self):
        state = make_state(status=ValidationStatus.GUARDRAIL_BLOCKED)
        assert should_continue(state) == "end"

    def test_complete_status_routes_to_end(self):
        state = make_state(status=ValidationStatus.COMPLETE)
        assert should_continue(state) == "end"

    def test_confirmed_routes_to_guardrail(self):
        state = make_state(status=ValidationStatus.CONFIRMED)
        assert should_continue(state) == "guardrail"

    def test_collecting_routes_to_info_extractor(self):
        state = make_state(status=ValidationStatus.COLLECTING)
        assert should_continue(state) == "info_extractor"

    def test_policy_question_routes_to_rag(self):
        state = make_state(
            status=ValidationStatus.COLLECTING,
            last_human="what is your cancellation policy",
        )
        assert should_continue(state) == "rag_policy"

    def test_max_retries_routes_to_end(self):
        state = make_state(retry_count=8)
        assert should_continue(state) == "end"

    def test_not_started_routes_to_info_extractor(self):
        state = make_state(status=ValidationStatus.NOT_STARTED)
        assert should_continue(state) == "info_extractor"