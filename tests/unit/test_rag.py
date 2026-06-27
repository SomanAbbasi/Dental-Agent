import pytest
from unittest.mock import patch, MagicMock
from app.rag.retriever import is_policy_question, retrieve_policy


class TestIsPolicyQuestion:
    """
    Tests the fast keyword check — no vector search, no LLM.
    """

    def test_detects_cancellation_question(self):
        assert is_policy_question("what is your cancellation policy") is True

    def test_detects_hours_question(self):
        assert is_policy_question("what time do you open") is True

    def test_detects_emergency_question(self):
        assert is_policy_question("what if i have an emergency") is True

    def test_detects_payment_question(self):
        assert is_policy_question("how do i pay") is True

    def test_detects_urdu_transliteration(self):
        assert is_policy_question("clinic band hai kya") is True

    def test_booking_message_is_not_policy(self):
        assert is_policy_question("mera naam Ali hai") is False

    def test_symptom_message_is_not_policy(self):
        assert is_policy_question("I have a toothache") is False

    def test_empty_string_is_not_policy(self):
        assert is_policy_question("") is False


class TestRetrievePolicy:
    """
    Tests the retrieval function with a mocked vectorstore
    so we never need the actual FAISS index during unit tests.
    """

    @patch("app.rag.retriever.get_vectorstore")
    def test_returns_context_above_threshold(self, mock_get_vs):
        mock_doc = MagicMock()
        mock_doc.page_content = "Appointments must be cancelled at least 2 hours before."
        mock_doc.metadata = {"source": "clinic_policies.txt"}

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (mock_doc, 0.85),
        ]
        mock_get_vs.return_value = mock_vs

        result = retrieve_policy("what is the cancellation policy")
        assert result is not None
        assert "cancelled" in result
        assert "0.85" in result

    @patch("app.rag.retriever.get_vectorstore")
    def test_returns_none_below_threshold(self, mock_get_vs):
        mock_doc = MagicMock()
        mock_doc.page_content = "Some weakly related content."
        mock_doc.metadata = {"source": "clinic_policies.txt"}

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (mock_doc, 0.10),  # below 0.35 threshold
        ]
        mock_get_vs.return_value = mock_vs

        result = retrieve_policy("completely unrelated question")
        assert result is None

    @patch("app.rag.retriever.get_vectorstore")
    def test_returns_none_for_empty_query(self, mock_get_vs):
        result = retrieve_policy("")
        assert result is None
        mock_get_vs.assert_not_called()

    @patch("app.rag.retriever.get_vectorstore")
    def test_returns_none_when_no_results(self, mock_get_vs):
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = []
        mock_get_vs.return_value = mock_vs

        result = retrieve_policy("some question")
        assert result is None