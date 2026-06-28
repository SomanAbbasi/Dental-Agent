

import pytest
import threading
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from app.database.slot_db import (
    load_db,
    save_db,
    _next_token_id,
)
from app.database.token_manager import SlotDatabase, TokenAssignmentError
from app.schemas.patient import PatientInfo
from app.schemas.appointment import AppointmentStatus

TEST_DB_PATH = Path("data/slots/test_appointments.csv")


@pytest.fixture(autouse=True)
def clean_test_db():
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    yield
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture
def db():
    return SlotDatabase()


@pytest.fixture
def complete_patient():
    return PatientInfo(
        name="Ali Khan",
        phone="03001234567",
        symptoms="Severe toothache",
        time_window="Tuesday morning",
    )


@pytest.fixture
def complete_patient_2():
    return PatientInfo(
        name="Sara Malik",
        phone="03211234567",
        symptoms="Broken tooth",
        time_window="Wednesday afternoon",
    )


class TestNextTokenId:

    def test_first_token_is_d100(self):
        df = pd.DataFrame(columns=["token_id"])
        assert _next_token_id(df) == "D-100"

    def test_increments_from_existing(self):
        df = pd.DataFrame({"token_id": ["D-100", "D-101", "D-102"]})
        assert _next_token_id(df) == "D-103"

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame()
        assert _next_token_id(df) == "D-100"

    def test_handles_gaps_in_sequence(self):
        df = pd.DataFrame({"token_id": ["D-100", "D-105"]})
        assert _next_token_id(df) == "D-106"


class TestSlotDatabase:

    @patch("app.database.slot_db.DB_PATH", TEST_DB_PATH)
    def test_assign_single_token(self, db, complete_patient):
        token = db.assign_token(complete_patient, "english")
        assert token.token_id == "D-100"
        assert token.patient_name == "Ali Khan"
        assert token.status == AppointmentStatus.CONFIRMED

    @patch("app.database.slot_db.DB_PATH", TEST_DB_PATH)
    def test_sequential_tokens(self, db, complete_patient, complete_patient_2):
        token1 = db.assign_token(complete_patient, "english")
        token2 = db.assign_token(complete_patient_2, "urdu")
        assert token1.token_id == "D-100"
        assert token2.token_id == "D-101"

    @patch("app.database.slot_db.DB_PATH", TEST_DB_PATH)
    def test_incomplete_patient_raises(self, db):
        incomplete = PatientInfo(name="Ali Khan")
        with pytest.raises(TokenAssignmentError) as exc:
            db.assign_token(incomplete, "english")
        assert "missing fields" in str(exc.value)

    @patch("app.database.slot_db.DB_PATH", TEST_DB_PATH)
    def test_concurrent_assignments_get_unique_tokens(self, db):
        patients = [
            PatientInfo(
                name=f"Patient {i}",
                phone=f"0300123{i:04d}",
                symptoms="Toothache",
                time_window="Monday morning",
            )
            for i in range(10)
        ]

        tokens = []
        errors = []

        def book(patient):
            try:
                token = db.assign_token(patient, "english")
                tokens.append(token.token_id)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=book, args=(p,)) for p in patients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent booking: {errors}"
        assert len(tokens) == 10
        assert len(set(tokens)) == 10, f"Duplicate tokens: {tokens}"

    @patch("app.database.slot_db.DB_PATH", TEST_DB_PATH)
    def test_cancel_token(self, db, complete_patient):
        token = db.assign_token(complete_patient, "english")
        result = db.cancel_token(token.token_id)
        assert result is True

    @patch("app.database.slot_db.DB_PATH", TEST_DB_PATH)
    def test_cancel_nonexistent_token_returns_false(self, db):
        result = db.cancel_token("D-999")
        assert result is False

    @patch("app.database.slot_db.DB_PATH", TEST_DB_PATH)
    def test_get_stats(self, db, complete_patient, complete_patient_2):
        db.assign_token(complete_patient, "english")
        db.assign_token(complete_patient_2, "urdu")
        stats = db.get_stats()
        assert stats["total"] == 2
        assert stats["confirmed"] == 2
        
        