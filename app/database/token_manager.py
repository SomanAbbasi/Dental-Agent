

import threading
from contextlib import contextmanager
from datetime import datetime, UTC
from typing import Optional
import pandas as pd
from app.database.slot_db import (
    load_db,
    save_db,
    _next_token_id,
    LOCK,
)
from app.schemas.appointment import AppointmentToken, AppointmentStatus
from app.schemas.patient import PatientInfo
from app.config.logger import get_logger

logger = get_logger(__name__)


class TokenAssignmentError(Exception):
    """Raised when token assignment fails for any reason."""
    pass


class SlotDatabase:
    """
    Thread-safe slot database manager..
    Usage:
        db = SlotDatabase()
        token = db.assign_token(patient_info, language)
    """

    def assign_token(
        self,
        patient_info: PatientInfo,
        language: str,
    ) -> AppointmentToken:
        """

        Steps:
        1. Acquire the global lock
        2. Load current DB state
        3. Calculate next token ID
        4. Write new row with PENDING status
        5. Save to CSV
        6. Release lock
        7. Return AppointmentToken

        If any step fails, the lock is released and the error
        is raised — no partial writes are left in the DB.
        """
        if not patient_info.is_complete():
            raise TokenAssignmentError(
                f"Cannot assign token — missing fields: "
                f"{patient_info.missing_fields()}"
            )

        with LOCK:
            try:
                df = load_db()
                token_id = _next_token_id(df)
                now = datetime.now(UTC).isoformat()

                new_row = {
                    "token_id": token_id,
                    "patient_name": patient_info.name,
                    "patient_phone": patient_info.phone,
                    "symptoms": patient_info.symptoms,
                    "time_window": patient_info.time_window,
                    "language": str(language),
                    "status": AppointmentStatus.CONFIRMED.value,
                    "created_at": now,
                    "updated_at": now,
                }

                new_df = pd.concat(
                    [df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                save_db(new_df)

                token = AppointmentToken(
                    token_id=token_id,
                    patient_name=patient_info.name,
                    patient_phone=patient_info.phone,
                    symptoms=patient_info.symptoms,
                    time_window=patient_info.time_window,
                    language=str(language),
                    status=AppointmentStatus.CONFIRMED,
                )

                logger.info(
                    "token_assigned",
                    token_id=token_id,
                    patient_name=patient_info.name,
                    phone=patient_info.phone,
                )

                return token

            except Exception as e:
                logger.error(
                    "token_assignment_failed",
                    error=str(e),
                    patient_name=patient_info.name if patient_info else "unknown",
                )
                raise TokenAssignmentError(
                    f"Failed to assign token: {e}"
                ) from e

    def get_appointment_by_phone(
        self,
        phone: str,
    ) -> Optional[dict]:
        """
        Looks up an appointment by phone number.
        Returns the most recent appointment or None.
        """
        with LOCK:
            df = load_db()
            if df.empty:
                return None

            matches = df[df["patient_phone"] == phone]
            if matches.empty:
                return None

            latest = matches.iloc[-1]
            return latest.to_dict()

    def get_all_appointments(self) -> list[dict]:
        """Returns all appointments as a list of dicts."""
        with LOCK:
            df = load_db()
            if df.empty:
                return []
            return df.to_dict(orient="records")

    def cancel_token(self, token_id: str) -> bool:
        """
        Marks an appointment as cancelled.
        Returns True if found and cancelled, False if not found.
        """
        with LOCK:
            df = load_db()
            mask = df["token_id"] == token_id
            if not mask.any():
                logger.warning(
                    "cancel_token_not_found",
                    token_id=token_id,
                )
                return False

            df.loc[mask, "status"] = AppointmentStatus.CANCELLED.value
            df.loc[mask, "updated_at"] = datetime.now(UTC).isoformat()
            save_db(df)

            logger.info("token_cancelled", token_id=token_id)
            return True

    def get_stats(self) -> dict:
        """Returns basic stats about the slot database."""
        with LOCK:
            df = load_db()
            if df.empty:
                return {
                    "total": 0,
                    "confirmed": 0,
                    "cancelled": 0,
                    "pending": 0,
                }
            return {
                "total": len(df),
                "confirmed": len(df[df["status"] == "confirmed"]),
                "cancelled": len(df[df["status"] == "cancelled"]),
                "pending": len(df[df["status"] == "pending"]),
            }


# Single shared instance — imported everywhere
slot_db = SlotDatabase()