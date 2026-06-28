
from fastapi import APIRouter, HTTPException
from app.api.schemas import AppointmentListResponse, AppointmentRecord
from app.database.token_manager import slot_db
from app.config.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/appointments", response_model=AppointmentListResponse)
async def list_appointments():
    """Returns all booked appointments and stats."""
    try:
        all_appointments = slot_db.get_all_appointments()
        stats = slot_db.get_stats()

        records = [
            AppointmentRecord(**appt)
            for appt in all_appointments
            if appt.get("token_id")
        ]

        return AppointmentListResponse(
            total=len(records),
            appointments=records,
            stats=stats,
        )
    except Exception as e:
        logger.error("list_appointments_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/appointments/{token_id}")
async def get_appointment(token_id: str):
    """Look up a specific appointment by token ID."""
    all_appointments = slot_db.get_all_appointments()
    for appt in all_appointments:
        if appt.get("token_id") == token_id:
            return appt
    raise HTTPException(
        status_code=404,
        detail=f"Appointment {token_id} not found",
    )


@router.delete("/appointments/{token_id}")
async def cancel_appointment(token_id: str):
    """Cancel an appointment by token ID."""
    result = slot_db.cancel_token(token_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Appointment {token_id} not found",
        )
    return {"message": f"Appointment {token_id} cancelled successfully"}
