

from langchain_core.messages import AIMessage
from app.schemas.state import AgentState
from app.schemas.validation import ValidationStatus
from app.config.logger import get_logger

logger = get_logger(__name__)


def validator_node(state: AgentState) -> dict:

    patient_info = state.get("patient_info")

    if patient_info is None:
        logger.error("validator_called_with_no_patient_info")
        return {
            "validation_status": ValidationStatus.REJECTED,
            "is_blocked": True,
        }

    if not patient_info.is_complete():
        missing = patient_info.missing_fields()
        logger.warning(
            "validator_found_incomplete_info",
            missing=missing,
        )
        return {
            "validation_status": ValidationStatus.COLLECTING,
            "is_blocked": False,
        }

    logger.info(
        "validator_passed",
        patient_name=patient_info.name,
        phone=patient_info.phone,
    )

    return {
        "validation_status": ValidationStatus.CONFIRMED,
        "is_blocked": False,
    }
    
    