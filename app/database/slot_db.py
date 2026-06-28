

import threading
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional
from app.config.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("data/slots/appointments.csv")
LOCK = threading.Lock()  # one global lock for all DB operations

COLUMNS = [
    "token_id",
    "patient_name",
    "patient_phone",
    "symptoms",
    "time_window",
    "language",
    "status",
    "created_at",
    "updated_at",
]


# Token format: D-001 through D-999 (expandable)
TOKEN_PREFIX = "D"
TOKEN_START = 100 


def _ensure_db_exists() -> None:
    """
    Creates the CSV file and directory if they don't exist.
    Called on every DB operation — safe to call multiple times.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(DB_PATH, index=False)
        logger.info("slot_db_created", path=str(DB_PATH))


def load_db() -> pd.DataFrame:
    """Loads the appointments CSV into a DataFrame."""
    _ensure_db_exists()
    df = pd.read_csv(DB_PATH, dtype=str)
    return df.fillna("")


def save_db(df: pd.DataFrame) -> None:
    """Saves DataFrame back to CSV atomically."""
    _ensure_db_exists()
    df.to_csv(DB_PATH, index=False)


def _next_token_id(df: pd.DataFrame) -> str:
    """
    Calculates the next sequential token ID.
    Reads existing tokens, finds the highest number, increments by 1.
    Always returns format D-XXX.
    """
    if df.empty or "token_id" not in df.columns:
        return f"{TOKEN_PREFIX}-{TOKEN_START}"

    existing = df["token_id"].dropna().tolist()
    if not existing:
        return f"{TOKEN_PREFIX}-{TOKEN_START}"

    # Parse numeric parts from D-XXX format
    nums = []
    for t in existing:
        try:
            parts = str(t).split("-")
            if len(parts) == 2 and parts[0] == TOKEN_PREFIX:
                nums.append(int(parts[1]))
        except (ValueError, IndexError):
            continue

    if not nums:
        return f"{TOKEN_PREFIX}-{TOKEN_START}"

    next_num = max(nums) + 1
    return f"{TOKEN_PREFIX}-{next_num}"

