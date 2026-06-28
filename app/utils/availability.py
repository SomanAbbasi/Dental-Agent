from datetime import datetime, time, date, timedelta
from zoneinfo import ZoneInfo
import re

CLINIC_TZ = ZoneInfo("Asia/Karachi")
OPEN_TIME = time(9, 0)
CLOSE_TIME = time(19, 0)

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

WEEKDAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]

HOURS_MESSAGE = (
    "Our clinic hours are Monday to Saturday, 9:00 AM to 7:00 PM. "
    "We are closed on Sundays and public holidays."
)


def is_clinic_open_now(now: datetime | None = None) -> bool:
    now = now or datetime.now(CLINIC_TZ)
    if now.weekday() == 6:
        return False
    return OPEN_TIME <= now.time() < CLOSE_TIME


def requests_today(text: str) -> bool:
    t = text.lower()
    return any(
        k in t
        for k in [
            "today", "tonight", "this evening", "right now", "asap",
            "this afternoon", "this morning", "now",
        ]
    )


def requests_sunday(text: str) -> bool:
    return "sunday" in text.lower()


def is_availability_question(text: str) -> bool:
    t = text.lower().strip()
    if requests_today(t) or requests_sunday(t):
        return True
    phrases = [
        "is this possible", "is it possible", "is that possible",
        "can i book", "can i come", "can i get an appointment",
        "available today", "open today", "booking today",
        "appointment today", "possible today", "are you open",
        "still open", "open now", "clinic open",
    ]
    return any(p in t for p in phrases)


def parse_appointment_date(text: str, now: datetime | None = None) -> date | None:
    """Best-effort parse of a date mentioned in free-text appointment requests."""
    now = now or datetime.now(CLINIC_TZ)
    t = text.lower()

    if "yesterday" in t:
        return now.date() - timedelta(days=1)

    for weekday_idx, weekday in enumerate(WEEKDAY_NAMES):
        if f"last {weekday}" in t:
            days_back = (now.weekday() - weekday_idx) % 7
            if days_back == 0:
                days_back = 7
            return now.date() - timedelta(days=days_back)

    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            pass

    m = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            pass

    for month_name, month_num in MONTH_NAMES.items():
        m = re.search(rf"\b{month_name}\s+(\d{{1,2}})(?:,?\s*(\d{{4}}))?\b", t)
        if m:
            day = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else now.year
            try:
                return date(year, month_num, day)
            except ValueError:
                pass
        m = re.search(rf"\b(\d{{1,2}})\s+{month_name}(?:,?\s*(\d{{4}}))?\b", t)
        if m:
            day = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else now.year
            try:
                return date(year, month_num, day)
            except ValueError:
                pass

    return None


def validate_time_window(time_window: str) -> str | None:
    """Return a rejection message if the requested slot is not bookable."""
    now = datetime.now(CLINIC_TZ)
    tw = time_window.lower()

    parsed = parse_appointment_date(time_window, now)
    if parsed:
        if parsed < now.date():
            return (
                f"I'm sorry, {parsed.strftime('%B %d, %Y')} has already passed. "
                f"Today is {now.strftime('%B %d, %Y')}. "
                "Please choose today or a future date."
            )
        if parsed.weekday() == 6:
            return (
                "I'm sorry, the clinic is closed on Sundays. "
                f"{HOURS_MESSAGE} Would you like to choose another day?"
            )

    if requests_sunday(tw):
        return (
            "I'm sorry, the clinic is closed on Sundays. "
            f"{HOURS_MESSAGE} Would you like to choose another day?"
        )

    if requests_today(tw):
        if now.weekday() == 6:
            return (
                "I'm sorry, the clinic is closed today (Sunday). "
                f"{HOURS_MESSAGE} Would you like to book for Monday instead?"
            )
        if now.time() >= CLOSE_TIME:
            return (
                "I'm sorry, we're closed for today — we close at 7:00 PM. "
                "Would you like to book for tomorrow or another day?"
            )

    return None


def get_availability_response(
    time_window: str | None,
    user_message: str,
) -> str | None:
    """
    Build a reply when the caller asks about same-day or immediate availability.
    Returns None if the message is not an availability question.
    """
    if not is_availability_question(user_message):
        return None

    now = datetime.now(CLINIC_TZ)
    combined = f"{time_window or ''} {user_message}".lower()

    if time_window:
        rejection = validate_time_window(time_window)
        if rejection:
            return rejection

    if now.weekday() == 6 or (requests_today(combined) and now.time() >= CLOSE_TIME):
        day_label = "today (Sunday)" if now.weekday() == 6 else "today"
        return (
            f"I'm sorry, the clinic is closed {day_label}. "
            f"{HOURS_MESSAGE} "
            "Would you like to book for the next available day?"
        )

    if requests_sunday(combined):
        return (
            "I'm sorry, the clinic is closed on Sundays. "
            f"{HOURS_MESSAGE} Please choose another day."
        )

    if requests_today(combined) and is_clinic_open_now(now):
        return (
            "Yes, we can book an appointment for later today — "
            "we're open until 7:00 PM. What time would you prefer?"
        )

    if requests_today(combined):
        return (
            "I'm sorry, we're not able to book for today at this time. "
            f"{HOURS_MESSAGE} Would you like to choose another day?"
        )

    if is_clinic_open_now(now):
        return (
            f"Yes, the clinic is open right now until 7:00 PM. "
            f"{HOURS_MESSAGE} Would you like to book an appointment?"
        )

    return (
        f"We're currently closed. {HOURS_MESSAGE} "
        "Would you like to book for the next available day?"
    )
