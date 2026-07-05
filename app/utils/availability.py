from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

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

# Requires am/pm (or 10am-style) so day numbers like "July 07" are not treated as times
TIME_WITH_MERIDIEM = re.compile(
    r"\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)\b"
    r"|\b(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)\b"
    r"|\b(\d{1,2})(am|pm)\b",
    re.IGNORECASE,
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


def requests_tomorrow(text: str) -> bool:
    return "tomorrow" in text.lower() or "kal" in text.lower()


def requests_sunday(text: str) -> bool:
    return "sunday" in text.lower()


def is_booking_intent(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in [
        "book", "booking", "appointment", "schedule", "meeting",
    ])


def is_availability_question(text: str) -> bool:
    t = text.lower().strip()

    # "Book for Tuesday" is booking intent, not a schedule-only question
    if is_booking_intent(t) and parse_appointment_date(text) and not requests_today(t):
        return False

    if requests_today(t) or requests_tomorrow(t) or requests_sunday(t):
        return True
    if any(w in t for w in WEEKDAY_NAMES):
        schedule_words = [
            "timing", "time", "hours", "open", "available", "doctor",
            "slot", "when", "schedule",
        ]
        if any(w in t for w in schedule_words) and not is_booking_intent(t):
            return True
    phrases = [
        "is this possible", "is it possible", "is that possible",
        "can i book", "can i come", "can i get an appointment",
        "available today", "open today", "booking today",
        "appointment today", "possible today", "are you open",
        "still open", "open now", "clinic open",
        "what time", "what timing", "doctor available",
        "doctor tomorrow", "doctor today",
    ]
    return any(p in t for p in phrases)


def _next_weekday(from_date: date, weekday_idx: int) -> date:
    days_ahead = (weekday_idx - from_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return from_date + timedelta(days=days_ahead)


def parse_appointment_date(text: str, now: datetime | None = None) -> date | None:
    """Parse absolute or relative dates from free-text appointment requests."""
    now = now or datetime.now(CLINIC_TZ)
    today = now.date()
    t = text.lower()

    if "day after tomorrow" in t or "parson" in t:
        return today + timedelta(days=2)

    if requests_tomorrow(t):
        return today + timedelta(days=1)

    if "yesterday" in t:
        return today - timedelta(days=1)

    week_match = re.search(
        r"(?:in|after)\s+(\d+|one|two|three|four|five|six)\s+weeks?",
        t,
    )
    if week_match:
        raw = week_match.group(1)
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        weeks = word_map.get(raw, int(raw) if raw.isdigit() else 1)
        return today + timedelta(weeks=weeks)

    if "one week" in t or "1 week" in t or "a week" in t:
        if any(k in t for k in ["after", "from now", "later", "next week", "in a week", "in one week"]):
            return today + timedelta(weeks=1)

    day_match = re.search(
        r"(?:in|after)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?",
        t,
    )
    if day_match:
        raw = day_match.group(1)
        word_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        days = word_map.get(raw, int(raw) if raw.isdigit() else 0)
        if days:
            return today + timedelta(days=days)

    for weekday_idx, weekday in enumerate(WEEKDAY_NAMES):
        if f"next {weekday}" in t:
            return _next_weekday(today, weekday_idx)
        if f"this {weekday}" in t:
            days_ahead = (weekday_idx - today.weekday()) % 7
            return today + timedelta(days=days_ahead)

    for weekday_idx, weekday in enumerate(WEEKDAY_NAMES):
        if re.search(rf"\b{weekday}\b", t):
            days_ahead = (weekday_idx - today.weekday()) % 7
            if days_ahead == 0 and not requests_today(t):
                days_ahead = 7
            return today + timedelta(days=days_ahead)

    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            return None

    m = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            return None

    for month_name, month_num in MONTH_NAMES.items():
        m = re.search(rf"\b{month_name}\s+(\d{{1,2}})(?:,?\s*(\d{{4}}))?\b", t)
        if m:
            day = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else now.year
            try:
                parsed = date(year, month_num, day)
                if not m.group(2) and parsed < today:
                    parsed = date(year + 1, month_num, day)
                return parsed
            except ValueError:
                return None
        m = re.search(rf"\b(\d{{1,2}})\s+{month_name}(?:,?\s*(\d{{4}}))?\b", t)
        if m:
            day = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else now.year
            try:
                parsed = date(year, month_num, day)
                if not m.group(2) and parsed < today:
                    parsed = date(year + 1, month_num, day)
                return parsed
            except ValueError:
                return None

    return None


def _parse_time_match(match: re.Match) -> time | None:
    """Parse a regex match into a time object."""
    if match.group(1) is not None:
        hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
    elif match.group(4) is not None:
        hour, minute, meridiem = int(match.group(4)), 0, match.group(5)
    else:
        hour, minute, meridiem = int(match.group(6)), 0, match.group(7)

    meridiem = meridiem.lower().replace(".", "")
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    try:
        return time(hour, minute)
    except ValueError:
        return None


def _extract_time_phrase(text: str) -> str | None:
    match = TIME_WITH_MERIDIEM.search(text.lower())
    if not match:
        return None
    parsed = _parse_time_match(match)
    if parsed is None:
        return None
    return parsed.strftime("%I:%M %p").lstrip("0")


def _validate_time_of_day(text: str) -> str | None:
    match = TIME_WITH_MERIDIEM.search(text.lower())
    if not match:
        return None
    requested = _parse_time_match(match)
    if requested is None:
        return "That time doesn't look valid. Please use a time like 10:00 AM or 4:30 PM."
    if requested < OPEN_TIME or requested >= CLOSE_TIME:
        return (
            f"I'm sorry, {requested.strftime('%I:%M %p').lstrip('0')} is outside clinic hours. "
            f"We are open {OPEN_TIME.strftime('%I:%M %p').lstrip('0')} to "
            f"{CLOSE_TIME.strftime('%I:%M %p').lstrip('0')}, Monday to Saturday."
        )
    return None


def describe_date_hours(target: date) -> str:
    if target.weekday() == 6:
        return "The clinic is closed on Sundays."
    return (
        f"On {target.strftime('%A, %B %d, %Y')}, the clinic is open "
        f"from {OPEN_TIME.strftime('%I:%M %p').lstrip('0')} to "
        f"{CLOSE_TIME.strftime('%I:%M %p').lstrip('0')}."
    )


def normalize_time_window(text: str, now: datetime | None = None) -> str:
    """Convert relative dates into a concrete, readable appointment string."""
    now = now or datetime.now(CLINIC_TZ)
    parsed_date = parse_appointment_date(text, now)
    time_part = _extract_time_phrase(text)

    if parsed_date:
        base = parsed_date.strftime("%A, %B %d, %Y")
        if time_part:
            return f"{base} at {time_part}"
        return base

    return text.strip()


def validate_time_window(time_window: str) -> str | None:
    """Return a rejection message if the requested slot is not bookable."""
    now = datetime.now(CLINIC_TZ)
    tw = time_window.lower().strip()

    if not tw or len(tw) < 3:
        return "Please provide a valid date and time for your appointment."

    parsed = parse_appointment_date(time_window, now)
    if parsed is None and not requests_today(tw):
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", tw):
            return (
                "That date doesn't look valid. Please use a real date such as "
                f"{now.strftime('%B %d, %Y')} or say 'tomorrow at 10 AM'."
            )

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

    time_rejection = _validate_time_of_day(time_window)
    if time_rejection:
        return time_rejection

    return None


def get_availability_response(
    time_window: str | None,
    user_message: str,
) -> str | None:
    """Answer schedule / availability questions without hallucinating slots."""
    if not is_availability_question(user_message):
        return None

    now = datetime.now(CLINIC_TZ)
    # Prefer the date mentioned in the latest message, not a stale stored value
    target = parse_appointment_date(user_message, now)
    if target is None and time_window and not requests_today(user_message):
        target = parse_appointment_date(time_window, now)

    combined = user_message.lower()

    if target:
        if target < now.date():
            return (
                f"{target.strftime('%A, %B %d, %Y')} has already passed. "
                "Please choose a future date."
            )
        if target.weekday() == 6:
            return (
                f"The clinic is closed on {target.strftime('%A, %B %d, %Y')}. "
                f"{HOURS_MESSAGE} Would you like another day?"
            )
        hours = describe_date_hours(target)
        suffix = " Would you like me to book an appointment for that day?"
        if _extract_time_phrase(user_message):
            rejection = validate_time_window(user_message)
            if rejection:
                return rejection
            suffix = " I can continue your booking with that time if you'd like."
        return hours + suffix

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

    if requests_tomorrow(combined):
        tomorrow = now.date() + timedelta(days=1)
        if tomorrow.weekday() == 6:
            return (
                "Tomorrow is Sunday and the clinic is closed. "
                "Would you like to book for Monday instead?"
            )
        return describe_date_hours(tomorrow) + " Would you like to book for tomorrow?"

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
