"""General-purpose helper functions used across the app."""
from datetime import date, timedelta

COMPANY_HOLIDAYS = [
    date(2026, 1, 1),
    date(2026, 1, 26),
    date(2026, 8, 15),
    date(2026, 10, 2),
    date(2026, 12, 25),
]


def calculate_working_days(start_date: date, end_date: date) -> int:
    """Calculates total days excluding weekends (Sat/Sun) and holidays."""
    working_days = 0
    current_date = start_date

    while current_date <= end_date:
        is_weekend = current_date.weekday() >= 5
        is_holiday = current_date in COMPANY_HOLIDAYS

        if not is_weekend and not is_holiday:
            working_days += 1

        current_date += timedelta(days=1)

    return working_days
