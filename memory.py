"""
memory.py - Saves past trip plans to SQLite and provides context for the agent.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from collections import Counter

DB_PATH = "travellk.db"


@contextmanager
def get_connection():
    """
    Single place that owns connection open/commit/close.
    Every DB function below uses this instead of repeating
    connect() ... commit() ... close() by hand.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the trips table and migrate in any columns added later (e.g. rating)."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                days INTEGER,
                interests TEXT,
                budget TEXT,
                itinerary TEXT,
                rating INTEGER,
                timestamp TEXT
            )
        """)
        # Safe migration for DBs created before the rating column existed.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trips)")}
        if "rating" not in existing_cols:
            conn.execute("ALTER TABLE trips ADD COLUMN rating INTEGER")


def save_trip(days, interests, budget, itinerary, rating=None) -> int:
    """Insert a new trip and return its id (so callers can update it later, e.g. on refine)."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO trips (days, interests, budget, itinerary, rating, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (days, interests, budget, itinerary, rating, datetime.now().isoformat()))
        return cursor.lastrowid


def update_trip(trip_id: int, itinerary: str):
    """
    Update an existing trip's itinerary in place instead of inserting a new row.
    Used when the user refines an itinerary they already generated, so refining
    the same trip 3 times doesn't create 3 extra DB rows and skew preference stats.
    """
    with get_connection() as conn:
        conn.execute("""
            UPDATE trips SET itinerary = ?, timestamp = ? WHERE id = ?
        """, (itinerary, datetime.now().isoformat(), trip_id))


def set_trip_rating(trip_id: int, rating: int):
    """Attach a 1-5 star rating to a trip after the fact (rating happens after generation)."""
    with get_connection() as conn:
        conn.execute("UPDATE trips SET rating = ? WHERE id = ?", (rating, trip_id))


def get_recent_trips(limit=5):
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT days, interests, budget, timestamp, itinerary
            FROM trips ORDER BY id DESC LIMIT ?
        """, (limit,))
        return cursor.fetchall()


def get_total_trips() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]


def get_memory_context(limit=3) -> str:
    """
    Returns a formatted string of recent trips to inject into AI prompts.
    This is how the agent 'remembers' what kind of trips the user has planned before.
    """
    recent = get_recent_trips(limit)
    if not recent:
        return ""
    lines = ["User's previous trip history (use this to personalise recommendations):"]
    for i, (days, interests, budget, timestamp, _) in enumerate(recent, 1):
        date_str = timestamp[:10] if timestamp else "unknown date"
        lines.append(f"  Trip {i}: {days} days | {interests} | {budget} | planned on {date_str}")
    return "\n".join(lines)


def load_itinerary(trip_index):
    """Load a past itinerary by index (0 = most recent)."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT days, interests, budget, itinerary, timestamp
            FROM trips ORDER BY id DESC LIMIT 1 OFFSET ?
        """, (trip_index,))
        return cursor.fetchone()


def get_user_preferences():
    """
    Analyses past trips to figure out what the user prefers.
    Returns preferred budget, average trip length, and top interests.
    Used to auto-personalise future itineraries.
    """
    with get_connection() as conn:
        rows = conn.execute("SELECT interests, budget, days FROM trips").fetchall()

    if len(rows) < 2:
        return None  # not enough data yet

    preferred_budget = Counter(r[1] for r in rows).most_common(1)[0][0]
    avg_days = round(sum(r[2] for r in rows) / len(rows))

    all_interests = []
    for r in rows:
        all_interests.extend(i.strip() for i in r[0].split(","))
    top_interests = [i for i, _ in Counter(all_interests).most_common(3)]

    return {
        "preferred_budget": preferred_budget,
        "avg_trip_length": avg_days,
        "top_interests": top_interests,
    }


def get_smart_memory_context(limit=3) -> str:
    """
    Enhanced version of get_memory_context().
    Injects past trips AND a preference summary so the LLM knows
    what this user consistently enjoys.
    """
    recent = get_recent_trips(limit)
    if not recent:
        return ""

    prefs = get_user_preferences()
    lines = ["User's previous trip history (use this to personalise recommendations):"]
    for i, (days, interests, budget, timestamp, _) in enumerate(recent, 1):
        date_str = timestamp[:10] if timestamp else "unknown date"
        lines.append(f"  Trip {i}: {days} days | {interests} | {budget} | planned on {date_str}")

    if prefs:
        lines.append("")
        lines.append("Based on their history, this user PREFERS:")
        lines.append(f"  - Budget tier: {prefs['preferred_budget']}")
        lines.append(f"  - Average trip length: {prefs['avg_trip_length']} days")
        lines.append(f"  - Top interests: {', '.join(prefs['top_interests'])}")
        lines.append("Use these preferences to make smarter, more personalised suggestions.")

    return "\n".join(lines)


def get_top_rated_trips(min_rating=4, limit=3):
    """Returns the highest rated past trips, useful for showing the user their best itineraries."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT days, interests, budget, timestamp, itinerary, rating
            FROM trips
            WHERE rating >= ?
            ORDER BY rating DESC, id DESC
            LIMIT ?
        """, (min_rating, limit))
        return cursor.fetchall()


def get_destination_frequency():
    """
    Counts how many times each destination appears across all saved itineraries.
    Useful for: 'You always go to Ella — want to try somewhere new this time?'
    """
    from agent import extract_place_names  # local import avoids a circular import at module load

    with get_connection() as conn:
        rows = conn.execute("SELECT itinerary FROM trips").fetchall()

    destination_counts = Counter()
    for (itinerary,) in rows:
        destination_counts.update(extract_place_names(itinerary))
    return destination_counts.most_common(10)


init_db()
