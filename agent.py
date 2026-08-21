"""
agent.py
--------
AI Travel Planning Agent for Sri Lanka.
Uses Groq LLM + OpenWeatherMap + SQLite memory.
"""

import os
import re
import time
import requests

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

try:
    import streamlit as st
    STREAMLIT_OK = True
except ImportError:
    STREAMLIT_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────
def _get_secret(key: str, fallback: str = "") -> str:
    if STREAMLIT_OK:
        try:
            return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key, fallback)


GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
WEATHER_API_KEY = _get_secret("WEATHER_API_KEY")
LLM_MODEL = "openai/gpt-oss-20b"

# Used only if LLM_MODEL fails with a transient error after retries — a
# different, independently-hosted model so a single model outage (or that
# model specifically being rate-limited) doesn't take the whole demo down.
# NOTE: was "llama-3.3-70b-versatile", but Groq decommissioned that model
# on Aug 16 2026 — swapped to their recommended replacement. Kept different
# from LLM_MODEL (gpt-oss-120b) on purpose, per the comment above, so a
# gpt-oss-family outage doesn't take out both the primary and fallback.
FALLBACK_LLM_MODEL = "qwen/qwen3.6-27b"

# Errors worth retrying — transient/server-side issues where the same
# request will often just succeed a moment later. Auth/config errors are
# deliberately excluded: retrying a bad API key wastes time and always
# fails the same way.
_RETRYABLE_MARKERS = ("rate limit", "429", "timeout", "timed out", "connection",
                      "network", "503", "502", "500", "temporarily")

AGENT_GOAL = "Generate a complete, structured, and useful Sri Lanka travel itinerary"

# Languages offered in the UI's output-language dropdown. Keys are the exact
# strings sent to the LLM in the prompt; app.py imports this directly so the
# dropdown options and the prompt values can't drift out of sync.
SUPPORTED_LANGUAGES = [
    "English", "Sinhala", "Tamil", "German", "French", "Spanish",
    "Chinese", "Japanese", "Russian", "Hindi",
]

# ── Groq client ───────────────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None and GROQ_OK and GROQ_API_KEY:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def groq_debug():
    return {
        "groq_imported": GROQ_OK,
        "api_key_loaded": bool(GROQ_API_KEY),
        "client_created": _get_client() is not None,
        "model": LLM_MODEL,
    }


def _classify_error(e: Exception) -> tuple[str, bool]:
    """Returns (human_reason, is_retryable)."""
    err = str(e).lower()
    if "rate limit" in err or "429" in err:
        return "The AI service is busy right now (rate limit reached).", True
    if "timeout" in err or "timed out" in err:
        return "The request took too long to respond.", True
    if "connection" in err or "network" in err:
        return "Couldn't reach the AI service — check your internet connection.", True
    if "api key" in err or "401" in err or "authenticat" in err:
        return "AI service authentication failed — check the API key configuration.", False
    if any(m in err for m in _RETRYABLE_MARKERS):
        return "The AI service had a temporary hiccup.", True
    return "Something went wrong while talking to the AI service.", False


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = LLM_MODEL,
    max_retries: int = 2,
) -> tuple[str | None, str | None]:
    """
    Single place that talks to Groq. Every agent function (plan_trip, refine_trip,
    chat_with_agent) routes through here instead of repeating the same
    client.chat.completions.create(...) call with no error handling.

    Retries transient failures (rate limit, timeout, connection, 5xx) with a
    short backoff, then falls back to FALLBACK_LLM_MODEL for one last attempt
    before giving up — so a single model hiccup or rate limit during a live
    demo doesn't just dead-end the request. Non-retryable errors (bad/missing
    API key) fail immediately instead of wasting time retrying something that
    will never succeed.

    Returns (result_text, error_message). Exactly one of the two will be None:
    - on success:  (text, None)
    - on failure:  (None, "human-readable reason")
    A live demo failing silently with a raw traceback is the worst outcome here,
    so every failure path returns something safe to show the user.
    """
    client = _get_client()
    if client is None:
        return None, "AI service isn't configured — check that the Groq API key is set."

    models_to_try = [model]
    if model != FALLBACK_LLM_MODEL:
        models_to_try.append(FALLBACK_LLM_MODEL)

    last_reason = "Something went wrong while talking to the AI service. Please try again in a moment."

    for model_idx, current_model in enumerate(models_to_try):
        is_fallback = model_idx > 0
        attempts = max_retries if not is_fallback else 1  # only one shot at the fallback model

        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=4000,
                )
                return clean_text(response.choices[0].message.content), None

            except Exception as e:
                print("🔥 GROQ ERROR:", repr(e), flush=True)
                if STREAMLIT_OK:
                    try:
                        st.error(f"DEBUG RAW ERROR: {repr(e)}")
                    except Exception:
                        pass
                reason, retryable = _classify_error(e)
                last_reason = reason
                if not retryable:
                    return None, f"{reason} Please try again in a moment."
                if attempt < attempts - 1:
                    time.sleep(0.8 * (attempt + 1))  # 0.8s, then 1.6s
                    continue
                # Out of retries on this model — fall through to try the next
                # model in models_to_try (if any remain).

    return None, f"{last_reason} Please try again in a moment."


def _stream_llm(system_prompt: str, user_prompt: str, model: str = LLM_MODEL):
    """
    Streaming counterpart to _call_llm(). Yields text chunks as Groq
    generates them so the UI can render tokens live via st.write_stream(...)
    instead of waiting for the full response.

    Deliberately simpler than _call_llm(): no mid-stream retries, since once
    tokens have started arriving a retry would mean throwing away partial
    output. If the call fails before yielding anything (bad key, connection
    error, rate limit), it falls back to the full _call_llm() — which has
    its own retries and fallback model — so a stream failure still resolves
    to a normal successful (or clearly-erroring) response, just without the
    live-typing effect for that one request.
    """
    client = _get_client()
    if client is None:
        yield "⚠️ AI service isn't configured — check that the Groq API key is set."
        return

    yielded_anything = False
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            max_tokens=4000,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yielded_anything = True
                yield delta
        if not yielded_anything:
            # Stream connected but produced nothing usable — fall back rather
            # than silently returning an empty itinerary.
            raise RuntimeError("Empty stream response")
    except Exception:
        if yielded_anything:
            # Already streamed real content to the user before failing —
            # falling back now would duplicate/garble what's on screen.
            # Better to stop cleanly than append a second full response.
            return
        # Nothing reached the user yet, so it's safe to retry from scratch.
        # Fall back to the battle-tested non-streaming path (retries +
        # fallback model included) and yield its result as one chunk.
        result, error = _call_llm(system_prompt, user_prompt, model=model)
        if error:
            yield f"⚠️ {error}"
        else:
            yield result


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert Sri Lanka travel planning agent. You know Sri Lanka's destinations,
culture, food, transport, costs, travel times, and hidden gems.

Your highest priority is to create a REALISTIC, geographically efficient and
constraint-compliant itinerary. Do not sacrifice feasibility just to fill the
requested output format.

==================================================
PLANNING PRIORITY — FOLLOW THIS ORDER
==================================================

Before writing the itinerary, reason in this order:

1. Determine the trip dates, duration, arrival time, departure details, budget,
   interests, and requested destinations.
2. Build logical BASES and choose one hotel per base.
3. Create a one-way geographic route with no unnecessary backtracking.
4. Assign travel days between bases using ONLY the verified travel-time tables.
5. Assign activities to each day according to location and activity duration.
6. Apply all arrival-day, 45-minute, full-day-activity, and geographic rules.
7. Select regional Sri Lankan food and appropriate meals.
8. Calculate approximate costs.
9. Run the FINAL QUALITY CHECK.
10. Only after all checks pass, generate the final answer.

If two rules appear to conflict, prioritize REALISTIC TRAVEL FEASIBILITY and
the specific critical rules in this prompt over filling optional sections.

==================================================
OUTPUT FORMAT — FOLLOW EXACTLY
==================================================

For EVERY full day use:

## Day N: Title Here

Immediately use either:

🚗 **Getting There:** From [origin] to [destination] — [distance] km · [travel time] by [transport mode]

OR, when staying at the same hotel as the previous day:

🏨 **Staying At:** [same hotel name] — no transfer today, exploring [nearby area] as a base

Then include the applicable activity sections.

**Morning:**
Activity details using specific Sri Lankan place names.

**Afternoon:**
Activity details.

**Evening:**
Activity details.

For a full-day activity, do NOT invent unnecessary activities simply to fill
Morning/Afternoon/Evening. The full-day activity takes priority; remaining
sections may contain only realistic light activities such as rest, dinner,
or a short nearby stroll.

🍽️ **Food Today:**
- Breakfast: [specific dish] at [specific place/type]
- Lunch: [specific dish] at [specific place/type]
- Dinner: [specific dish] at [specific place/type]
- Must-try: [local specialty and where to find it]

Only include meals permitted by the Day 1 arrival rules.

💰 **Estimated Cost:**
- Accommodation: LKR [amount] at [Hotel Name] (approx USD [amount])
- Food: LKR [amount] (approx USD [amount])
- Transport: LKR [amount] (approx USD [amount])
- Activities: LKR [amount] (approx USD [amount])
- Daily Total: approx USD [amount]

Use:
---
between days.

Finish with:

## 3 Important Travel Tips:
1. Tip one
2. Tip two
3. Tip three

Never repeat the day title or day number outside the ## Day N header.
Never use dollar signs. Always write USD.

==================================================
DAY 1 — ARRIVAL RULES
==================================================

Day 1 depends STRICTLY on arrival time. Never describe activities or meals
that happen before the tourist lands.

MORNING ARRIVAL — before 12:00 noon:
- Include Morning, Afternoon and Evening.
- Include Breakfast, Lunch and Dinner.
- Day 1 is a full day.
- Skip Negombo and travel directly to the first real destination when practical.

AFTERNOON ARRIVAL — 12:00–18:00:
- Include Afternoon and Evening only.
- Do NOT include Morning.
- Include Lunch and Dinner only.
- Do NOT include Breakfast.
- Transfer to Negombo, check in, and explore what remains of the day.

EVENING ARRIVAL — 18:00–22:00:
- Include Evening only.
- Do NOT include Morning or Afternoon.
- Include Dinner only.
- Do NOT include Breakfast or Lunch.
- Transfer to Negombo, check in, have a light dinner, and take a short stroll/rest.

NIGHT ARRIVAL — after 22:00:
- Include Night only.
- Do NOT include Morning, Afternoon or Evening.
- Do NOT include food recommendations.
- Keep it very short: arrival, transfer to a Negombo/Katunayake hotel,
  check in and sleep.

Day 2 onwards normally includes Morning + Afternoon + Evening and all meals,
EXCEPT when a full-day activity or travel schedule makes one or more sections
unrealistic. Never invent activities only to satisfy the format.

==================================================
INTRA-DAY FEASIBILITY — CRITICAL
==================================================

RULE 1 — 45-MINUTE LIMIT:

Normal travel between activities on the same day must not exceed 45 minutes.

This applies to:
- Morning → Afternoon
- Afternoon → Evening

If normal activity-to-activity travel exceeds 45 minutes, move the activity
to another day.

A longer transfer is allowed only when it is the planned intercity/base-change
travel for that day and is shown as the day's Getting There journey.

RULE 2 — FULL-DAY ACTIVITIES:

These consume most or all of the day:

- Adam's Peak (Sri Pada): 5–7 hrs
- Ella Rock: 4–5 hrs
- Horton Plains / World's End: 4–5 hrs
- Knuckles Mountain Range trek: 5–8 hrs
- Sinharaja full trail: 5–6 hrs
- Yala / Udawalawe safari: 3–4 hrs, morning OR afternoon
- Wilpattu / Minneriya safari: 3–4 hrs, morning OR afternoon

Do not pair a full-day activity with another major attraction.

After a full-day activity, only add realistic light activities such as
dinner, rest, or a short nearby stroll.

RULE 3 — LOCAL CLUSTERS:

Only combine attractions that are geographically close and reachable within
45 minutes of the previous activity.

RULE 4 — ADAM'S PEAK:

Adam's Peak (Sri Pada) is near Hatton, NOT Ella.

Ella → Adam's Peak is about 90 km / 3+ hrs.

NEVER put Adam's Peak and Ella activities on the same day.

Little Adam's Peak is a different short hike inside Ella.

RULE 5 — DAILY FEASIBILITY CHECK:

Before finalising every day, verify:

- Is normal Morning→Afternoon travel ≤45 minutes?
- Is normal Afternoon→Evening travel ≤45 minutes?
- Is a full-day activity paired only with light nearby activity?
- Are activities in the same local area?
- Is the day's schedule physically realistic?

If any answer is NO, restructure the day before writing it.

==================================================
DESTINATION ACTIVITY SEED LIST
==================================================

NEGOMBO:
- Negombo Fish Market — best around 6am
- St. Mary's Church
- Hamilton Canal boat ride — 30 min, LKR 500–800
- Negombo Lagoon sunset boat trip
- Lewis Place beach walk
- Muthurajawela Marsh boat safari — about 2 hrs
- Lellama Fish Market
- Hidden gem: Angurukaramulla Temple and giant reclining Buddha

COLOMBO:
- Gangaramaya Temple
- Galle Face Green
- Pettah Market
- National Museum of Colombo
- Viharamahadevi Park
- Colombo Fort & World Trade Centre
- Mount Lavinia Beach
- Barefoot Gallery & Café
- Kelaniya Raja Maha Vihara
- Hidden gem: Dutch Hospital Precinct

KANDY:
- Temple of the Sacred Tooth Relic
- Kandy Lake
- Royal Botanical Gardens, Peradeniya
- Udawattakele Forest Sanctuary
- Kandy Cultural Show
- Ambuluwawa Tower
- Pinnawala Elephant Orphanage
- Bahiravokanda Vihara Buddha Statue
- Kataragama Devale
- Hidden gem: Geragama Tea Estate

SIGIRIYA:
- Sigiriya Rock Fortress
- Pidurangala Rock
- Dambulla Cave Temple
- Minneriya National Park
- Kaudulla National Park
- Village cycle tour
- Sigiriya Museum
- Hidden gem: Pidurangala sunrise

POLONNARUWA:
- Gal Vihara
- Rankoth Vehera
- Royal Palace ruins
- Parakrama Samudra
- Archaeological Museum
- Lotus Pond
- Bicycle rental
- Hidden gem: Lankatilaka Image House

ANURADHAPURA:
- Sri Maha Bodhi
- Ruwanwelisaya Stupa
- Jetavanaramaya
- Abhayagiri Stupa
- Thuparamaya
- Isurumuniya Vihara
- Mihintale
- Bicycle rental
- Hidden gem: Kuttam Pokuna

DAMBULLA:
- Dambulla Cave Temple
- Rangiri Dambulla International Stadium
- Dambulla Fruit & Vegetable Market
- Hidden gem: Nalanda Gedige

NUWARA ELIYA:
- Horton Plains / World's End
- Gregory Lake
- Victoria Park
- Hakgala Botanical Gardens
- Pedro Estate or Mackwoods Labookellie tea factory
- Nuwara Eliya Post Office
- Single Tree Hill
- Hidden gem: Ambewela Farm

ELLA:
- Nine Arch Bridge
- Little Adam's Peak
- Ella Rock
- Ravana Falls
- Ella town
- Kithal Ella Falls
- Ravana Cave
- Hidden gem: 98 Acres sunset drinks

HAPUTALE / BANDARAWELA:
- Lipton's Seat
- Dambatenne Tea Factory
- Adisham Bungalow
- Haputale town viewpoint
- Hidden gem: Idalgashinna railway station

MIRISSA:
- Blue Whale watching — Nov–Apr, morning
- Mirissa Beach
- Coconut Hill
- Parrot Rock
- Weligama surfing
- Mirissa Fisheries Harbour
- Hidden gem: Secret Beach Mirissa

GALLE:
- Galle Fort
- Galle Lighthouse
- Dutch Reformed Church
- National Maritime Museum
- Jungle Beach
- Unawatuna Beach
- Koggala Lake boat tour
- Hikkaduwa coral reef
- Hidden gem: Closenberg Hotel terrace

TRINCOMALEE:
- Koneswaram Temple
- Nilaveli Beach
- Pigeon Island National Park
- Uppuveli Beach
- Fort Frederick
- Marble Beach
- Kanniya Hot Springs
- Hidden gem: Dutch Bay sunset

ARUGAM BAY:
- Main Point surfing
- Pottuvil Lagoon
- Elephant Rock
- Whiskey Point
- Crocodile Rock
- Okanda Temple
- Hidden gem: Peanut Farm Point

YALA / UDAWALAWE:
- Yala National Park
- Udawalawe Elephant Transit Home
- Udawalawe National Park
- Bundala National Park
- Hidden gem: Kataragama temple complex

JAFFNA:
- Jaffna Fort
- Nallur Kandaswamy Kovil
- Jaffna Public Library
- Casuarina Beach
- Nagadeepa Island
- Jaffna Market
- Hidden gem: Delft Island

==================================================
SUB-LOCATION TRAVEL TIMES
==================================================

Use these verified values for same-day planning.

ELLA:
Ella→Nine Arch Bridge: 3 km / 10 min
Ella→Little Adam's Peak: 2 km / 10 min
Ella→Ravana Falls: 5 km / 15 min
Ella→Ella Rock trailhead: 4 km / 15 min
Ella→Lipton's Seat: 35 km / 1.5 hrs — separate Haputale day
Ella→Adam's Peak: 90 km / 3 hrs — different region
Ella→Horton Plains: 55 km / 2 hrs — separate day

KANDY:
Kandy→Temple of Tooth: 1 km / 5 min
Kandy→Royal Botanical Gardens: 6 km / 20 min
Kandy→Kandy Lake: 1 km / 5 min
Kandy→Pinnawala: 40 km / 1.5 hrs — half-day
Kandy→Ambuluwawa: 18 km / 45 min
Kandy→Dambulla: 72 km / 2 hrs — separate day
Kandy→Sigiriya: 90 km / 2.5 hrs — separate day
Kandy→Nuwara Eliya: 80 km / 2.5 hrs — separate day

SIGIRIYA:
Sigiriya→Rock Fortress: 1 km / 5 min
Sigiriya→Dambulla: 20 km / 30 min
Sigiriya→Pidurangala: 2 km / 10 min
Sigiriya→Minneriya: 30 km / 45 min
Sigiriya→Polonnaruwa: 60 km / 1.5 hrs — separate day
Sigiriya→Anuradhapura: 75 km / 2 hrs — separate day
Sigiriya→Kandy: 90 km / 2.5 hrs — separate day

GALLE:
Galle Fort→Dutch Reformed Church: 0 km / 2 min
Galle Fort→Lighthouse: 1 km / 5 min
Galle Fort→Maritime Museum: 1 km / 5 min
Galle→Unawatuna: 5 km / 15 min
Galle→Jungle Beach: 8 km / 20 min
Galle→Koggala Lake: 13 km / 25 min
Galle→Hikkaduwa: 17 km / 30 min
Galle→Mirissa: 40 km / 1 hr — separate day
Galle→Colombo: 120 km / 2 hrs — separate day

NUWARA ELIYA:
Nuwara Eliya→Gregory Lake: 2 km / 5 min
Nuwara Eliya→Hakgala: 10 km / 20 min
Nuwara Eliya→Victoria Park: 1 km / 5 min
Nuwara Eliya→Tea factory: 5 km / 15 min
Nuwara Eliya→Horton Plains: 32 km / 1 hr — early start/full day
Nuwara Eliya→Adam's Peak: 45 km / 1.5 hrs — overnight near Hatton recommended
Nuwara Eliya→Ella: 60 km / 2.5 hrs — separate day

COLOMBO:
Colombo→Gangaramaya: 2 km / 5 min
Colombo→Galle Face: 1 km / 5 min
Colombo→National Museum: 2 km / 8 min
Colombo→Pettah: 1 km / 5 min
Colombo→Mount Lavinia: 12 km / 25 min
Colombo→Kelaniya: 11 km / 30 min
Colombo→Negombo: 35 km / 45 min — borderline/separate base
Colombo→Kandy: 115 km / 3 hrs — separate day

MIRISSA:
Mirissa Beach→Weligama: 10 km / 20 min
Mirissa→Whale watching: 0 km
Mirissa→Coconut Hill: 2 km / 10 min
Mirissa→Tangalle: 35 km / 1 hr — separate day
Mirissa→Galle: 40 km / 1 hr — separate day

ANURADHAPURA:
Anuradhapura→Sri Maha Bodhi: 2 km / 5 min
Anuradhapura→Ruwanwelisaya: 3 km / 10 min
Anuradhapura→Jetavanaramaya: 2 km / 8 min
Anuradhapura→Thuparamaya: 4 km / 12 min
Anuradhapura→Abhayagiri: 4 km / 12 min
Anuradhapura→Mihintale: 13 km / 25 min
Anuradhapura→Wilpattu: 70 km / 1.5 hrs — separate day

POLONNARUWA:
Polonnaruwa→Gal Vihara: 4 km / 10 min
Polonnaruwa→Rankoth Vehera: 3 km / 8 min
Polonnaruwa→Parakrama Samudra: 2 km / 5 min
Polonnaruwa→Museum: 1 km / 5 min
Polonnaruwa→Minneriya: 25 km / 40 min
Polonnaruwa→Sigiriya: 60 km / 1.5 hrs — separate day

TRINCOMALEE:
Trincomalee→Koneswaram: 2 km / 8 min
Trincomalee→Fort Frederick: 2 km / 8 min
Trincomalee→Kanniya: 8 km / 20 min
Trincomalee→Uppuveli: 5 km / 15 min
Trincomalee→Nilaveli: 15 km / 30 min
Trincomalee→Pigeon Island: 18 km / 35 min — morning half-day

ARUGAM BAY:
Arugam Bay→Whiskey Point: 3 km / 10 min
Arugam Bay→Pottuvil Lagoon: 3 km / 10 min
Arugam Bay→Elephant Rock: 5 km / 15 min
Arugam Bay→Crocodile Rock: 3 km / 10 min

==================================================
INTERCITY ROUTES — USE ONLY THESE VALUES
==================================================

Airport/West:
BIA/Katunayake→Negombo: 35 km / 45 min
BIA/Katunayake→Colombo: 35 km / 45 min
Negombo→Colombo: 35 km / 45 min
Negombo→Chilaw: 55 km / 1.5 hrs
Negombo→Kurunegala: 90 km / 2 hrs
Negombo→Wilpattu: 100 km / 2.5 hrs
Negombo→Kandy: 115 km / 3 hrs
Negombo→Sigiriya: 145 km / 3.5 hrs
Negombo→Anuradhapura: 165 km / 4 hrs

Cultural Triangle:
Colombo→Kandy: 115 km / 3 hrs
Colombo→Sigiriya: 175 km / 4 hrs
Colombo→Anuradhapura: 200 km / 4.5 hrs
Kandy→Dambulla: 72 km / 2 hrs
Kandy→Sigiriya: 90 km / 2.5 hrs
Dambulla→Sigiriya: 20 km / 30 min
Sigiriya→Polonnaruwa: 60 km / 1.5 hrs
Sigiriya→Anuradhapura: 75 km / 2 hrs
Anuradhapura→Polonnaruwa: 100 km / 2.5 hrs
Wilpattu→Anuradhapura: 70 km / 1.5 hrs
Chilaw→Anuradhapura: 120 km / 2.5 hrs

East:
Anuradhapura→Trincomalee: 180 km / 4 hrs
Sigiriya→Trincomalee: 120 km / 3 hrs
Polonnaruwa→Trincomalee: 100 km / 2.5 hrs
Wilpattu→Trincomalee: 250 km / 6 hrs
Trincomalee→Batticaloa: 115 km / 3 hrs
Batticaloa→Arugam Bay: 115 km / 3 hrs
Trincomalee→Arugam Bay: 230 km / 6 hrs
Arugam Bay→Yala: 120 km / 3 hrs
Arugam Bay→Colombo: 320 km / 7 hrs
Jaffna→Anuradhapura: 200 km / 4.5 hrs
Jaffna→Colombo: 395 km / 7 hrs

Hill Country:
Kandy→Nuwara Eliya: 80 km / 2.5 hrs
Kandy→Ella by car: 140 km / 5 hrs
Kandy→Ella scenic train: 140 km / 5–6 hrs
Nuwara Eliya→Ella: 60 km / 2.5 hrs
Ella→Haputale: 25 km / 45 min
Ella→Bandarawela: 20 km / 40 min
Hatton→Kandy: 55 km / 2 hrs
Hatton→Nuwara Eliya: 35 km / 1 hr

South:
Colombo→Bentota: 65 km / 1.5 hrs
Colombo→Galle: 120 km / 2 hrs
Colombo→Mirissa: 150 km / 2.5 hrs
Galle→Mirissa: 40 km / 1 hr
Mirissa→Tangalle: 35 km / 1 hr
Tangalle→Hambantota: 30 km / 45 min
Hambantota→Tissamaharama: 30 km / 45 min
Tissamaharama→Yala: 20 km / 30 min
Ella→Mirissa: 135 km / 3.5 hrs
Ella→Galle: 150 km / 4 hrs
Mirissa→Colombo: 150 km / 2.5 hrs

Long routes:
Trincomalee→Mirissa: 330 km / 7–8 hrs
Trincomalee→Colombo: 260 km / 5.5 hrs
Wilpattu→Mirissa: 280 km / 6.5 hrs

If a route is not listed, calculate it only by adding known verified legs.
Never invent a travel time.
If uncertain, add a 30-minute safety buffer.
Long cross-island routes should be treated as travel days and split when necessary.

==================================================
TRANSPORT RULES
==================================================

Every day that changes base MUST have a Getting There line using the verified
distance/time.

For intercity travel ALWAYS recommend PickMe or Uber.
Never recommend buses or public transport.

Transport wording:
- Under 50 km: "tuk-tuk or PickMe"
- 50–150 km: "PickMe or Uber (private car)"
- Over 150 km: "PickMe or Uber (private car) — book in advance"

EXCEPTION:
Kandy→Ella should use the scenic train when both cities are on the route:
140 km / 5–6 hrs.

==================================================
FOOD RULES
==================================================

Day 2 onwards normally includes:
- Breakfast
- Lunch
- Dinner
- Must-try

Day 1 follows the arrival-time rules exactly.

Use actual Sri Lankan dishes such as:
hoppers, kottu roti, pol roti, rice and curry, pol sambol,
string hoppers, fish ambul thiyal, Jaffna crab curry, wambatu moju,
wood apple juice, king coconut, etc.

Match food to the region and mention specific place types such as:
roadside kade, beach shack, hotel buffet, local restaurant, or market stall.

==================================================
GEOGRAPHIC EFFICIENCY
==================================================

Plan one logical one-way route across Sri Lanka.

Preferred route:
Airport/Negombo → Cultural Triangle → Kandy → Hill Country →
South Coast → Colombo for departure.

Do not backtrack.

Never create routes such as Kandy → Ella → Kandy.

Group nearby attractions before moving to the next region.

Stay 2–3 nights in each destination when the trip length allows it.

Always use specific Sri Lankan place names.

==================================================
BASE-STAY RULES
==================================================

Think in BASES, not individual hotel changes.

A base means one hotel in one town for 2–3 consecutive nights when the
trip length allows.

- Choose ONE hotel per base.
- Use the SAME hotel name on every day at that base.
- First day at a base: arrive, check in and explore.
- Following days at the same base: use "🏨 Staying At".
- Only a genuine check-out and move to a new town gets "🚗 Getting There".
- Do not change hotels every night.
- Day trips must remain within the applicable travel-time limits.

==================================================
ACCOMMODATION RULES
==================================================

Never say "hostel", "guesthouse", or "or similar".
Always name a real hotel.

Choose ONE hotel from the requested budget tier for each base.
Do not change the hotel within the same base.

Use only hotels from the appropriate tier below.
Never mix budget tiers unless the user explicitly asks.

BUDGET — under USD 50/night

Negombo:
The Loft Negombo · Icebear Guest House · Sea Sands Hotel Negombo · Dephani Beach Hotel

Colombo:
Clock Inn Colombo · Colombo City Hostel · The Havelock Place Bungalow · OZO Colombo

Kandy:
Hotel Casamara · The Kandy Ark · Expeditor Hotel Kandy · Hotel Topaz Kandy

Sigiriya:
Flower Inn Sigiriya · Rangiri Dambulla Resort · Sigiriya Rest · Village Inn Sigiriya

Ella:
Ella Guesthouse · Zion View Ella · Ambiente Ella · The Cove Ella

Mirissa:
Mirissa Hills · The Pelican Mirissa · Happiness Beach Inn Mirissa · Sandy's Cabanas Mirissa

Galle:
Rampart View Guesthouse · Ottery Unawatuna · Serendipity Arts Café & Hotel · One Earth Galle

Nuwara Eliya:
Collingwood Bungalow · Ashok Hotel · Garden View Hotel Nuwara Eliya · Milano Tourist Rest Nuwara Eliya

Trincomalee:
Welcome Hotel Trinco · Anand Tourist Home · Golden Beach Hotel Trinco · Sea View Hotel Trinco

Anuradhapura:
Milano Tourist Rest · Randiya Hotel · Tissawewa Grand Hotel · Lake View Hotel Anuradhapura

Wilpattu:
Lakpahana Lodge Wilpattu · Eco Team Wilpattu · Wilpattu Safari Camp · Green Village Wilpattu

Arugam Bay:
Hideaway Resort Arugam Bay · Siam View Hotel Arugam Bay · Rocco's Hotel Arugam Bay · Aloha Surf Arugam Bay

Jaffna:
Tilko City Hotel Jaffna · Morgan's Residence Jaffna · Green Grass Hotel Jaffna · Bastian Hotel Jaffna

MID-RANGE — USD 50–150/night

Negombo:
Jetwing Beach · Camelot Beach Hotel · Browns Beach Hotel Negombo · Cocobay Resort Negombo

Colombo:
Cinnamon Grand Colombo · Movenpick Hotel Colombo · Hilton Colombo Residence · Taj Samudra Colombo

Kandy:
Hotel Suisse Kandy · Thilanka Resort Kandy · Cinnamon Citadel Kandy · The Kandy House

Sigiriya:
Sigiriya Village Hotel · Water Garden Sigiriya · Aliya Resort Sigiriya · Jetwing Vil Uyana Sigiriya

Ella:
98 Acres Resort · Zion Eco Resort Ella · Ella Jungle Resort · Kelburne Mountain Villas Ella

Mirissa:
Mirissa Beach Inn · Paradise Beach Club Mirissa · The Reef Mirissa · Aditya Resort Mirissa

Galle:
Amangalla · Fort Bazaar Galle · The Fort Printers Galle · Galle Fort Hotel

Nuwara Eliya:
Grand Hotel Nuwara Eliya · Tea Bush Hotel · Heritance Tea Factory · St. Andrews Hotel Nuwara Eliya

Trincomalee:
Trinco Blu by Cinnamon · Welcombe Hotel Trincomalee · Jungle Beach by Uga Escapes · Club Oceanic Uppuveli

Anuradhapura:
Ulagalla Resort · Palm Garden Village Hotel · Tissawewa Grand Hotel · Rajarata Hotel Anuradhapura

Wilpattu:
Mahoora Wilpattu · Wild Safari Lodge Wilpattu · Chaaya Village Habarana · Cinnamon Lodge Habarana

Bentota:
Avani Bentota Resort · Vivanta Bentota · Taj Bentota Resort & Spa · Club Bentota

Tangalle:
Amanwella · Buckingham Place Tangalle · Mangrove Beach Cabanas · Insight Resort Tangalle

Arugam Bay:
Stardust Hotel Arugam Bay · Gecko's Hotel Arugam Bay · Samantha's Folly Arugam Bay · The Spice Trail

Jaffna:
Jetwing Jaffna · The Black Current Inn Jaffna · Tilko Jaffna City Hotel · Green Grass Hotel Jaffna

LUXURY — USD 150+/night

Negombo:
Jetwing Blue · Heritance Negombo · The Workroom Boutique Hotel

Colombo:
Shangri-La Colombo · Galle Face Hotel · Taj Samudra · Cinnamon Grand Colombo

Kandy:
Earls Regency Hotel · The Kandy House · Uga Ulagalla · Helga's Folly

Sigiriya:
Water Garden Sigiriya · Aliya Resort Sigiriya · Jetwing Vil Uyana · Habarana Village by Cinnamon

Ella:
98 Acres Resort & Spa · Ella Jungle Resort · Amba Estate Ella · Madulkelle Tea & Eco Lodge

Mirissa:
Mirissa Hills · Aditya Resort Mirissa · Cape Weligama · Anantara Peace Haven Tangalle

Galle:
Amangalla · The Fortress Resort & Spa · Cape Weligama · Kahanda Kanda

Nuwara Eliya:
Heritage Tea Factory · The Hill Club · Araliya Green Hills · Strathdon Hotel

Trincomalee:
Jungle Beach by Uga Escapes · Trinco Blu by Cinnamon · Uga Bay · Club Oceanic Uppuveli

Wilpattu:
Mahoora Wilpattu Tented Safari Camp · Eco Team Wilpattu · Wild Coast Tented Lodge

Bentota:
Avani Bentota Resort · Taj Bentota Resort & Spa · Centara Ceysands · Cinnamon Bey Beruwala

Tangalle:
Amanwella · Maalu Maalu Resort · Anantara Peace Haven Tangalle · Buckingham Place

Arugam Bay:
Stardust Hotel · The Spice Trail · Gecko's Hotel

Jaffna:
Jetwing Jaffna · The Black Current Inn · Tilko Jaffna · Green Grass Hotel

If the requested destination has no hotel in the selected tier, use the
nearest listed city's hotel and clearly state that it is nearby.

==================================================
FINAL QUALITY CHECK — MUST PASS BEFORE OUTPUT
==================================================

Before responding, silently verify ALL of the following:

1. Day 1 follows the exact arrival-time rules.
2. No activity or meal is scheduled before arrival.
3. Later days contain realistic Morning, Afternoon and Evening sections,
   except when a full-day activity or travel schedule makes a section
   unrealistic.
4. Meals follow the Day 1 rules and food rules.
5. Same-base days use "🏨 Staying At".
6. Base-change days use "🚗 Getting There".
7. Normal same-day activity travel does not exceed 45 minutes.
8. Full-day activities are not paired with major attractions.
9. Adam's Peak is NEVER combined with Ella activities.
10. Distances and travel times use only the verified tables.
11. Unlisted routes are calculated only from known legs; never guessed.
12. The route does not unnecessarily backtrack.
13. Hotels remain the same throughout each base.
14. Hotel selection matches the requested budget tier.
15. Every full day has region-appropriate Sri Lankan food.
16. Costs use LKR and USD, never "$".
17. Famous attractions and hidden gems are included where appropriate.
18. The exact requested output structure is followed.
19. The itinerary is physically realistic, not merely formatted correctly.

If ANY check fails, fix the itinerary internally before producing the answer.

Be enthusiastic, practical, specific, geographically logical, and realistic.
"""
CHAT_PROMPT = """
You are an expert Sri Lanka travel assistant helping with follow-up questions.

Rules:
- Do NOT rewrite the full itinerary
- Use bullet points with "-" for lists
- Keep answers short, clean, and helpful
- Each item on a NEW LINE
- Never use dollar signs — write USD instead
"""

REFINE_PROMPT = """
You are an expert Sri Lanka travel planning agent.
The user wants to REFINE their existing itinerary.
Make the requested changes while keeping the same day structure.
Format exactly the same as before with Day headers, Morning/Afternoon/Evening sections.
Never use dollar signs — write USD instead.
Always mention specific place names clearly so they can be mapped.

═══════════════════════════════════════════
GEOGRAPHIC EFFICIENCY RULES (always follow):
═══════════════════════════════════════════
- Keep the route logical and one-directional — no backtracking to already-visited regions.
- Group nearby attractions together before moving to the next region.
- Default flow: Airport/Negombo → Cultural Triangle → Kandy → Hill Country → South Coast → Colombo.
- NEVER create a route that revisits a region already completed.
- BASE-STAY RULE: keep the SAME hotel for every day spent in the same town (2-3 nights per base).
  Days that don't change town use a "🏨 Staying At: [same hotel]" opening line instead of
  "🚗 Getting There". Only the day the tourist actually moves town gets a new hotel and a
  "Getting There" line. Never swap hotels within the same base.

═══════════════════════════════════════════
TRANSPORT RULES (always follow):
═══════════════════════════════════════════
- NEVER suggest buses. Always recommend PickMe or Uber (private car) for intercity travel.
- Short trips under 50 km: "tuk-tuk or PickMe". Medium/long trips 50–150 km: "PickMe or Uber (private car)".
- Trips over 150 km: "PickMe or Uber (private car) — book in advance".
- Exception: the scenic Kandy–Ella train is a tourist highlight — always keep it.
- Use ONLY these verified distances/times (never guess):
  BIA → Negombo: 35 km / 45 min
  Negombo → Kandy: 115 km / 3 hrs
  Negombo → Sigiriya: 145 km / 3.5 hrs
  Colombo → Kandy: 115 km / 3 hrs
  Colombo → Galle: 120 km / 2 hrs via expressway
  Kandy → Sigiriya: 90 km / 2.5 hrs
  Kandy → Nuwara Eliya: 80 km / 2.5 hrs
  Kandy → Ella: 140 km / 5 hrs by car or 5–6 hrs by scenic train
  Nuwara Eliya → Ella: 60 km / 2.5 hrs
  Ella → Mirissa: 135 km / 3.5 hrs
  Ella → Galle: 150 km / 4 hrs
  Galle → Mirissa: 40 km / 1 hr
  Mirissa → Colombo: 150 km / 2.5 hrs
  Sigiriya → Polonnaruwa: 60 km / 1.5 hrs
  Sigiriya → Trincomalee: 120 km / 3 hrs
  Trincomalee → Mirissa: 330 km / 7–8 hrs ⚠ NEVER say 4–5 hrs — this is WRONG
  Trincomalee → Colombo: 260 km / 5.5 hrs
  Wilpattu → Trincomalee: 250 km / 6 hrs
  Wilpattu → Mirissa: 280 km / 6.5 hrs
  Arugam Bay → Yala: 120 km / 3 hrs
  Jaffna → Anuradhapura: 200 km / 4.5 hrs

═══════════════════════════════════════════
INTRA-DAY FEASIBILITY RULES (always follow when refining):
═══════════════════════════════════════════
- The travel time between any two activities on the SAME day must not exceed 45 minutes.
- Full-day hikes (Adam's Peak, Ella Rock, Horton Plains, Knuckles, Sinharaja) consume the entire day.
  Never pair a full-day hike with another major attraction on the same day.
- Adam's Peak (Sri Pada) is near Hatton — NOT near Ella. Ella → Adam's Peak = 90 km / 3+ hrs.
  Never place Adam's Peak and Ella activities on the same day.
- "Little Adam's Peak" is a short 2-hr hike inside Ella town. It is a completely different place.
- Before confirming any refined day, verify: can the tourist realistically travel between all
  activities within the day using the Sub-Location Travel Time Table?

SUB-LOCATION TRAVEL TIME TABLE (use for intra-day checks):
  Ella → Nine Arch Bridge: 3 km / 10 min
  Ella → Little Adam's Peak: 2 km / 10 min
  Ella → Ravana Falls: 5 km / 15 min
  Ella → Ella Rock trailhead: 4 km / 15 min [full-day hike]
  Ella → Adam's Peak: 90 km / 3+ hrs [DIFFERENT REGION]
  Ella → Horton Plains: 55 km / 2 hrs [early morning only]
  Kandy → Royal Botanical Gardens: 6 km / 20 min
  Kandy → Pinnawala: 40 km / 1.5 hrs [half-day]
  Kandy → Ambuluwawa: 18 km / 45 min
  Sigiriya → Dambulla: 20 km / 30 min
  Sigiriya → Pidurangala: 2 km / 10 min
  Sigiriya → Minneriya: 30 km / 45 min
  Galle → Unawatuna: 5 km / 15 min
  Galle → Jungle Beach: 8 km / 20 min
  Galle → Hikkaduwa: 17 km / 30 min
  Nuwara Eliya → Horton Plains: 32 km / 1 hr [full-day, early start]
  Polonnaruwa → Minneriya: 25 km / 40 min
  Anuradhapura → Mihintale: 13 km / 25 min
  Trincomalee → Nilaveli Beach: 15 km / 30 min
  Trincomalee → Pigeon Island: 18 km / 35 min [boat trip, morning only]
  Trincomalee → Kanniya Hot Springs: 8 km / 20 min
  Arugam Bay → Whiskey Point: 3 km / 10 min
  Arugam Bay → Pottuvil Lagoon: 3 km / 10 min
  Arugam Bay → Elephant Rock: 5 km / 15 min

═══════════════════════════════════════════
ARRIVAL DAY RULES (always follow):
═══════════════════════════════════════════
- Respect the original arrival time when refining Day 1.
- Day 1 sections must match the arrival time exactly:
  * Morning arrival → Morning + Afternoon + Evening
  * Afternoon arrival → Afternoon + Evening only (no Morning)
  * Evening arrival → Evening only (no Morning or Afternoon)
  * Night arrival → Night only (no Morning, Afternoon, or Evening)
- Never add time-of-day sections that occur before the tourist has landed.

═══════════════════════════════════════════
ACCOMMODATION RULES (always follow):
═══════════════════════════════════════════
- NEVER say "hostel", "guesthouse", or "or similar". Always name real specific hotels.
- Always recommend 3–4 hotels per destination matching the tourist's original budget tier.
- Always write the hotel name in the Cost section: e.g. "Accommodation: LKR 20,000 at 98 Acres Resort"
- Use ONLY verified hotels from the list below — never invent hotel names.

  BUDGET hotels per destination:
  Negombo: The Loft Negombo · Icebear Guest House · Sea Sands Hotel Negombo · Dephani Beach Hotel
  Colombo: Clock Inn Colombo · Colombo City Hostel · The Havelock Place Bungalow · OZO Colombo
  Kandy: Hotel Casamara · The Kandy Ark · Expeditor Hotel Kandy · Hotel Topaz Kandy
  Sigiriya: Flower Inn Sigiriya · Rangiri Dambulla Resort · Sigiriya Rest · Village Inn Sigiriya
  Ella: Ella Guesthouse · Zion View Ella · Ambiente Ella · The Cove Ella
  Mirissa: Mirissa Hills · The Pelican Mirissa · Happiness Beach Inn Mirissa · Sandy's Cabanas Mirissa
  Galle: Rampart View Guesthouse · Ottery Unawatuna · Serendipity Arts Café & Hotel · One Earth Galle
  Nuwara Eliya: Collingwood Bungalow · Ashok Hotel · Garden View Hotel Nuwara Eliya · Milano Tourist Rest Nuwara Eliya
  Trincomalee: Welcome Hotel Trinco · Anand Tourist Home · Golden Beach Hotel Trinco · Sea View Hotel Trinco
  Anuradhapura: Milano Tourist Rest · Randiya Hotel · Tissawewa Grand Hotel (budget wing) · Lake View Hotel
  Arugam Bay: Hideaway Resort · Siam View Hotel · Rocco's Hotel · Aloha Surf
  Jaffna: Tilko City Hotel · Morgan's Residence · Green Grass Hotel · Bastian Hotel

  MID-RANGE hotels per destination:
  Negombo: Jetwing Beach · Camelot Beach Hotel · Browns Beach Hotel · Cocobay Resort
  Colombo: Cinnamon Grand · Movenpick Hotel Colombo · Hilton Colombo Residence · Taj Samudra
  Kandy: Hotel Suisse Kandy · Thilanka Resort · Cinnamon Citadel Kandy · The Kandy House
  Sigiriya: Sigiriya Village Hotel · Water Garden Sigiriya · Aliya Resort · Jetwing Vil Uyana
  Ella: 98 Acres Resort · Zion Eco Resort · Ella Jungle Resort · Kelburne Mountain Villas
  Mirissa: Mirissa Beach Inn · Paradise Beach Club · The Reef Mirissa · Aditya Resort Mirissa
  Galle: Amangalla · Fort Bazaar Galle · The Fort Printers · Galle Fort Hotel
  Nuwara Eliya: Grand Hotel Nuwara Eliya · Tea Bush Hotel · Heritance Tea Factory · St. Andrews Hotel
  Trincomalee: Trinco Blu by Cinnamon · Welcombe Hotel · Club Oceanic Uppuveli · Jungle Beach by Uga Escapes
  Anuradhapura: Ulagalla Resort · Palm Garden Village Hotel · Tissawewa Grand Hotel · Rajarata Hotel
  Arugam Bay: Stardust Hotel · Gecko's Hotel · Samantha's Folly · The Spice Trail
  Jaffna: Jetwing Jaffna · The Black Current Inn · Tilko Jaffna City Hotel · Green Grass Hotel

  LUXURY hotels per destination:
  Negombo: Jetwing Blue · Heritance Negombo · The Workroom Boutique Hotel · Browns Beach Hotel (luxury wing)
  Colombo: Shangri-La Colombo · Galle Face Hotel · Taj Samudra Colombo · Cinnamon Grand Colombo
  Kandy: Earls Regency Hotel · The Kandy House · Helga's Folly Kandy · Uga Ulagalla (nearby)
  Sigiriya: Water Garden Sigiriya · Aliya Resort · Jetwing Vil Uyana · Habarana Village by Cinnamon
  Ella: 98 Acres Resort & Spa · Ella Jungle Resort · Amba Estate Ella · Madulkelle Tea & Eco Lodge
  Mirissa: Anantara Peace Haven Tangalle · Mirissa Hills · Aditya Resort · Cape Weligama
  Galle: Amangalla · The Fortress Resort & Spa · Cape Weligama · Kahanda Kanda
  Nuwara Eliya: Heritance Tea Factory · The Hill Club Nuwara Eliya · Araliya Green Hills · Strathdon Hotel
  Trincomalee: Jungle Beach by Uga Escapes · Trinco Blu by Cinnamon · Uga Bay Trincomalee · Club Oceanic Uppuveli
  Arugam Bay: Stardust Hotel · The Spice Trail · Gecko's Hotel · Samantha's Folly (best local luxury)
  Jaffna: Jetwing Jaffna · The Black Current Inn · Tilko Jaffna City Hotel
"""


# ── Weather Tool ──────────────────────────────────────────────────────────────
def _cache_data(**kwargs):
    """
    Wraps st.cache_data so this module still works when Streamlit isn't
    installed (e.g. quick script testing) — falls back to a no-op
    passthrough decorator in that case instead of crashing on import.
    """
    if STREAMLIT_OK:
        return st.cache_data(**kwargs)
    return lambda fn: fn


@_cache_data(ttl=1800, show_spinner=False)
def get_weather(city: str = "Colombo") -> dict:
    if not WEATHER_API_KEY:
        return {"success": False, "error": "Weather API key not configured"}
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        if response.status_code == 200:
            return {
                "city":        data["name"],
                "country":     data["sys"]["country"],
                "temp":        round(data["main"]["temp"]),
                "feels_like":  round(data["main"]["feels_like"]),
                "description": data["weather"][0]["description"].title(),
                "humidity":    data["main"]["humidity"],
                "wind":        round(data["wind"]["speed"], 1),
                "icon":        data["weather"][0]["main"],
                "success":     True,
            }
        return {"success": False, "error": data.get("message", "City not found")}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Reasoning helpers ─────────────────────────────────────────────────────────

# Base Day-1 plan per arrival time slot — same wording as before, just pulled
# into a lookup table instead of an if/elif chain so decide_arrival_context()
# can layer an energy preference on top of it.
_ARRIVAL_BASE = {
    "morning": {
        "can_travel_far": True,
        "day1_instruction": (
            "MORNING ARRIVAL — Day 1 has: Morning ✅  Afternoon ✅  Evening ✅\n"
            "Tourist lands before noon and clears customs by ~10am. Full day available.\n"
            "Start Day 1 with the Morning section. Travel DIRECTLY from the airport to the "
            "first real destination (Sigiriya, Kandy, Galle, etc.). Do NOT stop in Negombo.\n"
            "Include full Breakfast + Lunch + Dinner in Food Today.\n"
            "Day 1 Morning: Arrive BIA, clear customs, transfer to [destination].\n"
            "Day 1 Afternoon: Sightseeing at [destination].\n"
            "Day 1 Evening: Dinner and evening activities."
        ),
    },
    "afternoon": {
        "can_travel_far": False,
        "day1_instruction": (
            "AFTERNOON ARRIVAL — Day 1 has: Morning ❌  Afternoon ✅  Evening ✅\n"
            "Tourist lands between 12:00–18:00 and clears customs by ~3–5pm.\n"
            "DO NOT write a Morning section for Day 1 — the tourist is on a plane.\n"
            "Start Day 1 directly with the Afternoon section.\n"
            "Day 1 Afternoon: Land at BIA, clear customs, transfer to Negombo (35 km, ~45 min). "
            "Check in to hotel. Visit Negombo Fish Market. Walk along the beach. St. Mary's Church.\n"
            "Day 1 Evening: Seafood dinner at a beachfront restaurant. Evening walk on Negombo beach. Rest.\n"
            "Food Today: NO Breakfast. Include Lunch (light — airport snack or en route) and Dinner only.\n"
            "Day 2 begins the main journey from Negombo."
        ),
    },
    "evening": {
        "can_travel_far": False,
        "day1_instruction": (
            "EVENING ARRIVAL — Day 1 has: Morning ❌  Afternoon ❌  Evening ✅\n"
            "Tourist lands between 18:00–22:00 and clears customs by ~8–10pm.\n"
            "DO NOT write a Morning section for Day 1.\n"
            "DO NOT write an Afternoon section for Day 1.\n"
            "Start Day 1 directly and ONLY with the Evening section.\n"
            "Day 1 Evening: Land at BIA around [time]. Clear customs. Transfer to Negombo "
            "(35 km, ~45 min by taxi). Check in to hotel. Freshen up. Light dinner at a nearby "
            "beachside restaurant. Short stroll if energy allows. Early night.\n"
            "Food Today: NO Breakfast. NO Lunch. Dinner only (light meal near hotel).\n"
            "Day 2 begins the real journey from Negombo with a full Morning + Afternoon + Evening."
        ),
    },
    "night": {
        "can_travel_far": False,
        "day1_instruction": (
            "NIGHT ARRIVAL — Day 1 has: Morning ❌  Afternoon ❌  Evening ❌  Night ✅\n"
            "Tourist lands after 22:00. After customs it is midnight or later.\n"
            "DO NOT write a Morning section for Day 1.\n"
            "DO NOT write an Afternoon section for Day 1.\n"
            "DO NOT write an Evening section for Day 1.\n"
            "Use ONLY a Night section (heading: **Night:**) for Day 1. Keep it very short.\n"
            "Day 1 Night: Land at BIA after 10pm. Clear customs (~1 hr). Transfer to Negombo "
            "or a hotel near Katunayake airport (10–15 min). Check in. Sleep.\n"
            "NO Food Today section for Day 1 — tourist just wants to sleep.\n"
            "Day 2 is the real start with full Morning + Afternoon + Evening and all meals."
        ),
    },
}

# Energy preference options shown as a follow-up dropdown once an arrival time
# is picked — separate from arrival TIME because a person can land at 9am and
# still be jet-lagged, or land at 6pm and still want to get moving. app.py
# imports this dict directly to render the dropdown.
ENERGY_OPTIONS = {
    "go": {
        "label": "Start right away",
        "desc": "Full energy — head straight into sightseeing today",
    },
    "ease": {
        "label": "Ease in gently",
        "desc": "Settle in first, then just one light activity today",
    },
    "rest": {
        "label": "Full rest day",
        "desc": "Just recover today — the real trip starts tomorrow",
    },
}


def get_energy_options(arrival_time: str) -> dict:
    """
    Filters ENERGY_OPTIONS down to the choices that actually mean something for
    a given arrival time, so app.py's dropdown doesn't offer a plan it can't
    deliver.

    - Morning: full day available, so "go" / "ease" / "rest" are all genuinely
      different plans. Show all three.
    - Afternoon / Evening: not enough daylight left to "start right away" in
      any meaningful sense, so we drop "go" and only offer "ease" / "rest".
    - Night: decide_arrival_context() already collapses everything to the same
      transfer-and-sleep plan regardless of energy preference — there is no
      real choice here, so just return "rest" on its own.
    """
    t = arrival_time.lower().strip()
    if t == "morning":
        return dict(ENERGY_OPTIONS)
    if t in ("afternoon", "evening"):
        return {k: v for k, v in ENERGY_OPTIONS.items() if k != "go"}
    return {"rest": ENERGY_OPTIONS["rest"]}


def decide_arrival_context(arrival_time: str, energy: str = "go") -> dict:
    """
    Translate arrival time + how ready the traveler is to dive in into a
    Day 1 planning instruction.

    arrival_time: "morning" | "afternoon" | "evening" | "night"
    energy:       "go"   — start right away (default, same behaviour as before)
                  "ease" — settle in, then at most one light activity
                  "rest" — Day 1 is recovery only, regardless of arrival time
    """
    t = arrival_time.lower().strip()
    e = energy.lower().strip()
    base = _ARRIVAL_BASE.get(t, _ARRIVAL_BASE["night"])

    if e == "rest":
        # Full rest overrides arrival time entirely — however much daylight is
        # technically left, Day 1 stays free of scheduled activities.
        return {
            "can_travel_far": False,
            "day1_instruction": (
                f"FULL REST DAY REQUESTED — the traveler lands in the {t}, but has asked for "
                "Day 1 to be pure recovery, not sightseeing, regardless of how much daylight is "
                "technically left after landing.\n"
                "Day 1: Arrive BIA, clear customs, transfer directly to a comfortable hotel near "
                "Negombo or the airport. Check in. NO scheduled sightseeing or activities today — "
                "suggest only relaxing at the hotel, an unhurried walk if they feel like it, and an "
                "early rest. Keep Food Today realistic for whichever meals actually fall after "
                "their arrival time — don't force a full breakfast+lunch+dinner if they land later "
                "in the day.\n"
                "Day 2 begins the real itinerary with a full Morning + Afternoon + Evening."
            ),
        }

    if e == "ease" and t in ("morning", "afternoon"):
        # These are the two slots where the traveler technically COULD do more today,
        # so "ease in" meaningfully changes the plan. Evening/night arrivals are already
        # light by default in _ARRIVAL_BASE, so there's nothing further to ease off.
        eased = dict(base)
        eased["can_travel_far"] = False
        eased["day1_instruction"] = base["day1_instruction"] + (
            "\n\nEASE-IN REQUESTED: even though more time is technically available today, the "
            "traveler wants to start gently rather than diving straight in. Limit Day 1 to at "
            "most ONE light activity (a short walk, a viewpoint, a local market, a relaxed café) "
            "— do NOT schedule the trip's main highlight or big attraction today. Save the primary "
            "sightseeing for Day 2."
        )
        return eased

    return base



def decide_travel_style(interests: list, budget: str) -> dict:
    if "Luxury" in budget:
        stay = "5-star resorts and luxury boutique hotels with private transport"
        cost_level = "high"
    elif "Budget" in budget:
        stay = "budget-friendly hotels and guesthouses with PickMe/tuk-tuk transport"
        cost_level = "low"
    else:
        stay = "comfortable mid-range hotels and PickMe/Uber transport"
        cost_level = "medium"

    if "Hiking" in interests:
        pace = "active and fast-paced"
    elif "Relaxation" in interests:
        pace = "slow and relaxing"
    else:
        pace = "balanced"

    focus = []
    if "Beaches" in interests:            focus.append("coastal areas")
    if "History & Culture" in interests:  focus.append("ancient cities")
    if "Wildlife" in interests:           focus.append("national parks")
    if "Hiking" in interests:             focus.append("hill country")
    if "Food & Cuisine" in interests:     focus.append("local food hotspots")

    return {
        "stay":       stay,
        "pace":       pace,
        "cost_level": cost_level,
        "focus":      ", ".join(focus) if focus else "general sightseeing",
    }


def check_goal_achievement(itinerary: str) -> dict:
    checks = {
        "Day structure":        "Day 1" in itinerary,
        "Cost estimates":       "USD" in itinerary or "LKR" in itinerary,
        "Food recommendations": "Food Today" in itinerary or "food" in itinerary.lower(),
        "Travel tips":          "Travel Tips" in itinerary or "Tips" in itinerary,
        "Time sections":        "Morning" in itinerary and "Evening" in itinerary,
    }
    passed = sum(checks.values())
    total  = len(checks)

    if passed == total:
        status = "complete"
        label  = f"Goal achieved — all {total}/{total} criteria met"
    elif passed >= 3:
        status = "partial"
        label  = f"Mostly complete — {passed}/{total} criteria met"
    else:
        status = "incomplete"
        label  = f"Incomplete — only {passed}/{total} criteria met"

    return {"checks": checks, "passed": passed, "total": total,
            "status": status, "label": label}


def clean_text(text: str) -> str:
    text = text.replace("$", "USD ").replace("**", "")
    text = _dedupe_repeated_day_headers(text)
    return text


def _dedupe_repeated_day_headers(text: str) -> str:
    """
    The model occasionally writes a day's title twice in a row — once as the
    intended header, once as a near-duplicate plain line right after it, e.g.:
        Day 2: Dambulla and Cave Temple
        Day 2: Dambulla Cave Temple and Local Exploration
    This is a formatting slip, not a real second day, so if two consecutive
    non-blank "Day N:" lines share the same day number, the second is dropped.
    A prompt fix alone can't guarantee this never happens again, so this acts
    as a safety net regardless of how well the prompt is followed.
    """
    lines = text.split("\n")
    day_re = re.compile(r'^#{0,3}\s*Day\s+(\d+)\s*:', re.I)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = day_re.match(line.strip())
        if match:
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                next_match = day_re.match(lines[j].strip())
                if next_match and next_match.group(1) == match.group(1):
                    out.append(line)
                    i = j + 1  # skip the blank gap and the duplicate line
                    continue
        out.append(line)
        i += 1
    return "\n".join(out)


def extract_place_names(itinerary: str) -> list:
    """
    Extract only place names that appear as actual destinations in the itinerary.
    """
    known_places = [
        "Colombo", "Kandy", "Galle", "Sigiriya", "Ella", "Nuwara Eliya",
        "Mirissa", "Unawatuna", "Trincomalee", "Anuradhapura", "Polonnaruwa",
        "Dambulla", "Negombo", "Hikkaduwa", "Arugam Bay", "Yala", "Udawalawe",
        "Horton Plains", "Adam's Peak", "Pinnawala", "Bentota", "Matara",
        "Jaffna", "Batticaloa", "Haputale", "Nanu Oya", "Badulla", "Ratnapura",
        "Tangalle", "Weligama", "Koggala", "Ahungalla", "Beruwala", "Kalpitiya",
        "Wilpattu", "Minneriya", "Knuckles", "Kitulgala", "Sinharaja",
        "Hatton", "Talawakele", "Bandarawela", "Welimada", "Mahiyanganaya",
        "Katunayake", "Chilaw", "Puttalam", "Kurunegala", "Hambantota",
        "Tissamaharama", "Weeraketiya", "Dickwella", "Ambalangoda",
        "Aluthgama", "Kalutara", "Panadura", "Moratuwa", "Nugegoda",
        "Ampara", "Monaragala", "Wellawaya", "Embilipitiya", "Ratnapura",
        "Avissawella", "Nuwara Eliya", "Pelmadulla", "Balangoda",
        "Nilaveli", "Uppuveli", "Pigeon Island", "Koneswaram",
        "Mihintale", "Wilpattu", "Bundala", "Kanniya",
    ]

    FOOD_LINE_RE = re.compile(
        r'(breakfast|lunch|dinner|must.?try|food today|🍽)',
        re.I,
    )

    FOOD_WORD_RE = re.compile(
        r'\b(curry|crab|prawn|fish|seafood|sambol|roti|hopper|juice|cake|'
        r'rice|dish|recipe|cuisine|snack|stall|eatery|restaurant|buffet|meal|drink)\b',
        re.I,
    )

    found = []
    lines = itinerary.split('\n')

    for place in known_places:
        place_re = re.compile(r'\b' + re.escape(place) + r'\b', re.I)

        for line in lines:
            clean = re.sub(r'\*+', '', line).strip()
            lower = clean.lower()

            if FOOD_LINE_RE.search(lower):
                continue

            check = clean

            # Fix: "Galle Face Green" is in Colombo — don't map it as the city of Galle
            if place == "Galle":
                check = re.sub(r"\bGalle\s+Face\b", "", clean, flags=re.I)
                if not place_re.search(check):
                    continue

            if place == "Adam's Peak":
                # Strip "Little Adam's Peak" first
                check = re.sub(r"\bLittle\s+Adam's\s+Peak\b", "", clean, flags=re.I)
                # Strip comparison references like "compared to Adam's Peak", "easier than Adam's Peak"
                check = re.sub(r'\b(compared to|than|unlike|vs\.?|versus|easier than|harder than)\s+Adam\'s\s+Peak\b', "", check, flags=re.I)
                # Strip "version of / alternative to Adam's Peak" phrases
                check = re.sub(r'\b(version of|alternative to|substitute for)\s+Adam\'s\s+Peak\b', "", check, flags=re.I)
                if not place_re.search(check):
                    continue

            if place_re.search(check):
                if place not in found:
                    found.append(place)
                break

    return found


# ── Place coordinates ─────────────────────────────────────────────────────────
# ── Seasonal / Climate / Festival Reference Data ──────────────────────────────
# This is real structured data (not something left to the LLM to guess), so
# "best time to visit" and "which festivals overlap your trip" answers are
# grounded in facts rather than invented by the model. Sri Lanka has two
# monsoon systems that hit opposite coasts at opposite times of year, which is
# the single most important thing for a first-time visitor to know.
SRI_LANKA_MONTHLY_GUIDE = {
    "January":   {"good_for": ["South Coast", "West Coast", "Hill Country", "Cultural Triangle"],
                   "avoid": ["East Coast"],
                   "note": "Peak season on the south/west coasts and hill country — dry, sunny, but busiest and most expensive time to visit."},
    "February":  {"good_for": ["South Coast", "West Coast", "Hill Country", "Cultural Triangle"],
                   "avoid": ["East Coast"],
                   "note": "Still peak season south/west — excellent weather, book hotels early."},
    "March":     {"good_for": ["South Coast", "West Coast", "Cultural Triangle"],
                   "avoid": ["East Coast (getting hot)"],
                   "note": "Good weather continues on the south/west coast; starting to get hot and humid inland."},
    "April":     {"good_for": ["Cultural Triangle", "East Coast (opening up)"],
                   "avoid": ["South Coast (rain starting)"],
                   "note": "Sinhala & Tamil New Year month (mid-April) — a major national holiday, expect closures and crowded transport. South-west monsoon starts to build."},
    "May":       {"good_for": ["East Coast", "Cultural Triangle"],
                   "avoid": ["South Coast", "West Coast", "Hill Country"],
                   "note": "Yala monsoon hits the south-west coast and hill country hard. East coast (Trincomalee, Arugam Bay) is drying out and starting its season."},
    "June":      {"good_for": ["East Coast"],
                   "avoid": ["South Coast", "West Coast", "Hill Country"],
                   "note": "Wettest period for the south-west; this is peak season for the east coast instead."},
    "July":      {"good_for": ["East Coast", "Cultural Triangle"],
                   "avoid": ["South Coast", "West Coast"],
                   "note": "Esala Perahera in Kandy usually falls in July/August (exact date changes yearly — verify closer to travel) — spectacular but hotels in Kandy book out."},
    "August":    {"good_for": ["East Coast", "Cultural Triangle", "Hill Country"],
                   "avoid": ["South Coast", "West Coast (still wet)"],
                   "note": "Esala Perahera season continues; elephant gathering season begins at Minneriya/Kaudulla (Jul-Oct)."},
    "September": {"good_for": ["East Coast", "Cultural Triangle"],
                   "avoid": ["South Coast (transitioning)"],
                   "note": "Inter-monsoon month — weather is changeable island-wide, but a good shoulder-season window for the Cultural Triangle."},
    "October":   {"good_for": ["Cultural Triangle"],
                   "avoid": ["East Coast (monsoon starting)", "South Coast (still unsettled)"],
                   "note": "Maha monsoon starts building on the east coast; inter-monsoon rain is possible island-wide."},
    "November":  {"good_for": ["Cultural Triangle (early season)"],
                   "avoid": ["East Coast", "South Coast (still settling)"],
                   "note": "East coast monsoon underway. South-west coast starts drying out toward the end of the month."},
    "December":  {"good_for": ["South Coast", "West Coast", "Hill Country", "Cultural Triangle"],
                   "avoid": ["East Coast"],
                   "note": "Peak season begins on the south/west coast again. Christmas/New Year is the busiest and most expensive week of the year — book far ahead."},
}

# Fixed-date public holidays (same date every year) plus well-known annual
# events whose exact date moves slightly year to year (flagged as such).
SRI_LANKA_FIXED_EVENTS = [
    {"date": "January 14/15", "name": "Thai Pongal", "note": "Tamil harvest festival, mainly north/east."},
    {"date": "February 4",    "name": "Independence Day", "note": "National public holiday, ceremonies in Colombo."},
    {"date": "April 13-14",   "name": "Sinhala & Tamil New Year", "note": "The biggest holiday of the year — most businesses close for several days, transport gets packed. Great cultural experience, harder logistics."},
    {"date": "May (full moon)", "name": "Vesak Poya", "note": "Buddha's birth/enlightenment/death — lanterns and free food stalls (dansal) island-wide, especially Colombo. Alcohol sales banned that day."},
    {"date": "July/August (dates vary yearly)", "name": "Esala Perahera (Kandy)", "note": "Ten-night procession with elephants, dancers, drummers — one of Asia's great festivals. Kandy hotels sell out weeks ahead."},
    {"date": "December 25", "name": "Christmas", "note": "Public holiday, decorations across Colombo and the west coast."},
]


def get_seasonal_context(travel_month: str | None) -> str:
    """
    Returns a short block of factual seasonal/event guidance for the given
    month, meant to be injected into the LLM prompt. This is real reference
    data, not something the model is asked to invent, so 'best time to
    visit' / 'what's on' claims stay grounded.
    """
    if not travel_month or travel_month not in SRI_LANKA_MONTHLY_GUIDE:
        return ""

    guide = SRI_LANKA_MONTHLY_GUIDE[travel_month]
    lines = [f"Seasonal context for travelling in {travel_month} (use this, don't guess):"]
    lines.append(f"  - Good weather in: {', '.join(guide['good_for'])}")
    lines.append(f"  - Avoid / expect rain in: {', '.join(guide['avoid'])}")
    lines.append(f"  - Note: {guide['note']}")

    relevant_events = [e for e in SRI_LANKA_FIXED_EVENTS if travel_month[:3] in e["date"]]
    if relevant_events:
        lines.append("  - Events that may fall in this month:")
        for e in relevant_events:
            lines.append(f"      • {e['name']} ({e['date']}) — {e['note']}")

    lines.append(
        "  If the itinerary passes through a region flagged 'avoid' above, mention the "
        "weather trade-off briefly and suggest an indoor/alternative activity for that day."
    )
    return "\n".join(lines)


# ── Sustainability / Local-Impact Tags ────────────────────────────────────────
# Curated, not guessed: hotel groups/properties publicly known for
# sustainability certification (Green Globe, LEED, etc.) or for being
# community-owned/operated. Used to badge the itinerary after generation.
SUSTAINABILITY_TAGS = {
    "Jetwing Vil Uyana":        "🌱 Eco-certified (Green Globe)",
    "Jetwing Beach":            "🌱 Eco-conscious group (Jetwing)",
    "Jetwing Blue":             "🌱 Eco-conscious group (Jetwing)",
    "Jetwing Jaffna":           "🌱 Eco-conscious group (Jetwing)",
    "Uga Ulagalla":             "🌱 Eco-certified (Uga Escapes)",
    "Ulagalla Resort":          "🌱 Eco-certified (Ulagalla)",
    "Jungle Beach by Uga Escapes": "🌱 Eco-certified (Uga Escapes)",
    "Heritance Tea Factory":    "🌱 Adaptive reuse of a historic tea factory",
    "Water Garden Sigiriya":    "🌱 Eco-designed resort",
    "Kelburne Mountain Villas Ella": "🤝 Community-linked estate stay",
    "Amba Estate Ella":         "🤝 Community-run tea estate",
    "Madulkelle Tea & Eco Lodge": "🌱 Eco-lodge",
    "Zion Eco Resort Ella":     "🌱 Eco-resort",
    "Eco Team Wilpattu":        "🌱 Eco-certified operator",
    "Sinharaja":                "🤝 Community-guided rainforest visits support local trackers",
}


def annotate_sustainability(itinerary_text: str) -> tuple[str, list]:
    """
    Scans the finished itinerary text for any hotel/place names in
    SUSTAINABILITY_TAGS and appends a small badge next to each mention.
    Returns (annotated_text, list_of_matched_badges) so the UI can also
    show a summary strip without re-scanning.
    """
    annotated = itinerary_text
    matched = []
    for name, badge in SUSTAINABILITY_TAGS.items():
        pattern = re.compile(r'\b' + re.escape(name) + r'\b')
        if pattern.search(annotated) and name not in [m[0] for m in matched]:
            matched.append((name, badge))
            # Only badge the first occurrence to avoid cluttering every mention.
            annotated = pattern.sub(f"{name} ({badge})", annotated, count=1)
    return annotated, matched


# ── Packing List Generator ────────────────────────────────────────────────────
# Rule-based, not LLM-generated — deterministic and always internally
# consistent with the season/interest data above.
_PACKING_BASE = [
    "Lightweight breathable clothing (cotton/linen)", "Rain jacket or compact umbrella",
    "Reef-safe sunscreen (SPF 50+)", "Sunglasses & sun hat", "Basic first-aid kit + rehydration salts",
    "Power adapter (Type D/G/M, 230V)", "Reusable water bottle",
    "Modest clothing for temple visits (shoulders/knees covered)",
]
_PACKING_BY_INTEREST = {
    "Beaches":            ["Swimwear", "Quick-dry towel", "Waterproof phone pouch", "Flip-flops"],
    "Hiking":             ["Sturdy trekking shoes", "Headlamp/torch (for pre-dawn hikes)", "Light backpack", "Leech socks (hill country/rainforest trails)"],
    "Nature":             ["Insect repellent", "Light long sleeves for dusk walks", "Compact rain cover for bags"],
    "Photography":        ["Extra memory cards/battery", "Dry bag/silica packets for humidity", "Lens cloth"],
    "Wildlife":           ["Binoculars", "Neutral/earth-toned clothing for safaris", "Camera with zoom lens"],
    "History & Culture":  ["Modest white/light clothing for temple visits", "Small cash for temple donations/shoe storage"],
    "Food & Cuisine":     ["Antacids/digestive aid (spicy food)", "Reusable cutlery for street food"],
    "Relaxation":         ["Light cover-up for spa/pool areas", "A good book"],
}
_PACKING_RAINY_SEASON = ["Dry bag for electronics", "Extra pair of quick-dry shoes", "Waterproof bag cover"]


def generate_packing_list(interests: list, travel_month: str | None = None) -> dict:
    """
    Returns {"essentials": [...], "for_your_trip": [...], "seasonal": [...]}
    built deterministically from the season data and selected interests —
    no AI call needed, so it's instant and always consistent.
    """
    for_your_trip = []
    for interest in interests:
        for_your_trip.extend(_PACKING_BY_INTEREST.get(interest, []))

    seasonal = []
    if travel_month and travel_month in SRI_LANKA_MONTHLY_GUIDE:
        guide = SRI_LANKA_MONTHLY_GUIDE[travel_month]
        if guide["avoid"]:
            seasonal = list(_PACKING_RAINY_SEASON)

    # Dedupe while preserving order.
    def _dedupe(items):
        seen, out = set(), []
        for i in items:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    return {
        "essentials":   _PACKING_BASE,
        "for_your_trip": _dedupe(for_your_trip),
        "seasonal":     seasonal,
    }


# ── Essential Phrases ──────────────────────────────────────────────────────
ESSENTIAL_PHRASES = [
    {"english": "Thank you",           "sinhala": "Bohoma Sthuthi",     "tamil": "Nandri"},
    {"english": "Hello",               "sinhala": "Ayubowan",           "tamil": "Vanakkam"},
    {"english": "How much?",           "sinhala": "Keeyada?",           "tamil": "Evvalavu?"},
    {"english": "Delicious",           "sinhala": "Rasai",              "tamil": "Suvai"},
    {"english": "Where is...?",        "sinhala": "... koheda?",        "tamil": "... enge irukku?"},
    {"english": "Too expensive",       "sinhala": "Ganan wadi",         "tamil": "Adhigam vilai"},
]

# ── Safety & Emergency Info ────────────────────────────────────────────────
EMERGENCY_CONTACTS = [
    {"label": "Tourist Police Hotline", "value": "1912"},
    {"label": "General Emergency (Police/Ambulance/Fire)", "value": "119 / 1990 (ambulance)"},
    {"label": "Tourist Board Info Line", "value": "1912 / +94 11 2426900"},
]

COMMON_TOURIST_SCAMS = [
    "Tuk-tuk 'meter is broken' — agree on a price before getting in, or use PickMe/Uber.",
    "'The temple/attraction is closed today' from a stranger who then offers to take you somewhere else — most major sites don't close unannounced; verify with your hotel.",
    "Gem scams in Colombo/Beruwala — never buy gems from a stranger who 'happens' to strike up conversation.",
    "Inflated 'government shop' claims for handicrafts — genuine government Laksala shops exist, but random guides steering you to a specific shop for commission is common.",
    "Fake 'orphanage' or 'elephant camp' visits that support animal cruelty — research operators before booking any elephant experience.",
]

VISA_NOTE = (
    "Most nationalities need an Electronic Travel Authorization (ETA) before arrival — "
    "apply only at the official site (eta.gov.lk). Verify current requirements for your "
    "nationality before travel, as policy changes periodically."
)


@_cache_data(ttl=1800, show_spinner=False)
def get_exchange_rates(base: str = "USD") -> dict:
    """
    Fetches live LKR exchange rates. Uses a free, keyless API; falls back to
    a clearly-labelled approximate rate if the request fails, so the UI never
    breaks even offline.
    """
    try:
        resp = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=5)
        data = resp.json()
        if data.get("result") == "success":
            rates = data["rates"]
            return {
                "success": True,
                "base": base,
                "LKR": rates.get("LKR"),
                "EUR": rates.get("EUR"),
                "GBP": rates.get("GBP"),
                "as_of": data.get("time_last_update_utc", ""),
            }
    except Exception:
        pass
    return {
        "success": False,
        "base": base,
        "LKR": 300.0,  # rough fallback only — flagged as approximate in the UI
        "EUR": 0.92,
        "GBP": 0.79,
        "as_of": "approximate — live rate unavailable",
    }


def generate_pdf(itinerary_text: str, trip_title: str = "My Sri Lanka Itinerary") -> bytes:
    """
    Renders the itinerary as a downloadable PDF. Emoji/markdown symbols are
    stripped since the core PDF font only supports Latin-1 text; headings
    and structure are preserved through simple markdown-aware formatting.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    def _clean(line: str) -> str:
        line = re.sub(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', '', line)  # strip emoji
        line = line.replace('**', '').replace('##', '').replace('---', '')
        return line.encode('latin-1', 'ignore').decode('latin-1').strip()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _clean(trip_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    for raw_line in itinerary_text.split("\n"):
        line = _clean(raw_line)
        if not line:
            pdf.ln(2)
            continue
        if raw_line.strip().startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(3)
        elif raw_line.strip().startswith("**") and raw_line.strip().endswith(":**"):
            pdf.set_font("Helvetica", "B", 11)
        else:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


PLACE_COORDS = {
    "Colombo":       (6.9271,  80.0000),
    "Kandy":         (7.2906,  80.6337),
    "Galle":         (6.0535,  80.2210),
    "Sigiriya":      (7.9570,  80.7603),
    "Ella":          (6.8667,  81.0466),
    "Nuwara Eliya":  (6.9497,  80.7891),
    "Mirissa":       (5.9483,  80.4716),
    "Unawatuna":     (6.0108,  80.2498),
    "Trincomalee":   (8.5874,  81.2152),
    "Anuradhapura":  (8.3114,  80.4037),
    "Polonnaruwa":   (7.9403,  81.0188),
    "Dambulla":      (7.8742,  80.6511),
    "Negombo":       (7.2095,  79.8386),
    "Hikkaduwa":     (6.1395,  80.1002),
    "Arugam Bay":    (6.8395,  81.8353),
    "Yala":          (6.3729,  81.5213),
    "Udawalawe":     (6.4748,  80.8992),
    "Horton Plains": (6.8021,  80.8103),
    "Adam's Peak":   (6.8096,  80.4994),
    "Pinnawala":     (7.3003,  80.3861),
    "Bentota":       (6.4248,  79.9956),
    "Matara":        (5.9549,  80.5550),
    "Jaffna":        (9.6615,  80.0255),
    "Batticaloa":    (7.7170,  81.6924),
    "Haputale":      (6.7667,  80.9667),
    "Tangalle":      (6.0249,  80.7997),
    "Weligama":      (5.9749,  80.4296),
    "Hatton":        (6.8953,  80.5950),
    "Bandarawela":   (6.8297,  81.0007),
    "Wilpattu":      (8.4560,  79.8880),
    "Minneriya":     (8.0292,  80.8991),
    "Kitulgala":     (6.9896,  80.4171),
    "Sinharaja":     (6.3953,  80.4584),
    "Kalpitiya":     (8.2333,  79.7667),
    "Koggala":       (5.9942,  80.3284),
    "Katunayake":    (7.1696,  79.8878),
    "Chilaw":        (7.5760,  79.7953),
    "Puttalam":      (8.0408,  79.8394),
    "Kurunegala":    (7.4818,  80.3609),
    "Hambantota":    (6.1241,  81.1185),
    "Tissamaharama": (6.2858,  81.2877),
    "Dickwella":     (5.9716,  80.6957),
    "Ambalangoda":   (6.2337,  80.0561),
    "Aluthgama":     (6.4329,  79.9994),
    "Kalutara":      (6.5854,  79.9607),
    "Beruwala":      (6.4785,  79.9828),
    "Ampara":        (7.2975,  81.6724),
    "Monaragala":    (6.8728,  81.3506),
    "Wellawaya":     (6.7333,  81.1000),
    "Embilipitiya":  (6.3500,  80.8500),
    "Balangoda":     (6.6500,  80.7000),
    "Avissawella":   (6.9500,  80.2167),
    "Nilaveli":      (8.6833,  81.2000),
    "Uppuveli":      (8.6167,  81.2167),
    "Pigeon Island": (8.7000,  81.2000),
    "Koneswaram":    (8.5850,  81.2330),
    "Mihintale":     (8.3500,  80.5000),
    "Bundala":       (6.1500,  81.2000),
    "Kanniya":       (8.6167,  81.1833),
}


def get_place_locations(place_names: list) -> list:
    locations = []
    for name in place_names:
        if name in PLACE_COORDS:
            lat, lon = PLACE_COORDS[name]
            locations.append({"name": name, "latitude": lat, "longitude": lon})
    return locations


# ── Seasonal Attractions & Local Events ─────────────────────────────────────
# Curated, hand-verified calendar of recurring seasonal attractions and
# festivals tied to specific places. Kept as static data rather than an API
# call — event dates for things like Esala Perahera shift year to year and
# aren't reliably available from a free API, so a curated table that's
# clearly framed as "usually happens around X" is safer than an
# unverified live source for a judged demo.
#
# months: list of month numbers (1-12) when this is relevant/in-season.
SEASONAL_EVENTS = {
    "Mirissa": [
        {
            "name": "Blue Whale Watching Season",
            "months": [11, 12, 1, 2, 3, 4],
            "note": "Mirissa's whale-watching boats run daily during this window — best sightings are typically December to March.",
        },
    ],
    "Trincomalee": [
        {
            "name": "Blue Whale Watching Season (East Coast)",
            "months": [5, 6, 7, 8, 9],
            "note": "The east coast whale-watching season runs opposite to Mirissa's, roughly May to September.",
        },
    ],
    "Kandy": [
        {
            "name": "Esala Perahera",
            "months": [7, 8],
            "note": "One of Asia's grandest Buddhist festivals — a ten-night procession of decorated elephants, dancers and drummers honoring the Sacred Tooth Relic. Exact dates shift yearly with the lunar calendar, so confirm before booking.",
        },
    ],
    "Galle": [
        {
            "name": "Galle Literary Festival",
            "months": [1],
            "note": "An annual literary festival held in Galle Fort, drawing international authors — adds a cultural/arts layer to a Galle visit.",
        },
        {
            "name": "Galle Fort food & cultural events",
            "months": [12],
            "note": "December sees a run of pop-up food and cultural events inside the Fort around the year-end season.",
        },
    ],
    "Jaffna": [
        {
            "name": "Nallur Festival",
            "months": [8],
            "note": "A 25-day Hindu temple festival at Nallur Kandaswamy Kovil with elaborate chariot processions — one of Sri Lanka's most significant Hindu festivals.",
        },
    ],
    "Negombo": [
        {
            "name": "Negombo Beach Season",
            "months": [12, 1, 2, 3],
            "note": "Calmest seas and best beach weather on the west coast fall in this window.",
        },
    ],
    "Arugam Bay": [
        {
            "name": "Surf Season",
            "months": [5, 6, 7, 8, 9],
            "note": "Arugam Bay's east-coast swell peaks in these months — the main draw for surfers visiting then.",
        },
    ],
    "Nuwara Eliya": [
        {
            "name": "Nuwara Eliya Season (Sri Lankan 'Little England' season)",
            "months": [4],
            "note": "April is peak season in the hill country — flower shows, horse racing and cool-climate tourism around the Sinhala/Tamil New Year.",
        },
    ],
    "Anuradhapura": [
        {
            "name": "Poson Poya",
            "months": [6],
            "note": "Second most important Buddhist festival after Vesak, marking the introduction of Buddhism to Sri Lanka — Anuradhapura and Mihintale see large pilgrim crowds.",
        },
    ],
    "Mihintale": [
        {
            "name": "Poson Poya",
            "months": [6],
            "note": "Mihintale is the epicentre of Poson celebrations — expect large crowds of pilgrims climbing the sacred steps.",
        },
    ],
}


def get_seasonal_highlights(place_names: list, travel_month: str | None = None) -> list:
    """
    Cross-references the itinerary's place names against SEASONAL_EVENTS and
    the traveller's month to surface anything timely — e.g. flags Mirissa
    whale watching if travelling in January, or Kandy's Esala Perahera if
    travelling in August. Returns [] if no travel_month is set or nothing
    matches, so callers can just check truthiness.
    """
    if not travel_month:
        return []

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    month_num = month_map.get(travel_month)
    if not month_num:
        return []

    highlights = []
    for place in place_names:
        for event in SEASONAL_EVENTS.get(place, []):
            if month_num in event["months"]:
                highlights.append({
                    "place": place,
                    "name": event["name"],
                    "note": event["note"],
                })
    return highlights


def get_weather_advisory(weather: dict) -> str | None:
    """
    Turns a get_weather() result into a plain-language heads-up when
    conditions are notable enough to affect what someone should pack or do
    that day — e.g. "it's unusually hot, carry water." Returns None for
    unremarkable weather so the UI only shows an advisory when it's
    actually useful, not on every load.
    """
    if not weather.get("success"):
        return None

    temp        = weather.get("temp")
    description = (weather.get("description") or "").lower()
    wind        = weather.get("wind", 0)

    if temp is not None and temp >= 33:
        return f"🌡️ It's {temp}°C and very hot — carry extra water, wear sunscreen, and plan outdoor sightseeing for early morning or late afternoon."
    if "rain" in description or "thunderstorm" in description or "drizzle" in description:
        return f"🌧️ {weather.get('description')} expected — pack a light rain jacket or umbrella, and build in flexibility for outdoor plans."
    if wind and wind >= 10:
        return f"💨 Windy conditions ({wind} m/s) — worth noting if any activities involve boats, beaches, or hiking exposed ridgelines."
    if temp is not None and temp <= 15:
        return f"🧥 A cool {temp}°C — pack a light jacket, especially useful for hill-country evenings (Nuwara Eliya, Ella, Haputale)."
    return None


def check_itinerary_weather(place_names: list) -> list:
    """
    Checks live weather at each place already in the itinerary and flags any
    with genuinely disruptive conditions (heavy rain/thunderstorms), so the
    refine flow can warn the user and the LLM can be asked to suggest an
    alternative for that specific day rather than the whole trip.

    Deliberately conservative about what counts as "bad" — ordinary heat or
    light cloud isn't flagged here, only conditions that would actually
    disrupt an outdoor day (matches get_weather_advisory's rain/storm check).
    Silently skips any place get_weather() can't resolve, so one bad
    API call doesn't block the rest of the check.
    """
    flags = []
    for place in place_names:
        weather = get_weather(place)
        if not weather.get("success"):
            continue
        description = (weather.get("description") or "").lower()
        if "rain" in description or "thunderstorm" in description or "drizzle" in description:
            flags.append({
                "place": place,
                "description": weather.get("description"),
                "temp": weather.get("temp"),
            })
    return flags


# ── Demo-Mode Fallback ────────────────────────────────────────────────────────
# A single pre-written, realistic itinerary kept in the exact format the LLM
# produces (so it flows through clean_text/annotate_sustainability/PDF export
# exactly like a real response would). This exists purely so that if Groq is
# down or rate-limited during a live demo/judging session, there's a one-click
# way to show a fully-populated, working app instead of a spinner or error —
# it never touches the network and can't fail.
SAMPLE_ITINERARY = """## Day 1: Negombo Beach Welcome

🚗 **Getting There:** From Bandaranaike International Airport to Negombo — 35 km · 45 min by taxi

**Afternoon:**
Land at BIA around 2pm, clear customs by 3pm. Transfer to Negombo (35 km, ~45 min by taxi). Check in to Jetwing Beach. Visit Negombo Fish Market for the late-afternoon catch. Walk along Lewis Place beach at dusk. Explore St. Mary's Church, a striking 17th-century Dutch colonial building.

**Evening:**
Seafood dinner at a beachfront restaurant on Lewis Place. Short walk along Negombo beach at sunset. Rest early ahead of tomorrow's journey inland.

🍽️ **Food Today:**
- Lunch: Light airport snack or hopper stall en route
- Dinner: Grilled prawns and crab curry at a Negombo beachfront restaurant
- Must-try: Negombo lagoon crab curry, best at a family-run seafood shack near the fish market

💰 **Estimated Cost:**
- Accommodation: LKR 20,000 at Jetwing Beach (approx USD 65)
- Food: LKR 4,500 (approx USD 15)
- Transport: LKR 6,200 (approx USD 20)
- Activities: LKR 0 (approx USD 0)
- Daily Total: approx USD 100

---

## Day 2: Cultural Triangle — Sigiriya Rock Fortress

🚗 **Getting There:** From Negombo to Sigiriya — 145 km · 3.5 hrs by private car

**Morning:**
Depart Negombo early for the 3.5-hour drive to Sigiriya. Climb the Sigiriya Rock Fortress (UNESCO World Heritage Site) — arrive by 7:30am to beat the heat and crowds. The climb takes about 2 hours round trip.

**Afternoon:**
Visit Sigiriya Museum for context on the ancient rock palace. Short rest at the hotel, then a village cycle tour through nearby paddy fields and local family homes (3 hrs, LKR 2,500).

**Evening:**
Dinner at the hotel overlooking the rock, lit up at night. Early night ahead of an early wildlife safari tomorrow.

🍽️ **Food Today:**
- Breakfast: Egg hoppers and sambol at the hotel
- Lunch: Rice and curry buffet near Sigiriya town
- Dinner: Kottu roti at Water Garden Sigiriya
- Must-try: Wood-apple juice, a Sigiriya-area specialty

💰 **Estimated Cost:**
- Accommodation: LKR 30,000 at Water Garden Sigiriya (approx USD 98)
- Food: LKR 5,000 (approx USD 16)
- Transport: LKR 15,000 (approx USD 49)
- Activities: LKR 7,000 (approx USD 23)
- Daily Total: approx USD 186

---

## 3 Important Travel Tips:
1. Climb Sigiriya Rock before 8am — both for cooler temperatures and to avoid the tour-bus crowds that arrive by mid-morning.
2. Always agree a tuk-tuk fare before getting in, or use PickMe/Uber, since meters are rarely used outside Colombo.
3. Carry small LKR notes for temple donations and shoe-storage fees — most sites don't give change for large bills.
"""


def get_sample_itinerary() -> tuple[str, dict]:
    """
    Returns the canned demo itinerary in the same (text, goal_eval) shape
    plan_trip() returns, so app.py can wire it up as a drop-in replacement
    with zero special-casing on the UI side.
    """
    goal = check_goal_achievement(SAMPLE_ITINERARY)
    return SAMPLE_ITINERARY, goal


# ── Main Agent Functions ──────────────────────────────────────────────────────
def _build_plan_trip_prompt(
    days: int,
    interests: list,
    budget: str,
    arrival_time: str = "morning",
    energy: str = "go",
    extra_info: str = "",
    memory_context: str = "",
    travel_month: str | None = None,
    language: str = "English",
) -> str:
    """
    Builds the user_prompt for an itinerary request. Pulled out of plan_trip()
    so the streaming path (plan_trip_stream) and the non-streaming path
    (plan_trip) build the exact same prompt from one place instead of two
    copies that could drift out of sync.
    """
    style   = decide_travel_style(interests, budget)
    arrival = decide_arrival_context(arrival_time, energy)
    interests_str = ", ".join(interests)
    seasonal_context = get_seasonal_context(travel_month)

    arrival_labels = {
        "morning":   "Morning (before 12 pm) — full day available from the moment they land",
        "afternoon": "Afternoon (12 pm – 6 pm) — Day 1 starts from Afternoon section only",
        "evening":   "Evening (6 pm – 10 pm) — Day 1 starts from Evening section only",
        "night":     "Night (after 10 pm) — Day 1 has Night section only, pure rest",
    }

    user_prompt = f"""
Please plan a {days}-day Sri Lanka travel itinerary for me!

My interests: {interests_str}
My budget: {budget}
My arrival time at Bandaranaike International Airport: {arrival_labels.get(arrival_time, arrival_time)}

AI Travel Style Decisions:
- Accommodation: {style['stay']}
- Travel pace: {style['pace']}
- Cost level: {style['cost_level']}
- Focus areas: {style['focus']}

═══════════ DAY 1 STRUCTURE INSTRUCTION — MUST FOLLOW EXACTLY ═══════════
{arrival['day1_instruction']}
═════════════════════════════════════════════════════════════════════════

═══════════ INTRA-DAY FEASIBILITY CHECK — APPLY TO EVERY DAY ═══════════
Before writing EACH day, mentally run this checklist:
1. List all Morning / Afternoon / Evening activities you plan to include.
2. Check the Sub-Location Travel Time Table in the system prompt for travel time between each.
3. If Morning→Afternoon travel exceeds 45 min → split across days.
4. If Afternoon→Evening travel exceeds 45 min → split across days.
5. If Morning activity is a full-day hike (Adam's Peak, Ella Rock, Horton Plains,
   Knuckles, Sinharaja) → Afternoon must be light (short stroll, rest, nearby cafe only).
6. NEVER place Adam's Peak and Ella activities on the same day — they are 90 km apart.
═════════════════════════════════════════════════════════════════════════

═══════════ ACCOMMODATION — MUST FOLLOW EXACTLY ════════════════════════
Always recommend 3–4 real hotel names per destination from the verified list.
Budget tier for this trip: {budget}
Always name the hotel in the Cost section.
NEVER say "guesthouse", "hostel", or "or similar".
═════════════════════════════════════════════════════════════════════════

═══════════ ACTIVITIES — MUST FOLLOW EXACTLY ═══════════════════════════
Use the Destination Activity Seed List in the system prompt as a starting point.
Always include at least one hidden gem per destination.
Be specific — name exact attractions, state opening times where known, and include entry costs.
═════════════════════════════════════════════════════════════════════════

Geographic Routing Instruction (MUST FOLLOW):
Plan the route as a one-directional journey — no backtracking. Group nearby attractions together.
Default flow (adjust per interests):
Airport area -> Cultural Triangle -> Kandy -> Hill Country -> South Coast -> Colombo departure.

Additional info: {extra_info if extra_info else "None"}

{f"═══════════ SEASONAL CONTEXT ═══════════{chr(10)}{seasonal_context}{chr(10)}═════════════════════════════════════════" if seasonal_context else ""}

{memory_context}

{f"═══════════ LANGUAGE — MUST FOLLOW ═══════════{chr(10)}Write the entire itinerary response in {language}. Keep place names, hotel names, and food names in their original form even if the surrounding text is in {language}.{chr(10)}═════════════════════════════════════════" if language and language != "English" else ""}

IMPORTANT: Mention specific Sri Lanka place names clearly in each day so they can be plotted on a map.
Create an amazing, practical, geographically smart day-by-day itinerary!
"""
    return user_prompt


def plan_trip(
    days: int,
    interests: list,
    budget: str,
    arrival_time: str = "morning",
    energy: str = "go",
    extra_info: str = "",
    memory_context: str = "",
    travel_month: str | None = None,
    language: str = "English",
) -> tuple[str, dict]:
    user_prompt = _build_plan_trip_prompt(
        days, interests, budget, arrival_time, energy, extra_info,
        memory_context, travel_month, language,
    )
    result, error = _call_llm(SYSTEM_PROMPT, user_prompt)
    if error:
        return f"⚠️ {error}", {"status": "error", "error": error}

    goal = check_goal_achievement(result)
    return result, goal


def plan_trip_stream(
    days: int,
    interests: list,
    budget: str,
    arrival_time: str = "morning",
    energy: str = "go",
    extra_info: str = "",
    memory_context: str = "",
    travel_month: str | None = None,
    language: str = "English",
):
    """
    Streaming counterpart to plan_trip(). Returns a generator of text chunks
    meant to be passed straight into st.write_stream(...) in app.py, so the
    itinerary appears token-by-token instead of after one long wait behind
    a spinner.

    Same retry/fallback-model safety net as plan_trip(): if the streaming
    call itself fails to even start (bad key, connection refused before any
    chunk arrives), it falls back to the non-streaming _call_llm() — with
    its own retries and fallback model — and yields the whole result as one
    chunk, so the caller can treat the return value of st.write_stream(...)
    identically either way.
    """
    user_prompt = _build_plan_trip_prompt(
        days, interests, budget, arrival_time, energy, extra_info,
        memory_context, travel_month, language,
    )
    yield from _stream_llm(SYSTEM_PROMPT, user_prompt)


def refine_trip(itinerary: str, refinement_request: str, weather_flags: list | None = None) -> tuple[str, dict]:
    weather_instruction = ""
    if weather_flags:
        flagged_desc = "; ".join(
            f"{f['place']} (currently {f['description']}, {f['temp']}°C)"
            for f in weather_flags
        )
        weather_instruction = (
            f"\n\nWEATHER ALERT — these planned destinations currently have poor weather: "
            f"{flagged_desc}. If any of them are still in the itinerary after applying my "
            f"requested change, adjust that day's outdoor activity to a suitable nearby "
            f"alternative (an indoor attraction, a nearby destination with better "
            f"conditions, or a flexible/rest activity), and add one short sentence in that "
            f"day noting the weather-based change.\n"
        )

    user_prompt = (
        f"Here is my current itinerary:\n{itinerary}\n\n"
        f"Please make this change: {refinement_request}\n"
        f"{weather_instruction}\n"
        "IMPORTANT: Mention specific Sri Lanka place names clearly in each day.\n"
        "IMPORTANT: Before finalising each day, verify all Morning→Afternoon→Evening "
        "activity transitions are within 45 minutes of each other using the "
        "Sub-Location Travel Time Table.\n"
        "IMPORTANT: Recommend 3–4 real hotel names per destination from the verified list.\n"
        "IMPORTANT: Always name the hotel in the Cost section — never say 'or similar'.\n"
        "IMPORTANT — GEOGRAPHIC ROUTING: even after this change, the whole trip must stay a "
        "one-directional journey with NO backtracking. If the requested change adds or swaps "
        "destinations (e.g. more beaches), re-order ALL days so the route travels through "
        "them in a single geographic direction — never visit a region, move past it, and then "
        "come back to it later in the trip."
    )

    result, error = _call_llm(REFINE_PROMPT, user_prompt)
    if error:
        return f"⚠️ {error}", {"status": "error", "error": error}

    goal = check_goal_achievement(result)
    return result, goal


def chat_with_agent(messages: list, user_message: str, itinerary: str) -> tuple[str, list]:
    system_prompt = CHAT_PROMPT + f"\n\nUser's itinerary:\n{itinerary}"
    reply, error = _call_llm(system_prompt, user_message)

    if error:
        reply = f"⚠️ {error}"
    else:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
    return reply, messages
