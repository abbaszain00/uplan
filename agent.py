import json
import os
import requests
from datetime import datetime, timedelta
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Secrets ───────────────────────────────────────────────────────────────────
try:
    import streamlit as st
    if "OPENROUTER_API_KEY" in st.secrets:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    if "TICKETMASTER_API_KEY" in st.secrets:
        os.environ["TICKETMASTER_API_KEY"] = st.secrets["TICKETMASTER_API_KEY"]
except Exception:
    pass

# ── Firestore ─────────────────────────────────────────────────────────────────
_db = None

def get_db():
    global _db
    if _db is not None:
        return _db
    try:
        import streamlit as st
        from google.cloud import firestore
        from google.oauth2 import service_account

        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        _db = firestore.Client(credentials=credentials, project=creds_dict["project_id"])
        return _db
    except Exception as e:
        print(f"Firestore init failed: {e}")
        return None


USER_DOC = "default_user"


def load_memory() -> dict:
    try:
        db = get_db()
        if db is None:
            return {}
        doc = db.collection("uplan_memory").document(USER_DOC).get()
        if doc.exists:
            return doc.to_dict()
        return {}
    except Exception as e:
        print(f"load_memory error: {e}")
        return {}


def save_memory(data: dict):
    try:
        db = get_db()
        if db is None:
            return
        ref = db.collection("uplan_memory").document(USER_DOC)
        ref.set(data, merge=True)
    except Exception as e:
        print(f"save_memory error: {e}")


# ── Event cache (in-memory per session) ───────────────────────────────────────
_event_cache: dict = {}


def get_cached_events(key: str, max_age_minutes: int = 30) -> list | None:
    if key not in _event_cache:
        return None
    entry = _event_cache[key]
    cached_at = datetime.fromisoformat(entry["timestamp"])
    if datetime.now() - cached_at < timedelta(minutes=max_age_minutes):
        return entry["events"]
    return None


def save_event_cache(key: str, data: list):
    _event_cache[key] = {"timestamp": datetime.now().isoformat(), "events": data}


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def remember_user_facts(facts: dict) -> str:
    """
    Store important long-term facts about the user persistently.
    Pass a dict of key-value pairs, e.g.:
    {"budget": "zero/free only", "location": "London", "interests": ["music", "sport"]}
    Call this whenever the user shares something important about themselves.
    """
    save_memory(facts)
    return f"Got it — I've remembered: {json.dumps(facts, indent=2)}"


@tool
def get_user_facts() -> str:
    """
    Retrieve all stored long-term facts about the user.
    ALWAYS call this at the very start of every new session.
    """
    memory = load_memory()
    if not memory:
        return "No user facts stored yet — this appears to be a new user."
    return json.dumps(memory, indent=2)


@tool
def search_events(location: str, keyword: str = "", start_date: str = "", end_date: str = "") -> str:
    """
    Search for live events using the Ticketmaster API.
    Args:
        location: City name, e.g. 'London'
        keyword: Optional search term, e.g. 'music', 'sport', 'comedy'
        start_date: Optional ISO date string YYYY-MM-DD (defaults to today)
        end_date: Optional ISO date string YYYY-MM-DD (defaults to 2 weeks from today)
    Returns a formatted list of upcoming events.
    """
    api_key = os.getenv("TICKETMASTER_API_KEY", "")
    if not api_key:
        return "Ticketmaster API key not configured."

    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%dT00:00:00Z")
    else:
        start_date = f"{start_date}T00:00:00Z"

    if not end_date:
        end_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%dT00:00:00Z")
    else:
        end_date = f"{end_date}T00:00:00Z"

    cache_key = f"{location}_{keyword}_{start_date}_{end_date}"
    cached = get_cached_events(cache_key)
    if cached:
        events = cached
        source = "cache"
    else:
        url = "https://app.ticketmaster.com/discovery/v2/events.json"
        params = {
            "apikey": api_key,
            "city": location,
            "keyword": keyword,
            "startDateTime": start_date,
            "endDateTime": end_date,
            "size": 10,
            "sort": "date,asc",
            "locale": "*",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"Error fetching events: {str(e)}"

        raw_events = data.get("_embedded", {}).get("events", [])
        events = []
        for e in raw_events:
            venue = e.get("_embedded", {}).get("venues", [{}])[0]
            price_ranges = e.get("priceRanges", [])
            if price_ranges:
                price_str = f"£{price_ranges[0].get('min', '?')} – £{price_ranges[0].get('max', '?')}"
            else:
                price_str = "Price not listed"

            events.append({
                "name": e.get("name", "Unknown"),
                "date": e.get("dates", {}).get("start", {}).get("localDate", "TBC"),
                "time": e.get("dates", {}).get("start", {}).get("localTime", ""),
                "venue": venue.get("name", "Unknown venue"),
                "address": venue.get("address", {}).get("line1", ""),
                "url": e.get("url", ""),
                "price": price_str,
                "genre": e.get("classifications", [{}])[0].get("genre", {}).get("name", ""),
            })

        save_event_cache(cache_key, events)
        source = "live"

    if not events:
        return f"No events found in {location} for '{keyword}' in that date range."

    lines = [f"📍 Events in {location}" + (f" — '{keyword}'" if keyword else "") + f" (source: {source})\n"]
    for i, ev in enumerate(events, 1):
        time_str = f" at {ev['time'][:5]}" if ev['time'] else ""
        lines.append(
            f"{i}. **{ev['name']}**\n"
            f"   📅 {ev['date']}{time_str}\n"
            f"   🏟️  {ev['venue']}{', ' + ev['address'] if ev['address'] else ''}\n"
            f"   💰 {ev['price']}\n"
            f"   🔗 {ev['url']}\n"
        )
    return "\n".join(lines)


@tool
def vet_recommendations(events_text: str, user_facts_json: str) -> str:
    """
    Vetting sub-agent: given a list of events and user facts, filter and rank
    them according to the user's stated preferences (budget, interests, availability).
    Returns a curated, preference-aligned shortlist with reasoning.
    """
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        temperature=0.3,
    )

    system = """You are UPlan's vetting agent. Your job is to filter and rank a list of events 
based on the user's stored preferences. 

Rules:
- If the user has a tight/zero budget, prioritise free or very low cost events. Flag expensive ones clearly.
- Match interests where possible (music, sport, etc.)
- If availability/dates are known, filter to those windows.
- Return a concise ranked shortlist (max 5) with a one-line reason for each pick.
- Be honest if nothing fits well — say so and suggest alternatives.
- Keep your tone friendly and brief."""

    prompt = f"""User preferences:
{user_facts_json}

Events to vet:
{events_text}

Return a curated shortlist with brief reasoning for each pick."""

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return response.content


# ── Graph state ───────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── Agent ─────────────────────────────────────────────────────────────────────

def build_agent():
    tools = [remember_user_facts, get_user_facts, search_events, vet_recommendations]

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        temperature=0.7,
    )

    llm_with_tools = llm.bind_tools(tools)

    SYSTEM_PROMPT = """You are UPlan 🎯 — a warm, enthusiastic personal hobby and event planner.

Your personality: friendly, encouraging, curious about the user, never overwhelming.

Your capabilities:
1. REMEMBER users long-term — at the START of every conversation, ALWAYS call get_user_facts first before responding.
2. LEARN from users — when they share preferences (budget, location, interests, availability), call remember_user_facts immediately.
3. FIND events — when a user wants event suggestions, call search_events with appropriate parameters.
4. VET recommendations — after fetching events, ALWAYS call vet_recommendations passing the events and user facts before presenting results. Never show raw unvetted results.
5. UPDATE memory — if the user changes their mind (e.g. switches from music to sport), update memory with the new preference.

Conversation flow:
- First message of ANY session: silently call get_user_facts FIRST, then greet.
- If you know them: reference what you remember naturally e.g. "Welcome back! I remember you're keen on keeping things budget-friendly — still the case?"
- If new user: introduce yourself briefly and ask what they're interested in.
- Ask for missing info naturally (location, budget, dates) — one question at a time.
- When showing events: present the VETTED shortlist only, not the raw API dump.
- Keep responses concise and conversational. Use emojis sparingly but warmly.

Budget awareness: If a user mentions tight/zero budget, this is top priority. Never recommend paid events without flagging the cost clearly.

Today's date: """ + datetime.now().strftime("%A, %d %B %Y")

    tool_node = ToolNode(tools)

    def call_model(state: State):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: State):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


agent = build_agent()


def chat(message: str, history: list) -> str:
    """Main entry point. history is a list of (human, ai) tuples."""
    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        if ai:
            messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=message))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content