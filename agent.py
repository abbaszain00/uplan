import json
import os
import requests
import secrets_loader  # noqa: F401 — loads API keys into env
from datetime import datetime, timedelta
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Memory file ──────────────────────────────────────────────────────────────
MEMORY_FILE = "user_memory.json"


def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_memory(data: dict):
    existing = load_memory()
    existing.update(data)
    with open(MEMORY_FILE, "w") as f:
        json.dump(existing, f, indent=2)


# ── Event cache ───────────────────────────────────────────────────────────────
EVENT_CACHE_FILE = "event_cache.json"


def load_event_cache() -> dict:
    if os.path.exists(EVENT_CACHE_FILE):
        with open(EVENT_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_event_cache(key: str, data: list):
    cache = load_event_cache()
    cache[key] = {"timestamp": datetime.now().isoformat(), "events": data}
    with open(EVENT_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_cached_events(key: str, max_age_minutes: int = 30) -> list | None:
    cache = load_event_cache()
    if key not in cache:
        return None
    entry = cache[key]
    cached_at = datetime.fromisoformat(entry["timestamp"])
    if datetime.now() - cached_at < timedelta(minutes=max_age_minutes):
        return entry["events"]
    return None


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def remember_user_facts(facts: dict) -> str:
    """
    Store important long-term facts about the user.
    Pass a dict of key-value pairs, e.g.:
    {"budget": "zero/free only", "location": "London", "interests": ["music", "sport"]}
    Keys can be anything meaningful: budget, location, interests, availability, preferences, etc.
    """
    save_memory(facts)
    return f"Got it — I've remembered: {json.dumps(facts, indent=2)}"


@tool
def get_user_facts() -> str:
    """
    Retrieve all stored long-term facts about the user.
    Call this at the start of a new session or when you need to recall what you know about the user.
    """
    memory = load_memory()
    if not memory:
        return "No user facts stored yet."
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

    # Default dates
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
        model="mistralai/mistral-small-3.1-24b-instruct:free",
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


# ── LLM + tools ───────────────────────────────────────────────────────────────

def build_agent():
    tools = [remember_user_facts, get_user_facts, search_events, vet_recommendations]

    llm = ChatOpenAI(
        model="mistralai/mistral-small-3.1-24b-instruct:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        temperature=0.7,
    )

    llm_with_tools = llm.bind_tools(tools)

    SYSTEM_PROMPT = """You are UPlan 🎯 — a warm, enthusiastic personal hobby and event planner.

Your personality: friendly, encouraging, curious about the user, never overwhelming.

Your capabilities:
1. REMEMBER users long-term — at the START of every conversation, call get_user_facts to recall what you know.
2. LEARN from users — when they share preferences (budget, location, interests, availability), call remember_user_facts immediately.
3. FIND events — when a user wants event suggestions, call search_events with appropriate parameters.
4. VET recommendations — after fetching events, ALWAYS call vet_recommendations passing the events and user facts before presenting results to the user. Never show raw unvetted results.
5. UPDATE memory — if the user changes their mind (e.g. switches from music to sport), update memory accordingly.

Conversation flow:
- First message of a session: greet warmly + silently call get_user_facts to check if you know them.
- If you know them: reference what you remember naturally ("Welcome back! Still keen on keeping things budget-friendly?")
- If new user: introduce yourself briefly and ask what they're interested in.
- Ask for missing info naturally (location, budget, dates) — don't bombard with all questions at once.
- When showing events: present the VETTED shortlist, not the raw API dump.
- Keep responses concise and conversational. Use emojis sparingly but warmly.

Budget awareness: If a user mentions tight budget or free-only, this is a top priority filter. Never recommend paid events without flagging the cost clearly.

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