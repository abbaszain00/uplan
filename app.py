import streamlit as st
from agent import chat

st.set_page_config(
    page_title="UPlan — Your Hobby & Event Planner",
    page_icon="🎯",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 720px; margin: auto; }
    .stChatMessage { border-radius: 12px; }
    h1 { color: #6C63FF; }
    .subtitle { color: #888; font-size: 0.95rem; margin-top: -12px; margin-bottom: 24px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 UPlan")
st.markdown('<p class="subtitle">Your personal hobby & event planner — powered by AI</p>', unsafe_allow_html=True)

with st.expander("ℹ️ What can UPlan do?", expanded=False):
    st.markdown("""
- 🎵 **Discover hobbies** — explore new interests through conversation
- 📍 **Find live events** — real upcoming events via Ticketmaster
- 🧠 **Remember you** — recalls your preferences across sessions
- 💰 **Respects your budget** — filters recommendations to your constraints
- 🏃 **Any interest** — music, sport, arts, food, outdoors and more

**Try saying:**
- *"I'm looking for free music events in London this weekend"*
- *"I want to try a new sport, I'm free on Friday"*
- *"What do you know about me?"*
- *"I've changed my mind — I'm more into comedy now"*
    """)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "greeted" not in st.session_state:
    st.session_state.greeted = False

# ── Auto-greeting on first load ───────────────────────────────────────────────
if not st.session_state.greeted:
    with st.spinner("UPlan is waking up..."):
        greeting = chat("Hello, I'm starting a new session.", [])
    st.session_state.history.append(("Hello, I'm starting a new session.", greeting))
    st.session_state.greeted = True

# ── Render chat history ───────────────────────────────────────────────────────
for human_msg, ai_msg in st.session_state.history:
    if human_msg != "Hello, I'm starting a new session.":
        with st.chat_message("user"):
            st.markdown(human_msg)
    with st.chat_message("assistant", avatar="🎯"):
        st.markdown(ai_msg)

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Tell UPlan what you're looking for..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎯"):
        with st.spinner("Thinking..."):
            response = chat(prompt, st.session_state.history)
        st.markdown(response)

    st.session_state.history.append((prompt, response))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("UPlan remembers your preferences across sessions. Events sourced live from Ticketmaster.")
with col2:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.greeted = False
        st.rerun()