from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import streamlit as st

from energy_agent import EnergyNewsAgent, PRIORITY_ORDER
from news_collector import collect_energy_candidates, deduplicate_articles
from sources import CATEGORIES, SOURCES


st.set_page_config(
    page_title="Pakistan Energy News Agent",
    page_icon="🛢️",
    layout="wide",
)

st.title("🛢️ Pakistan Energy News Agent")
st.caption(
    "AI-curated petroleum, OMC, refinery, gas and power-sector news from approved Pakistani newspapers."
)

with st.sidebar:
    st.header("Settings")

    secret_key = ""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    groq_key = secret_key or os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input(
            "Groq API key",
            type="password",
            help="For deployment, store this in Streamlit Cloud Secrets instead of typing it each time.",
        )

    period_label = st.radio(
        "News period",
        ["Last 24 hours", "Last 3 days", "Last 7 days"],
        index=0,
    )
    window_days = {
        "Last 24 hours": 1,
        "Last 3 days": 3,
        "Last 7 days": 7,
    }[period_label]

    categories = st.multiselect(
        "Categories",
        CATEGORIES,
        default=CATEGORIES,
    )

    priorities = st.multiselect(
        "Priority",
        ["High", "Medium", "General"],
        default=["High", "Medium", "General"],
    )

    max_stories = st.slider("Maximum stories", 5, 25, 12, 1)

    broad_scan = st.checkbox(
        "Broader scan",
        value=False,
        help=(
            "Scans more headlines before AI classification. It can find less obvious stories "
            "but takes longer and uses more requests."
        ),
    )

    st.divider()
    st.subheader("Approved sources")
    for src in SOURCES:
        st.markdown(f"• [{src['name']}]({src['base_url']})")


def priority_icon(priority: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "General": "⚪"}.get(priority, "⚪")


def format_date(value) -> str:
    if not value:
        return "Publication time not detected"
    if isinstance(value, datetime):
        return value.astimezone(ZoneInfo("Asia/Karachi")).strftime("%d %b %Y, %I:%M %p PKT")
    return str(value)


def domain_label(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def cached_collect(window_days: int, broad_scan: bool):
    return collect_energy_candidates(
        window_days=window_days,
        broad_scan=broad_scan,
        per_source_limit=45,
    )


if "brief" not in st.session_state:
    st.session_state.brief = []
if "collection_errors" not in st.session_state:
    st.session_state.collection_errors = []


if st.button("Generate Energy Brief", type="primary", use_container_width=True):
    if not groq_key:
        st.error("Please enter your Groq API key, or add GROQ_API_KEY to Streamlit Secrets.")
        st.stop()

    if not categories:
        st.error("Select at least one category.")
        st.stop()

    try:
        with st.status("Building your energy brief…", expanded=True) as status:
            st.write("Reading the approved newspaper sources…")
            candidates, errors = cached_collect(window_days, broad_scan)

            if not candidates:
                status.update(label="No candidate energy stories found.", state="complete")
                st.session_state.brief = []
                st.session_state.collection_errors = errors
            else:
                st.write(f"Found {len(candidates)} candidate stories. Running AI relevance screening…")
                agent = EnergyNewsAgent(groq_key)
                classified = agent.classify(candidates, categories)

                relevant = [
                    a
                    for a in classified
                    if a.get("relevant")
                    and a.get("category") in categories
                    and a.get("priority") in priorities
                ]

                relevant = deduplicate_articles(relevant)
                relevant.sort(
                    key=lambda a: (
                        PRIORITY_ORDER.get(a.get("priority", "General"), 9),
                        -(a["published_at"].timestamp() if a.get("published_at") else 0),
                    )
                )
                relevant = relevant[:max_stories]

                if relevant:
                    st.write(f"Summarizing the top {len(relevant)} relevant stories…")
                    brief = agent.summarize(relevant)
                else:
                    brief = []

                st.session_state.brief = brief
                st.session_state.collection_errors = errors
                status.update(label="Energy brief ready.", state="complete")

    except Exception as exc:
        st.exception(exc)
        st.info(
            "If this is a temporary newspaper blocking/rate-limit issue, try again later. "
            "If it is an AI/API error, verify the Groq key in Streamlit Secrets."
        )


brief = st.session_state.brief

if brief:
    high_count = sum(1 for x in brief if x.get("priority") == "High")
    medium_count = sum(1 for x in brief if x.get("priority") == "Medium")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stories", len(brief))
    c2.metric("High priority", high_count)
    c3.metric("Medium priority", medium_count)

    st.divider()

    for item in brief:
        priority = item.get("priority", "General")
        category = item.get("category", "Energy Policy")
        headline = item.get("short_headline") or item.get("title")

        st.subheader(f"{priority_icon(priority)} {headline}")
        st.caption(
            f"{priority.upper()}  ·  {category}  ·  {item.get('source', '')}  ·  "
            f"{format_date(item.get('published_at'))}"
        )
        st.write(item.get("summary", ""))

        if item.get("why_it_matters"):
            st.markdown(f"**Why it matters:** {item['why_it_matters']}")

        cols = st.columns([1, 4])
        with cols[0]:
            st.link_button("Read original", item["url"], use_container_width=True)
        with cols[1]:
            alternates = item.get("alternate_sources", [])
            if alternates:
                alt_links = " · ".join(
                    f"[{a['source']}]({a['url']})" for a in alternates[:4]
                )
                st.markdown(f"**Other coverage:** {alt_links}")

        st.divider()

elif st.session_state.collection_errors:
    st.warning("No brief was produced. Some source pages could not be read.")


with st.expander("Source diagnostics"):
    errors = st.session_state.collection_errors
    if not errors:
        st.write("No collection errors recorded in this session.")
    else:
        st.caption(
            "News websites can change layout or temporarily block automated requests. "
            "These diagnostics help identify which source needs an extractor update."
        )
        for err in errors[:30]:
            st.code(err)

st.caption(
    "AI summaries can contain errors. Use the original article links for verification before making decisions."
)
