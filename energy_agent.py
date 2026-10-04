from __future__ import annotations

import json
import re
from datetime import datetime

from crewai import Agent, Crew, LLM, Process, Task

from sources import CATEGORIES


PRIORITY_ORDER = {"High": 0, "Medium": 1, "General": 2}
MAX_AI_CANDIDATES = 8


def _extract_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError("No JSON found in model output.")

    start = min(starts)
    for end_char in ("]", "}"):
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue

    raise ValueError("Could not parse JSON from model output.")


class EnergyNewsAgent:
    """CrewAI-powered editor optimized for Groq's 8K TPM on-demand limit."""

    def __init__(self, api_key: str):
        # CrewAI custom_openai removes one leading "openai/" routing prefix.
        # The double prefix ensures Groq receives the actual catalog model ID:
        # openai/gpt-oss-20b
        self.llm = LLM(
            model="openai/openai/gpt-oss-20b",
            custom_openai=True,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            temperature=0.1,
            max_tokens=1800,
        )

        self.editor = Agent(
            role="Pakistan Energy Intelligence Editor",
            goal=(
                "Select and summarize decision-relevant Pakistan petroleum, OMC, refinery, "
                "gas and power-sector news accurately and concisely."
            ),
            backstory=(
                "You are a senior Pakistan energy-sector editor. You understand petroleum "
                "pricing, OMCs, refineries, OGRA, Petroleum Division, gas/LNG, NEPRA, "
                "electricity, circular debt and power-sector regulation. You never invent "
                "facts not contained in the supplied story."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
        )

    def analyze(self, articles: list[dict], selected_categories: list[str]) -> list[dict]:
        """
        One LLM call performs relevance classification + priority + summary.
        This is intentionally compact to remain below Groq's 8K TPM limit.
        """
        if not articles:
            return []

        articles = articles[:MAX_AI_CANDIDATES]

        payload = []
        for idx, a in enumerate(articles):
            published = a.get("published_at")
            if isinstance(published, datetime):
                published = published.isoformat()

            description = (a.get("description") or a.get("listing_snippet") or "")[:320]
            excerpt = (a.get("body") or "")[:520]

            payload.append(
                {
                    "id": idx,
                    "source": a.get("source", ""),
                    "title": a.get("title", "")[:220],
                    "published": published,
                    "description": description,
                    "excerpt": excerpt,
                }
            )

        description = f"""
Create a compact Pakistan energy intelligence brief from the newspaper items below.

Selected categories:
{", ".join(selected_categories)}

Allowed categories:
{", ".join(CATEGORIES)}

A story is relevant when it materially concerns Pakistan's petroleum/fuel market, OMCs,
refineries, gas/LNG/LPG, electricity/power, or energy regulation/policy. International
oil or LNG news is relevant only when it has a clear potential implication for Pakistan.

Priority:
- High = immediate pricing, supply, regulatory or major commercial consequence.
- Medium = meaningful sector/company/project/policy development.
- General = useful context but lower immediate impact.

For every input article return ONE JSON object with:
- id
- relevant: true/false
- category: one allowed category
- priority: High, Medium or General
- short_headline: max 12 words
- summary: max 2 concise sentences
- why_it_matters: max 1 concise sentence

Use only facts in the supplied text. If impact is inferred, use cautious wording such as
"may" or "could". Do not add external facts.

Return ONLY a valid JSON array.

ARTICLES:
{json.dumps(payload, ensure_ascii=False)}
"""

        task = Task(
            description=description,
            expected_output=(
                "A valid compact JSON array containing relevance, category, priority, "
                "headline, summary and why_it_matters for each input article."
            ),
            agent=self.editor,
        )

        crew = Crew(
            agents=[self.editor],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        parsed = _extract_json(result.raw)

        if not isinstance(parsed, list):
            raise ValueError("Energy editor did not return a JSON array.")

        by_id = {
            int(item["id"]): item
            for item in parsed
            if isinstance(item, dict) and "id" in item
        }

        output = []
        for idx, article in enumerate(articles):
            decision = by_id.get(idx, {})
            item = dict(article)

            item["relevant"] = bool(decision.get("relevant", False))
            item["category"] = decision.get("category", "Energy Policy")
            item["priority"] = decision.get("priority", "General")
            item["short_headline"] = (
                decision.get("short_headline") or item.get("title", "")
            )
            item["summary"] = (
                decision.get("summary")
                or item.get("description")
                or item.get("listing_snippet", "")
            )
            item["why_it_matters"] = decision.get("why_it_matters", "")

            if item["category"] not in CATEGORIES:
                item["category"] = "Energy Policy"
            if item["priority"] not in PRIORITY_ORDER:
                item["priority"] = "General"

            output.append(item)

        return output
