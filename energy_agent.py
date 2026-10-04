from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task

from sources import CATEGORIES


PRIORITY_ORDER = {"High": 0, "Medium": 1, "General": 2}


def _extract_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Recover the largest plausible JSON array/object from verbose model output.
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
    """CrewAI-powered editor for Pakistan petroleum and power-sector intelligence."""

    def __init__(self, api_key: str):
        self.llm = LLM(
            model="openai/openai/gpt-oss-20b",
            custom_openai=True,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            temperature=0.1,
        )

        self.editor = Agent(
            role="Pakistan Energy Intelligence Editor",
            goal=(
                "Identify and explain the most decision-relevant Pakistan petroleum, "
                "OMC, refinery, gas and power-sector news with high factual discipline."
            ),
            backstory=(
                "You are a senior energy-sector news editor familiar with Pakistan's "
                "petroleum supply chain, OMCs, refineries, OGRA, Petroleum Division, "
                "NEPRA, power-sector circular debt, gas/LNG markets and energy policy. "
                "You never invent facts that are absent from the supplied article."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
        )

    def classify(self, articles: list[dict], selected_categories: list[str]) -> list[dict]:
        if not articles:
            return []

        payload = []
        for idx, a in enumerate(articles):
            payload.append(
                {
                    "id": idx,
                    "source": a.get("source"),
                    "title": a.get("title"),
                    "description": a.get("description") or a.get("listing_snippet", ""),
                    "body_excerpt": (a.get("body") or "")[:1800],
                }
            )

        description = f"""
Evaluate the supplied newspaper stories for a Pakistan-focused energy intelligence brief.

Allowed categories:
{", ".join(CATEGORIES)}

The user currently selected these categories:
{", ".join(selected_categories)}

A story is RELEVANT when it materially concerns Pakistan's:
- oil, petroleum products, petrol, diesel/HSD, PMG, crude, POL, petroleum levy or fuel pricing;
- OMCs, dealer economics, petroleum logistics, storage, imports, supply or sales;
- refineries, refinery policy, refinery upgrades, furnace oil or related downstream matters;
- natural gas, LNG, RLNG or LPG;
- electricity/power, NEPRA, CPPA, DISCOs, NTDC, IPPs, K-Electric, generation, transmission,
  circular debt, load management, hydel, solar or renewables;
- energy regulation, taxation, government policy or major decisions affecting the above.

International oil/LNG stories are relevant only when they have clear potential implications for
Pakistan's supply, prices, imports or energy security.

Priority rules:
- High: immediate commercial/regulatory/supply/pricing consequence.
- Medium: meaningful sector development, project, company or policy development.
- General: contextual international/market information useful to an energy professional.

Return ONLY a valid JSON array. One object per input item:
[
  {{
    "id": 0,
    "relevant": true,
    "category": "Petroleum",
    "priority": "High",
    "reason": "one short sentence"
  }}
]

If not relevant, category must still be the closest allowed category and priority should be "General".

ARTICLES:
{json.dumps(payload, ensure_ascii=False)}
"""

        task = Task(
            description=description,
            expected_output="A valid JSON array only, following the requested schema.",
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
            raise ValueError("Classifier did not return a JSON array.")

        by_id = {int(x["id"]): x for x in parsed if isinstance(x, dict) and "id" in x}
        output = []
        for idx, article in enumerate(articles):
            decision = by_id.get(idx, {})
            article = dict(article)
            article["relevant"] = bool(decision.get("relevant", False))
            article["category"] = decision.get("category", "Energy Policy")
            article["priority"] = decision.get("priority", "General")
            article["classification_reason"] = decision.get("reason", "")
            if article["category"] not in CATEGORIES:
                article["category"] = "Energy Policy"
            if article["priority"] not in PRIORITY_ORDER:
                article["priority"] = "General"
            output.append(article)
        return output

    def summarize(self, articles: list[dict]) -> list[dict]:
        if not articles:
            return []

        payload = []
        for idx, a in enumerate(articles):
            published = a.get("published_at")
            if isinstance(published, datetime):
                published = published.isoformat()

            payload.append(
                {
                    "id": idx,
                    "source": a.get("source"),
                    "title": a.get("title"),
                    "published": published,
                    "category": a.get("category"),
                    "priority": a.get("priority"),
                    "description": a.get("description", ""),
                    "article_text": (a.get("body") or a.get("listing_snippet", ""))[:5000],
                }
            )

        description = f"""
Prepare concise intelligence summaries of the supplied energy news stories for a senior
Pakistan petroleum/OMC professional.

For EACH article:
1. summary: 2-3 short sentences. State only facts supported by the supplied article.
2. why_it_matters: 1 short sentence explaining the likely commercial, supply, regulatory,
   pricing or sector significance. Clearly use cautious wording ("may", "could") when the
   impact is an inference rather than an explicit fact.
3. short_headline: a concise factual headline, maximum 14 words.

Do not add external facts. Do not invent figures, company names, policies or consequences.
Return ONLY valid JSON:
[
  {{
    "id": 0,
    "short_headline": "...",
    "summary": "...",
    "why_it_matters": "..."
  }}
]

ARTICLES:
{json.dumps(payload, ensure_ascii=False)}
"""
        task = Task(
            description=description,
            expected_output="A valid JSON array only with a concise summary for every article.",
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
            raise ValueError("Summarizer did not return a JSON array.")

        by_id = {int(x["id"]): x for x in parsed if isinstance(x, dict) and "id" in x}
        output = []
        for idx, article in enumerate(articles):
            summary = by_id.get(idx, {})
            item = dict(article)
            item["short_headline"] = summary.get("short_headline") or item.get("title", "")
            item["summary"] = summary.get("summary") or item.get("description", "")
            item["why_it_matters"] = summary.get("why_it_matters") or item.get(
                "classification_reason", ""
            )
            output.append(item)
        return output
