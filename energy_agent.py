from __future__ import annotations

import json
from datetime import datetime

from openai import OpenAI

from sources import CATEGORIES


PRIORITY_ORDER = {"High": 0, "Medium": 1, "General": 2}
MAX_AI_CANDIDATES = 16


class EnergyNewsAgent:
    """
    Pakistan Energy News Agent.

    Uses Groq's OpenAI-compatible endpoint with strict JSON Schema output.
    This removes the fragile free-form JSON parsing step.
    """

    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def analyze(self, articles: list[dict], selected_categories: list[str]) -> list[dict]:
        if not articles:
            return []

        articles = articles[:MAX_AI_CANDIDATES]

        payload = []
        for idx, article in enumerate(articles):
            published = article.get("published_at")
            if isinstance(published, datetime):
                published = published.isoformat()

            # Keep the input compact to stay comfortably below the 8K TPM limit.
            payload.append(
                {
                    "id": idx,
                    "source": article.get("source", ""),
                    "title": (article.get("title") or "")[:220],
                    "published": published or "",
                    "description": (
                        article.get("description")
                        or article.get("listing_snippet")
                        or ""
                    )[:180],
                    "excerpt": (article.get("body") or "")[:260],
                }
            )

        instructions = f"""
You are a senior Pakistan energy-sector intelligence editor.

Review the supplied newspaper items and decide which are materially relevant to:
- Petroleum and fuel pricing
- Oil Marketing Companies (OMCs)
- Refineries
- Gas / LNG / LPG
- Power and electricity
- Energy policy and regulation

Selected categories:
{", ".join(selected_categories)}

Allowed categories:
{", ".join(CATEGORIES)}

Relevance rules:
- Relevant stories must materially concern Pakistan's petroleum/fuel market, OMCs,
  refineries, gas/LNG/LPG, electricity/power, or energy policy/regulation.
- International oil/LNG stories are relevant only when the supplied text itself gives
  a clear connection or potential implication for Pakistan.
- Do not invent facts or rely on unstated background knowledge.

Priority rules:
- High: immediate pricing, supply, regulatory, financial, or major commercial consequence.
- Medium: meaningful company, project, sector, or policy development.
- General: useful context with lower immediate impact.

For each article:
- relevant: true/false
- category: one allowed category
- priority: High, Medium, or General
- short_headline: maximum 12 words
- summary: maximum 2 concise sentences
- why_it_matters: maximum 1 concise sentence

If an impact is an inference, use cautious wording such as "may" or "could".
"""

        schema = {
            "name": "energy_news_brief",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "stories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "relevant": {"type": "boolean"},
                                "category": {
                                    "type": "string",
                                    "enum": CATEGORIES,
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["High", "Medium", "General"],
                                },
                                "short_headline": {"type": "string"},
                                "summary": {"type": "string"},
                                "why_it_matters": {"type": "string"},
                            },
                            "required": [
                                "id",
                                "relevant",
                                "category",
                                "priority",
                                "short_headline",
                                "summary",
                                "why_it_matters",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["stories"],
                "additionalProperties": False,
            },
        }

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        instructions
                        + "\n\nNEWSPAPER ITEMS:\n"
                        + json.dumps(payload, ensure_ascii=False)
                    ),
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
            temperature=0.1,
            max_completion_tokens=2400,
            extra_body={
                "reasoning_effort": "low",
                "reasoning_format": "hidden",
            },
        )

        content = response.choices[0].message.content or '{"stories":[]}'
        parsed = json.loads(content)
        results = parsed.get("stories", [])

        by_id = {
            int(item["id"]): item
            for item in results
            if isinstance(item, dict) and "id" in item
        }

        output = []
        for idx, article in enumerate(articles):
            decision = by_id.get(idx)

            # Strict schema should return every story, but fail safely if one is absent.
            if decision is None:
                continue

            item = dict(article)
            item["relevant"] = decision["relevant"]
            item["category"] = decision["category"]
            item["priority"] = decision["priority"]
            item["short_headline"] = (
                decision["short_headline"] or item.get("title", "")
            )
            item["summary"] = (
                decision["summary"]
                or item.get("description")
                or item.get("listing_snippet", "")
            )
            item["why_it_matters"] = decision["why_it_matters"]

            output.append(item)

        return output
