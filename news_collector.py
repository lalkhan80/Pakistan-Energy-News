from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from rapidfuzz import fuzz

from sources import SOURCES, ENERGY_TERMS, EXCLUDED_PATH_PARTS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PakistanEnergyNewsAgent/1.0; "
        "+https://streamlit.io)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 15


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
        path = re.sub(r"/+$", "", p.path) or "/"
        return urlunparse((p.scheme or "https", p.netloc.lower(), path, "", "", ""))
    except Exception:
        return url


def _same_domain(url: str, base_url: str) -> bool:
    host = urlparse(url).netloc.lower().replace("www.", "")
    base = urlparse(base_url).netloc.lower().replace("www.", "")
    return host == base or host.endswith("." + base)


def _looks_like_article(url: str, anchor_text: str, base_url: str) -> bool:
    if not url.startswith("http") or not _same_domain(url, base_url):
        return False

    lower = url.lower()
    if any(part in lower for part in EXCLUDED_PATH_PARTS):
        return False

    path = urlparse(url).path.strip("/")
    if not path:
        return False

    # Known article URL structures on the approved publishers.
    known_article_markers = ("/news/", "/story/", "/print/", "/latest/")
    if any(marker in lower for marker in known_article_markers):
        return True

    # Generic fallback for article cards whose URL pattern changes.
    words = anchor_text.split()
    return len(words) >= 5 and len(anchor_text) >= 28 and path.count("/") >= 1


def _extract_listing_links(source: dict[str, Any], per_page_limit: int = 60) -> tuple[list[dict], list[str]]:
    found: dict[str, dict] = {}
    errors: list[str] = []

    for listing_url in source["listing_urls"]:
        try:
            r = requests.get(listing_url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            for a in soup.find_all("a", href=True):
                text = _clean_text(a.get_text(" ", strip=True))
                if len(text) < 20:
                    continue

                url = canonical_url(urljoin(listing_url, a["href"]))
                if not _looks_like_article(url, text, source["base_url"]):
                    continue

                # Use nearby card text as an inexpensive snippet.
                parent = a.find_parent(["article", "div", "li"])
                snippet = ""
                if parent:
                    snippet = _clean_text(parent.get_text(" ", strip=True))
                    if snippet == text:
                        snippet = ""
                snippet = snippet[:700]

                if url not in found:
                    found[url] = {
                        "source": source["name"],
                        "title": text[:300],
                        "url": url,
                        "listing_snippet": snippet,
                    }
                if len(found) >= per_page_limit * max(1, len(source["listing_urls"])):
                    break
        except Exception as exc:
            errors.append(f"{source['name']} — {listing_url}: {exc}")

    return list(found.values()), errors


def _keyword_candidate(article: dict) -> bool:
    text = f"{article.get('title', '')} {article.get('listing_snippet', '')}".lower()
    return any(term in text for term in ENERGY_TERMS)


def _parse_json_ld_date(soup: BeautifulSoup) -> datetime | None:
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or tag.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if isinstance(node, dict) and "@graph" in node and isinstance(node["@graph"], list):
                nodes.extend(node["@graph"])

        for node in nodes:
            if not isinstance(node, dict):
                continue
            for key in ("datePublished", "dateModified", "uploadDate"):
                if node.get(key):
                    try:
                        dt = dateparser.parse(str(node[key]))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.astimezone(timezone.utc)
                    except Exception:
                        pass
    return None


def _parse_published_date(soup: BeautifulSoup) -> datetime | None:
    meta_candidates = [
        ("property", "article:published_time"),
        ("name", "article:published_time"),
        ("name", "date"),
        ("name", "pubdate"),
        ("itemprop", "datePublished"),
    ]
    for attr, value in meta_candidates:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            try:
                dt = dateparser.parse(tag["content"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass

    dt = _parse_json_ld_date(soup)
    if dt:
        return dt

    time_tag = soup.find("time")
    if time_tag:
        raw = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)
        if raw:
            try:
                dt = dateparser.parse(raw, fuzzy=True)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return None


def _extract_article_body(soup: BeautifulSoup) -> str:
    # Remove obvious non-editorial elements.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    selectors = [
        "article",
        "[itemprop='articleBody']",
        ".story__content",
        ".story-detail",
        ".story-detail-content",
        ".article-content",
        ".news-detail",
        ".entry-content",
        ".post-content",
    ]

    containers = []
    for selector in selectors:
        try:
            containers.extend(soup.select(selector))
        except Exception:
            pass

    paragraphs = []
    seen = set()

    search_roots = containers if containers else [soup]
    for root in search_roots:
        for p in root.find_all("p"):
            text = _clean_text(p.get_text(" ", strip=True))
            if len(text) < 45:
                continue
            if text in seen:
                continue
            seen.add(text)
            paragraphs.append(text)

    # Keep enough material for a good brief without sending an entire article to the LLM.
    return "\n".join(paragraphs)[:12000]


def hydrate_article(article: dict) -> tuple[dict, str | None]:
    try:
        r = requests.get(article["url"], headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            article["url"] = canonical_url(urljoin(article["url"], canonical["href"]))

        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            article["title"] = _clean_text(og_title["content"])[:300]

        desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", property="og:description"
        )
        article["description"] = _clean_text(desc.get("content") if desc else "")[:1200]
        article["published_at"] = _parse_published_date(soup)
        article["body"] = _extract_article_body(soup)
        return article, None
    except Exception as exc:
        article.setdefault("description", article.get("listing_snippet", ""))
        article.setdefault("published_at", None)
        article.setdefault("body", "")
        return article, f"{article['source']} — {article['url']}: {exc}"


def _within_window(article: dict, window_days: int) -> bool:
    published = article.get("published_at")
    if published is None:
        # Keep unknown-date stories; the AI can still assess them and UI will label the date unknown.
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    return published >= cutoff


def deduplicate_articles(articles: list[dict], threshold: int = 86) -> list[dict]:
    """Cluster very similar headlines and retain alternate source links."""
    output: list[dict] = []

    def norm(title: str) -> str:
        title = re.sub(r"[^a-z0-9\s]", " ", title.lower())
        return _clean_text(title)

    for article in articles:
        title_n = norm(article.get("title", ""))
        if not title_n:
            continue

        matched = None
        for existing in output:
            score = fuzz.token_set_ratio(title_n, norm(existing.get("title", "")))
            if score >= threshold:
                matched = existing
                break

        if matched:
            matched.setdefault("alternate_sources", [])
            if article["url"] != matched["url"]:
                matched["alternate_sources"].append(
                    {
                        "source": article["source"],
                        "url": article["url"],
                        "title": article["title"],
                    }
                )
            # Prefer the version with more extracted body text.
            if len(article.get("body", "")) > len(matched.get("body", "")):
                alternates = matched.get("alternate_sources", [])
                old_primary = {
                    "source": matched["source"],
                    "url": matched["url"],
                    "title": matched["title"],
                }
                article["alternate_sources"] = alternates + [old_primary]
                output[output.index(matched)] = article
        else:
            article["alternate_sources"] = article.get("alternate_sources", [])
            output.append(article)
    return output


def collect_energy_candidates(
    window_days: int = 1,
    broad_scan: bool = False,
    per_source_limit: int = 45,
) -> tuple[list[dict], list[str]]:
    all_links: list[dict] = []
    errors: list[str] = []

    for source in SOURCES:
        links, source_errors = _extract_listing_links(source, per_page_limit=per_source_limit)
        errors.extend(source_errors)
        all_links.extend(links)

    # Exact URL dedupe.
    unique = {}
    for item in all_links:
        unique[item["url"]] = item
    all_links = list(unique.values())

    if not broad_scan:
        all_links = [a for a in all_links if _keyword_candidate(a)]

    hydrated = []
    for item in all_links:
        full, error = hydrate_article(item)
        if error:
            errors.append(error)
        if _within_window(full, window_days):
            hydrated.append(full)

    return deduplicate_articles(hydrated), errors
