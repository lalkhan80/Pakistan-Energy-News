# Pakistan Energy News Agent

A Streamlit + CrewAI application that reads approved Pakistani newspaper pages and produces a short AI-curated brief focused on:

- Petroleum and fuel pricing
- Oil Marketing Companies (OMCs)
- Refineries
- Gas / LNG / LPG
- Power and electricity
- Energy policy and regulation

## Approved sources

1. Dawn — https://www.dawn.com/
2. Business Recorder — https://www.brecorder.com/
3. The News International — https://www.thenews.com.pk/
4. The Express Tribune — https://tribune.com.pk/

The app uses normal Python code to collect and deduplicate stories, then uses a CrewAI energy editor powered by Groq `openai/gpt-oss-20b` to decide relevance, assign category/priority, summarize the story, and explain why it matters.

## Files

- `app.py` — Streamlit interface
- `sources.py` — approved sources, categories and energy vocabulary
- `news_collector.py` — article discovery, extraction, date filtering and deduplication
- `energy_agent.py` — CrewAI classification and summarization agent
- `requirements.txt` — Python packages for Streamlit Cloud
- `.gitignore` — prevents local secrets from being committed
- `secrets.toml.example` — example secret format

---

# Deploy without VS Code, Terminal or Colab

## Step 1 — Get a Groq API key

Create/sign in to a GroqCloud account and create an API key.

Keep the key private. Do not paste the real key into GitHub files.

## Step 2 — Create a GitHub repository

1. Sign in to GitHub.
2. Click **New repository**.
3. Repository name: `pakistan-energy-news-agent`
4. Select **Private** if you want to keep the project private.
5. Click **Create repository**.

## Step 3 — Upload the project files in your browser

1. Unzip the downloaded project on your computer.
2. Open the new GitHub repository.
3. Click **Add file → Upload files**.
4. Drag these files into the upload page:
   - `app.py`
   - `energy_agent.py`
   - `news_collector.py`
   - `sources.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
   - `secrets.toml.example`
5. Click **Commit changes**.

Do not rename `app.py` or `requirements.txt`.

## Step 4 — Deploy on Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **Create app** / **New app**.
4. Select your GitHub repository.
5. Branch: `main`
6. Main file path: `app.py`
7. Open **Advanced settings** if offered.
8. In **Secrets**, add:

```toml
GROQ_API_KEY = "YOUR_REAL_GROQ_API_KEY"
```

9. Click **Deploy**.

Streamlit will install the packages from `requirements.txt` automatically.

## Step 5 — Use the agent

Choose:

- Last 24 hours / 3 days / 7 days
- News categories
- Priority levels
- Maximum number of stories

Then click **Generate Energy Brief**.

The result shows:

- Priority
- Category
- Source
- Publication time where detectable
- 2–3 sentence summary
- "Why it matters"
- Original article link
- Alternate source links when similar coverage is detected

---

# How the workflow works

```text
Approved newspaper pages
        |
        v
Collect article links
        |
        v
Broad energy pre-filter
        |
        v
Read candidate articles
        |
        v
Remove duplicate stories
        |
        v
CrewAI relevance editor
        |
        +--> Not relevant -> discard
        |
        v
Category + priority
        |
        v
CrewAI summary + why it matters
        |
        v
Streamlit Energy Brief
```

## Broader Scan

The default collector uses a broad energy vocabulary before the AI stage so that the app does not waste API calls on sports, entertainment and unrelated news.

Turn on **Broader scan** if you want AI to inspect more headlines. It is slower and can generate more website requests.

## Important limitations

News websites change their HTML layouts and automated-access policies from time to time. If a publisher changes its layout or blocks requests, that source may temporarily return fewer stories. The app includes a **Source diagnostics** panel so the affected extractor can be identified and updated.

AI summaries should be treated as an intelligence aid, not the authoritative source. The original article link is always provided for verification.

## Recommended next upgrades

- Automatic 7:00 AM daily brief
- Email delivery
- WhatsApp/Telegram delivery
- Historical searchable archive
- Company watchlists such as PSO, APL, PARCO, ARL, PRL and OGDCL
- Alerts for high-priority fuel-price, OGRA, refinery, supply and power-sector developments
- Firecrawl fallback for sites that block normal HTTP extraction
