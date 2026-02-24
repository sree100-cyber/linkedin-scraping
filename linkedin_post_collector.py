import streamlit as st
import requests
import json
import re
import time
import google.generativeai as genai
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Lead Collector",
    page_icon="🎯",
    layout="wide"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def scrape_post_text(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        meta = soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return meta["content"]

        return soup.get_text(separator=" ", strip=True)[:3000]
    except:
        return ""


def search_linkedin_posts(serpapi_key: str, phrase: str, max_results: int):
    query = f'site:linkedin.com/posts "{phrase}" OR {phrase}'

    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key,
        "num": max_results,
        "hl": "en",
    }

    r = requests.get("https://serpapi.com/search", params=params)
    data = r.json()

    results = []
    for item in data.get("organic_results", []):
        link = item.get("link", "")
        if "linkedin.com" in link:
            results.append({
                "url": link,
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", "")
            })
    return results


def ai_relevance_score(gemini_key, example_text, phrase, candidate_text, model):
    genai.configure(api_key=gemini_key)
    gmodel = genai.GenerativeModel(model)

    prompt = f"""
You are a B2B lead generation signal detector.

Reference post:
\"\"\"{example_text[:800]}\"\"\"

Target theme: "{phrase}"

Candidate LinkedIn post:
\"\"\"{candidate_text[:1000]}\"\"\"

Score from 0–100 based on:

1. Topic relevance
2. Signs of active problem
3. Buying intent
4. Decision-maker language
5. Urgency signals

Return ONLY valid JSON:

{{
  "relevance_score": <0-100>,
  "reason": "<why this is or isn't a good lead>",
  "key_match": "<intent signal detected>"
}}
"""
    try:
        resp = gmodel.generate_content(prompt)
        raw = re.sub(r"```json|```", "", resp.text.strip())
        return json.loads(raw)
    except:
        return {"relevance_score": 0, "reason": "AI error", "key_match": ""}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙ Configuration")

    serpapi_key = st.text_input("SerpAPI Key", type="password")
    gemini_key  = st.text_input("Gemini API Key", type="password")

    st.markdown("### Tuning")

    ai_threshold = st.slider("Min AI relevance score", 0, 100, 45)
    max_results  = st.slider("Max posts to fetch", 10, 50, 30)

    model = st.selectbox("Gemini Model", [
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro-latest",
        "models/gemini-2.0-flash"
    ])


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.title("🎯 LinkedIn Lead Collector")

example_url = st.text_input("Example Post URL")
target_phrase = st.text_input("Target Phrase / Theme")

run_btn = st.button("▶ RUN LEAD COLLECTION")

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

if run_btn:

    if not serpapi_key or not gemini_key or not example_url or not target_phrase:
        st.error("All fields are required.")
        st.stop()

    progress = st.progress(0)

    # STEP 1: Scrape example
    example_text = scrape_post_text(example_url)
    progress.progress(20)

    # STEP 2: Search LinkedIn
    raw_posts = search_linkedin_posts(serpapi_key, target_phrase, max_results)
    progress.progress(40)

    # STEP 3: AI Scoring
    scored_posts = []

    for i, post in enumerate(raw_posts):
        full_text = scrape_post_text(post["url"])
        content = full_text if full_text else post["title"] + " " + post["snippet"]

        ai_result = ai_relevance_score(
            gemini_key,
            example_text,
            target_phrase,
            content,
            model
        )

        post["ai_score"]  = ai_result.get("relevance_score", 0)
        post["reason"]    = ai_result.get("reason", "")
        post["key_match"] = ai_result.get("key_match", "")

        scored_posts.append(post)

        progress.progress(40 + int((i+1)/len(raw_posts)*40))
        time.sleep(0.2)

    progress.progress(100)

    # FILTERING
    qualified_posts = [p for p in scored_posts if p["ai_score"] >= ai_threshold]
    high_intent = [p for p in scored_posts if p["ai_score"] >= 80]

    # ─────────────────────────────
    # RESULTS
    # ─────────────────────────────

    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Posts Found", len(raw_posts))
    col2.metric("Qualified Leads", len(qualified_posts))
    col3.metric("🔥 High Intent (80+)", len(high_intent))

    # 🔗 ALWAYS SHOW ALL POSTS
    st.subheader("🔗 All Found Posts")

    for post in scored_posts:
        st.markdown(
            f"- **AI {post['ai_score']}** → [{post['url']}]({post['url']})"
        )

    # 🎯 Qualified Section
    if qualified_posts:
        st.subheader("🎯 Qualified Leads")

        for post in sorted(qualified_posts, key=lambda x: x["ai_score"], reverse=True):
            st.markdown(f"""
---
**AI Score:** {post['ai_score']}  
**URL:** {post['url']}  

**Why:** {post['reason']}  
**Signal:** {post['key_match']}
""")

        df = pd.DataFrame(qualified_posts)
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Qualified Leads CSV",
            csv,
            f"linkedin_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    else:
        st.warning("No posts met AI threshold. Try lowering to 35–40.")