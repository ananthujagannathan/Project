
import os
import time
import json
import re
import hashlib
import urllib.parse
import urllib.request

import streamlit as st
import trafilatura

from dotenv import load_dotenv
from google import genai
from google.genai import types


# -------------------------------
# Configuration
# -------------------------------

st.set_page_config(
    page_title="VerifAI",
    page_icon="🔍",
    layout="wide"
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")

MODEL = "gemini-3.6-flash"
MAX_RETRIES = 1
COOLDOWN_SECONDS = 10


# -------------------------------
# Identify temporary Gemini errors
# -------------------------------

def is_temporary_error(error):

    error_message = str(error)

    status_code = (
        getattr(error, "status_code", None)
        or getattr(error, "code", None)
    )

    return (
        status_code in (503, 504)
        or "503" in error_message
        or "504" in error_message
        or "UNAVAILABLE" in error_message
        or "DEADLINE_EXCEEDED" in error_message
    )


# -------------------------------
# Level 6: Extract article from URL
# -------------------------------

def extract_article_text(url):

    parsed_url = urllib.parse.urlparse(url.strip())

    if (
        parsed_url.scheme not in ("http", "https")
        or not parsed_url.netloc
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
    ):
        raise ValueError(
            "Please enter a valid public HTTP or HTTPS article URL."
        )

    request = urllib.request.Request(
        url.strip(),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; VerifAI/1.0)"
            )
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=20
    ) as response:

        final_url = urllib.parse.urlparse(response.geturl())

        if (
            final_url.scheme not in ("http", "https")
            or not final_url.hostname
        ):
            raise ValueError(
                "The website redirected to an invalid address."
            )

        html = response.read(2 * 1024 * 1024)

        charset = response.headers.get_content_charset() or "utf-8"

        html = html.decode(
            charset,
            errors="replace"
        )

    article_text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True
    )

    if not article_text or len(article_text.strip()) < 100:
        raise ValueError(
            "VerifAI could not extract enough article text. "
            "The website may block automated access or "
            "the page may not contain a readable article."
        )

    return article_text.strip()[:12000]


# -------------------------------
# Custom styling
# -------------------------------

st.markdown("""
<style>
.stApp {
    background: #FFFFFF;
    color: #000000;
}

header {
    background: #FFFFFF;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    margin-left: 0 !important;
    margin-right: auto !important;
}

h1, h2, h3 {
    color: #5425A8 !important;
    text-align: left !important;
    font-weight: 800 !important;
}

p, label {
    color: #000000 !important;
    font-weight: 600 !important;
    text-align: left !important;
}

h1 {
    font-size: 55px !important;
    font-weight: 800 !important;
    letter-spacing: -2px;
    margin-bottom: 30px !important;
}

.stTextArea textarea,
.stTextInput input {
    background: #FAF8FF !important;
    color: #000000 !important;
    border: 1px solid #D8C5F5 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #777777 !important;
    font-weight: 500 !important;
}

.stButton button {
    background: #7138C8 !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    padding: 10px 25px;
}

.stButton button p,
.stButton button span,
.stButton button div {
    color: #FFFFFF !important;
    font-weight: bold !important;
}

.stButton button:hover {
    background: #5425A8 !important;
}

div[role="radiogroup"] label {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* White expandable sections */

div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D8C5F5 !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] details {
    background-color: #FFFFFF !important;
}

div[data-testid="stExpander"] summary {
    background-color: #FFFFFF !important;
}

div[data-testid="stExpander"] summary p {
    color: #5425A8 !important;
    font-weight: 700 !important;
}

div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    background-color: #FFFFFF !important;
}

div[data-testid="stAlert"],
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Gemini: analysis + claims in ONE call
# -------------------------------

def analyze_news_and_claims(news):

    client = genai.Client(api_key=API_KEY)

    prompt = f"""
You are VerifAI, an AI assistant that helps users
assess news credibility.

Analyse the following news article and return ONE JSON object
with exactly these keys:

{{
  "summary": "Briefly explain what the article claims.",
  "assessment": "One of: Appears credible | Contains questionable claims | Potentially misleading | Insufficient information",
  "reasons": "The main reasons for your assessment.",
  "what_to_check": "What evidence or reliable sources would help verify the claims.",
  "claims": ["short checkable factual claim 1", "claim 2", "claim 3"]
}}

Rules for the analysis:
- Do not invent facts, sources, URLs or evidence.
- Do not claim a story is verified without evidence.
- Clearly mention uncertainty.
- This is a preliminary AI assessment, not a definitive fact-check.
- Keep each text field concise and easy to understand.

Rules for "claims":
- Only factual claims that can be checked. No opinions.
- Keep each claim short and clear, preserving the original meaning.
- Do not invent additional claims.
- Maximum 3 claims. Use an empty list [] if there are none.

Return only the JSON object, with no extra text.

NEWS ARTICLE:
{news}
"""

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                    response_mime_type="application/json",
                    max_output_tokens=1500
                )
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return parse_gemini_json(response.text)

        except Exception as error:

            if is_temporary_error(error) and attempt < MAX_RETRIES:

                st.info(
                    "Gemini is temporarily unavailable. "
                    "Retrying the analysis..."
                )

                time.sleep(2)
                continue

            raise


# -------------------------------
# Parse Gemini JSON
# -------------------------------

def parse_gemini_json(text):

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip()
    )

    data = json.loads(cleaned)

    claims = data.get("claims", [])

    if not isinstance(claims, list):
        claims = []

    claims = [
        str(claim).strip()
        for claim in claims
        if str(claim).strip()
        and str(claim).strip().upper() != "NONE"
    ][:3]

    return {
        "summary": str(data.get("summary", "")).strip(),
        "assessment": str(data.get("assessment", "")).strip(),
        "reasons": str(data.get("reasons", "")).strip(),
        "what_to_check": str(data.get("what_to_check", "")).strip(),
        "claims": claims,
    }


# -------------------------------
# Gemini caching
# -------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def analyze_news_cached(news):

    return analyze_news_and_claims(news)


# -------------------------------
# Google Fact Check API
# -------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def search_fact_checks(claim):

    if not FACT_CHECK_API_KEY:
        raise ValueError(
            "Fact Check API key not found in .env file."
        )

    query = claim.strip()[:1000]

    endpoint = (
        "https://factchecktools.googleapis.com/"
        "v1alpha1/claims:search"
    )

    parameters = {
        "query": query,
        "key": FACT_CHECK_API_KEY,
        "pageSize": 5
    }

    request_url = (
        endpoint + "?" +
        urllib.parse.urlencode(parameters)
    )

    request = urllib.request.Request(request_url)

    with urllib.request.urlopen(
        request,
        timeout=20
    ) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

    return data.get("claims", [])


# -------------------------------
# Display published fact-checks
# -------------------------------

def display_fact_checks(claims):

    if not claims:

        st.info(
            "Not yet verified: No matching published "
            "fact-check was found. This does not mean "
            "the claim is true or false."
        )

        return

    all_reviews = []

    for claim in claims:

        reviews = claim.get("claimReview", [])
        all_reviews.extend(reviews)

    if not all_reviews:

        st.info(
            "A possible matching claim was found, but "
            "publisher review details were unavailable."
        )

        return

    st.success(
        f"Previously fact-checked: Found "
        f"{len(all_reviews)} published review(s)."
    )

    for index, review in enumerate(all_reviews, start=1):

        with st.expander(
            f"Published fact-check {index}: "
            + review.get(
                "title",
                "Read the fact-check article"
            )
        ):

            publisher = review.get(
                "publisher", {}
            ).get("name", "Unknown publisher")

            rating = review.get(
                "textualRating",
                "Rating unavailable"
            )

            title = review.get(
                "title",
                "Fact-check article"
            )

            review_date = review.get(
                "reviewDate",
                "Date unavailable"
            )

            article_url = review.get("url", "")

            st.markdown(f"**Publisher:** {publisher}")
            st.markdown(f"**Publisher's rating:** {rating}")
            st.markdown(f"**Review date:** {review_date}")
            st.markdown(f"**Article:** {title}")

            if article_url.startswith("https://"):
                st.link_button(
                    "Read fact-check article",
                    article_url
                )

            st.divider()

    st.caption(
        "These ratings belong to the original publishers. "
        "VerifAI displays their findings but does not "
        "independently confirm them."
    )


# -------------------------------
# Level 7: Final Credibility Report
# -------------------------------

def display_credibility_report(
    result,
    fact_check_results,
    article_url=None
):

    st.divider()

    st.markdown("## 📄 VerifAI Credibility Report")

    st.caption(
        "A structured report combining AI analysis "
        "and available published fact-check findings."
    )

    # 1. Article information

    st.markdown("### 1. Article Information")

    if article_url:
        st.markdown(
            f"[Open original news article]({article_url})"
        )
    else:
        st.write("Source: Article text provided by the user.")

    # 2. Article summary

    st.markdown("### 2. Article Summary")

    st.write(
        result.get("summary") or "Summary unavailable."
    )

    # 3. AI assessment

    st.markdown("### 3. Overall AI Assessment")

    st.info(
        result.get("assessment") or "Assessment unavailable."
    )

    # 4. Reasons

    st.markdown("### 4. Reasons Behind the Assessment")

    st.write(
        result.get("reasons") or "No reasons provided."
    )

    # 5. Factual claims

    st.markdown("### 5. Key Factual Claims")

    extracted_claims = result.get("claims", [])

    if extracted_claims:

        for index, claim in enumerate(
            extracted_claims,
            start=1
        ):

            st.markdown(f"**Claim {index}:** {claim}")

    else:

        st.write("No factual claims were extracted.")

    # 6. Published fact-check findings

    st.markdown("### 6. Published Fact-Check Findings")

    if not FACT_CHECK_API_KEY:

        st.info(
            "Fact-check search was not performed because "
            "the Fact Check API key is unavailable."
        )

    elif not extracted_claims:

        st.info(
            "There were no extracted claims to search."
        )

    else:

        for item in fact_check_results:

            st.markdown(f"**Claim:** {item['claim']}")

            if item["error"]:

                st.warning(
                    "Search unavailable for this claim."
                )

                continue

            matches = item["matches"]

            if not matches:

                st.info(
                    "Not yet verified: No matching published "
                    "fact-check was found. This does not mean "
                    "the claim is true or false."
                )

                continue

            review_count = 0

            for match in matches:

                reviews = match.get("claimReview", [])

                for review in reviews:

                    review_count += 1

                    publisher = review.get(
                        "publisher", {}
                    ).get("name", "Unknown publisher")

                    rating = review.get(
                        "textualRating",
                        "Rating unavailable"
                    )

                    title = review.get(
                        "title",
                        "Published fact-check"
                    )

                    review_url = review.get("url", "")

                    st.markdown(
                        f"- **Publisher:** {publisher}"
                    )

                    st.markdown(
                        f"  **Publisher's rating:** {rating}"
                    )

                    st.write(title)

                    if review_url.startswith("https://"):

                        st.markdown(
                            f"[Read original fact-check]({review_url})"
                        )

            if review_count == 0:

                st.info(
                    "A possible matching claim was found, "
                    "but publisher review details were unavailable."
                )

    # 7. Further verification

    st.markdown("### 7. What Should Readers Verify Further?")

    st.write(
        result.get("what_to_check")
        or "No additional verification points provided."
    )

    # 8. Limitations

    st.markdown("### 8. Limitations")

    st.markdown("""
    - AI-generated assessments may contain errors.
    - The AI assessment is preliminary and is not a definitive fact-check.
    - Published fact-checks represent the findings of their original publishers.
    - A missing fact-check does not mean a claim is true or false.
    - Some claims may require additional research and reliable evidence.
    - VerifAI does not guarantee the truth or accuracy of a news article.
    """)

    st.divider()

    st.caption(
        "VerifAI is an AI-assisted news credibility tool. "
        "Always consult reliable sources before reaching a conclusion."
    )


# -------------------------------
# Cooldown helper
# -------------------------------

def news_key(news):

    return hashlib.sha256(
        news.strip().encode("utf-8")
    ).hexdigest()


def check_cooldown(news):

    seen = st.session_state.setdefault(
        "seen_news", set()
    )

    if news_key(news) in seen:
        return 0

    last_call = st.session_state.get(
        "last_api_call", 0
    )

    remaining = (
        COOLDOWN_SECONDS -
        (time.time() - last_call)
    )

    return (
        max(0, int(remaining) + 1)
        if remaining > 0
        else 0
    )


def mark_call(news):

    st.session_state["last_api_call"] = time.time()

    st.session_state.setdefault(
        "seen_news", set()
    ).add(news_key(news))


# -------------------------------
# Run the full news analysis
# -------------------------------

def run_news_analysis(news, article_url=None):

    if not API_KEY:

        st.error(
            "Gemini API key not found. "
            "Please check your .env file."
        )

        return

    wait = check_cooldown(news)

    if wait:

        st.warning(
            f"Please wait {wait} more second(s) "
            "before analysing another article. "
            "This protects your Gemini quota."
        )

        return

    try:

        with st.spinner(
            "VerifAI is analysing your news..."
        ):

            result = analyze_news_cached(news)

        mark_call(news)

        st.success(
            "AI analysis completed successfully!"
        )

        # AI analysis

        st.markdown("### VerifAI Analysis")

        st.markdown("**1. SUMMARY**")
        st.markdown(result["summary"] or "Not provided.")

        st.markdown("**2. CREDIBILITY ASSESSMENT**")
        st.markdown(
            result["assessment"] or "Not provided."
        )

        st.markdown("**3. REASONS**")
        st.markdown(result["reasons"] or "Not provided.")

        st.markdown("**4. WHAT TO CHECK**")
        st.markdown(
            result["what_to_check"] or "Not provided."
        )

        st.caption(
            "This is a preliminary AI assessment, "
            "not a definitive fact-check."
        )

        # Individual claims

        st.divider()

        st.markdown("### Individual Claim Analysis")

        extracted_claims = result["claims"]

        fact_check_results = []

        if not extracted_claims:

            st.info(
                "No individual factual claims "
                "could be extracted from this article."
            )

        else:

            st.markdown("### Extracted Factual Claims")

            for index, claim in enumerate(
                extracted_claims,
                start=1
            ):

                st.markdown(
                    f"**Claim {index}:** {claim}"
                )

            # Published fact-checks

            if FACT_CHECK_API_KEY:

                st.divider()

                st.markdown(
                    "### Evidence-Based Claim Assessment"
                )

                st.caption(
                    "VerifAI searches for previously "
                    "published fact-checks for each claim. "
                    "Publisher ratings are displayed as "
                    "reported, not independently verified."
                )

                for index, claim in enumerate(
                    extracted_claims,
                    start=1
                ):

                    st.markdown(f"#### Claim {index}")
                    st.markdown(f"**{claim}**")

                    try:

                        with st.spinner(
                            f"Searching published fact-checks "
                            f"for Claim {index}..."
                        ):

                            fact_results = search_fact_checks(
                                claim
                            )

                        fact_check_results.append({
                            "claim": claim,
                            "matches": fact_results,
                            "error": None
                        })

                        display_fact_checks(fact_results)

                    except Exception as fact_error:

                        fact_check_results.append({
                            "claim": claim,
                            "matches": [],
                            "error": str(fact_error)
                        })

                        st.warning(
                            "Search unavailable: VerifAI "
                            "could not retrieve fact-check "
                            f"results for Claim {index}."
                        )

                        st.caption(
                            "Technical details: "
                            + str(fact_error)[:500]
                        )

            else:

                st.warning(
                    "Fact Check API key not found. "
                    "Only  analysis is available."
                )

        # Level 7: Final report

        display_credibility_report(
            result,
            fact_check_results,
            article_url
        )

    except Exception as error:

        error_message = str(error)

        status_code = (
            getattr(error, "status_code", None)
            or getattr(error, "code", None)
        )

        if (
            status_code == 429
            or "429" in error_message
        ):

            st.error(
                " Rrequest limit has been "
                "reached. Please wait before trying again."
            )

            st.code(error_message[:1500])

        elif (
            status_code == 403
            or "403" in error_message
        ):

            st.error(
                " Access was denied. "
                "Please check your API key and API access."
            )

        elif isinstance(error, json.JSONDecodeError):

            st.error(
                "Gemini returned a response in an "
                "unexpected format. Please try again."
            )

        elif is_temporary_error(error):

            st.error(
                " VerifAI is temporarily unavailable "
                "or taking too long to respond. "
                "Please try again shortly."
            )

            st.caption(
                "Technical details: "
                + error_message[:1000]
            )

        else:

            st.error(
                "Something went wrong while "
                "analysing your news."
            )

            st.caption(
                "Technical details: "
                + error_message[:1000]
            )


# -------------------------------
# Main heading
# -------------------------------

st.markdown(
    "<h1 style='color:#5425A8; text-align:left;'>"
    "VerifAI</h1>",
    unsafe_allow_html=True
)

st.divider()


# -------------------------------
# News analysis section
# -------------------------------

st.markdown(
    "<h2 style='color:#5425A8; text-align:left;'>"
    "🔍 Analyze a News Article</h2>",
    unsafe_allow_html=True
)

option = st.radio(
    "Choose your input method:",
    ["Paste News", "News URL"],
    horizontal=True
)


# -------------------------------
# Paste News option
# -------------------------------

if option == "Paste News":

    news = st.text_area(
        "Paste your news article here:",
        height=200,
        max_chars=12000,
        placeholder="Paste or type your news article..."
    )

    if st.button("🔍 Analyze News"):

        if news.strip():

            run_news_analysis(news.strip())

        else:

            st.warning(
                "Please enter a news article."
            )


# -------------------------------
# News URL option
# -------------------------------

else:

    url = st.text_input(
        "Enter news article URL:",
        placeholder="https://example.com/news"
    )

    if st.button("🌐 Extract Article Text"):

        if not url.strip():

            st.warning(
                "Please enter a news article URL."
            )

        else:

            try:

                with st.spinner(
                    "VerifAI is retrieving the article..."
                ):

                    extracted_text = extract_article_text(
                        url.strip()
                    )

                st.session_state["extracted_article_text"] = (
                    extracted_text
                )

                st.session_state["extracted_article_url"] = (
                    url.strip()
                )

                st.success(
                    "Article text extracted successfully!"
                )

            except Exception as error:

                st.session_state.pop(
                    "extracted_article_text",
                    None
                )

                st.session_state.pop(
                    "extracted_article_url",
                    None
                )

                st.error(
                    "VerifAI could not extract the article."
                )

                st.caption(
                    "Technical details: "
                    + str(error)[:1000]
                )

    # Show extracted article if it belongs to current URL

    saved_url = st.session_state.get(
        "extracted_article_url", ""
    )

    saved_text = st.session_state.get(
        "extracted_article_text", ""
    )

    if saved_text and saved_url == url.strip():

        st.markdown("### Extracted Article Preview")

        st.caption(
            f"Extracted {len(saved_text)} characters."
        )

        with st.expander(
            "View extracted article text",
            expanded=True
        ):

            st.markdown(saved_text)

        st.caption(
            "You can now analyse this extracted article. "
            "AI analysis requires available Gemini quota."
        )

        if st.button("🔍 Analyze Extracted Article"):

            run_news_analysis(
                saved_text,
                saved_url
            )

    elif saved_text and saved_url != url.strip():

        st.info(
            "The URL has changed. Extract the new article "
            "before analysing it."
        )
