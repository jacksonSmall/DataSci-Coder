#!/usr/bin/env python3
"""Step 1: Clean and structure Daily Dose of DS email data.

Cleans raw email JSON → structured, section-split JSON suitable for
training data generation.
"""

import json
import re
import base64
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).parent
EMAILS_DIR = BASE_DIR / "data" / "emails"
CLEAN_DIR = BASE_DIR / "data" / "clean_emails"

# ── Unicode cleaning ────────────────────────────────────────────────
# Invisible / zero-width characters to strip
INVISIBLE_RE = re.compile(
    "["
    "\u00ad"          # soft hyphen
    "\u034f"          # combining grapheme joiner
    "\u061c"          # arabic letter mark
    "\u115f\u1160"    # hangul fillers
    "\u180e"          # mongolian vowel separator
    "\u200b-\u200f"   # zero-width spaces, joiners, marks
    "\u202a-\u202e"   # bidi overrides
    "\u2060-\u2064"   # invisible operators
    "\u2066-\u2069"   # bidi isolates
    "\u206a-\u206f"   # deprecated formatting
    "\ufeff"          # BOM / ZWNBSP
    "\ufff9-\ufffb"   # interlinear annotations
    "\U000e0001"      # language tag
    "\U000e0020-\U000e007f"  # tag space–tilde
    "͏"               # U+034F again (literal)
    "]+"
)

# Leading emoji / bullet-point decoration to strip from lines
LEADING_EMOJI_RE = re.compile(
    r"^[\U0001f300-\U0001f9ff\u2600-\u26ff\u2700-\u27bf"
    r"\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    r"\u2192\u2190-\u21ff"  # arrows
    r"]+\s*"
)

# ── Section detection ───────────────────────────────────────────────
# Canonical section markers (case-insensitive match)
SECTION_MARKERS = {
    "TODAY'S ISSUE", "TODAY'S TOPICS",
    "MACHINE LEARNING", "DEEP LEARNING",
    "LLMS", "NLP", "COMPUTER VISION",
    "MLOPS", "DATA ENGINEERING", "STATISTICS", "PYTHON",
    "GENERATIVE AI", "OPEN SOURCE",
    "HANDS-ON", "VISUAL EXPLAINERS",
    "MCP", "RAG", "AGENTS", "AGENTIC AI",
    "REINFORCEMENT LEARNING", "GRAPH ML",
}

# Footer markers — everything from here onward is boilerplate
FOOTER_MARKERS = [
    "THAT'S A WRAP",
    "NO-FLUFF DS/ML RESOURCES TO",
    "SPONSOR US",
    "Partner with US",
    "ADVERTISE TO",
]

# ── Link / URL cleaning ────────────────────────────────────────────
# Footer link texts to skip (lowercased)
FOOTER_LINK_TEXTS = {
    "succeed in ai engineering roles",
    "master full-stack ai engineering",
    "succeed in ds/ml roles",
    "develop industry ml skills",
    "course with 18 parts",
    "course with 9 parts",
    "crash course with 9 parts →",
    "this course with 14 parts",
    "a crash course with 14 parts",
    "this course", "this crash course",
    "here", "quantization techniques",
    "conformal predictions", "practical guide",
    "test new models in production",
    "federated learning", "compress ml models",
    "update your profile", "unsubscribe",
    "premium ds/ml resources",
    "model compression", "industry ml guides",
}

SKIP_LINK_DOMAINS = {
    "unsubscribe", "twitter.com", "linkedin.com/share",
    "facebook.com", "mailto:", "list-manage.com",
    "tracking", "click.convertkit", "open.substack",
    "instagram.com", "youtube.com",
}

# ── Image filtering ─────────────────────────────────────────────────
TRACKING_IMAGE_DOMAINS = [
    "open.convertkit", "click.convertkit",
    "beacon.", "tracking.", "pixel.", "t.co",
]

# ── Article noise patterns ──────────────────────────────────────────
ARTICLE_NOISE_RE = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in [
        r"Accept all cookies.*?cookie policy",
        r"We use cookies.*?\.(?:\s|$)",
        r"This site uses cookies.*?\.(?:\s|$)",
        r"Skip to (?:main )?content",
        r"Sign up for our newsletter.*?$",
        r"Subscribe.*?newsletter",
        r"Share this (?:article|post).*?$",
        r"Follow us on.*?$",
        r"Privacy Policy.*?Terms of Service",
        r"©\s*\d{4}.*?(?:All rights reserved|Inc\.|LLC).*?$",
        r"Join\s+\d+[\w,]*\s+(?:readers|subscribers).*?$",
    ]
]

# ── Category detection ──────────────────────────────────────────────
# More precise keywords to avoid everything landing in "llm"
CATEGORY_KEYWORDS = {
    "agents": [
        "agentic", "ai agent", "multi-agent", "crew", "autogen",
        "langgraph", "tool use", "function calling", "a2a",
        "ag-ui", "mcp server", "mcp client",
    ],
    "rag": [
        "retrieval augmented", "retrieval-augmented", " rag ",
        "vector database", "vector db", "chromadb", "pinecone",
        "weaviate", "knowledge base", "chunking strateg",
        "embedding model", "reranking", "reranker",
    ],
    "llm": [
        "large language model", "fine-tun", "fine tun", "qlora",
        "lora ", "quantiz", "gguf", "ggml", "llama", "mistral",
        "gemma", "qwen", "gpt-4", "claude", "ollama",
        "inference optim", "kv cache", "speculative decod",
        "context window", "token limit",
    ],
    "prompt_engineering": [
        "prompt engineer", "chain of thought", "few-shot",
        "zero-shot", "system prompt", "prompt template",
    ],
    "mcp": [
        "model context protocol", " mcp ", "mcp server",
        "mcp client",
    ],
    "deep_learning": [
        "neural network", "backpropagation", "dropout",
        "batch norm", "activation function", "cnn ", "rnn ",
        "lstm", "gru ", "autoencoder", "diffusion model",
        "attention mechanism", "transformer architecture",
    ],
    "nlp": [
        "natural language", "text classification", "sentiment",
        "named entity", "tokeniz", "bert ", "word2vec",
        "text embedding", "text generation",
    ],
    "computer_vision": [
        "computer vision", "object detection", "yolo",
        "image segmentation", "image classification",
        "convolution", "resnet", "vision transformer",
    ],
    "machine_learning": [
        "random forest", "gradient boost", "xgboost",
        "lightgbm", "catboost", "decision tree",
        "logistic regression", "linear regression",
        "support vector", "k-nearest", "feature engineer",
        "cross-validation", "hyperparameter", "sklearn",
        "scikit-learn", "ensemble",
    ],
    "mlops": [
        "mlops", "model deployment", "model serving",
        "docker", "kubernetes", "ci/cd", "model monitor",
        "feature store", "ml pipeline", "model registry",
        "a/b test",
    ],
    "data_engineering": [
        "data engineer", "etl ", "spark ", "kafka ",
        "airflow", "data pipeline", "data warehouse",
        "data lake", "dbt ", "batch processing",
        "stream processing",
    ],
    "statistics": [
        "hypothesis test", "p-value", "confidence interval",
        "bayesian", "distribution", "statistical",
        "causal inference", "a/b test", "conformal predict",
        "probability",
    ],
    "python": [
        "pandas", "numpy", "matplotlib", "seaborn",
        "polars", "python tip", "pythonic",
    ],
    "graph_ml": [
        "graph neural", "gnn", "graph ml", "networkx",
        "node2vec", "graph attention", "knowledge graph",
        "graphsage",
    ],
}


def decode_convertkit_url(tracking_url: str) -> str:
    """Decode ConvertKit tracking URL → real destination.

    ConvertKit URLs encode the real URL as base64 in the last path segment.
    """
    if "convertkit" not in tracking_url:
        return tracking_url
    parsed = urlparse(tracking_url)
    segments = parsed.path.rstrip("/").split("/")
    if not segments:
        return tracking_url
    last = segments[-1]
    try:
        padded = last + "=" * (-len(last) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
        if decoded.startswith("http"):
            return decoded
    except Exception:
        pass
    return tracking_url


def clean_text(text: str) -> str:
    """Remove invisible chars, preheader padding, normalize whitespace."""
    text = INVISIBLE_RE.sub("", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped or all(c in " \t\u00a0" for c in stripped):
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(stripped)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_leading_emoji(text: str) -> str:
    """Remove leading emoji / decorative Unicode from each line."""
    lines = text.split("\n")
    out = []
    for line in lines:
        out.append(LEADING_EMOJI_RE.sub("", line))
    return "\n".join(out)


def strip_footer(text: str) -> str:
    """Remove footer boilerplate from THAT'S A WRAP onward."""
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
            break
    return text


def is_section_header(line: str) -> str | None:
    """Return canonical section name if line is a section header, else None.

    Avoids false positives on single-letter lines (math variables),
    short numbers, or formulas.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Must be short (section headers are 1-4 words typically)
    if len(stripped) > 60:
        return None

    upper = stripped.upper()

    # Exact match against known markers
    for marker in SECTION_MARKERS:
        if upper == marker:
            return marker

    # Reject single characters, numbers, formulas, math notation
    if len(stripped) <= 2:
        return None
    if re.match(r"^[\d\.\-\+\*/=<>~%()]+$", stripped):
        return None
    if re.match(r"^[A-Z]\(", stripped):  # P(A|B) etc.
        return None
    if re.match(r"^\[.*\]$", stripped):  # [CLS] etc.
        return None
    if re.match(r"^~?\d+", stripped):  # ~3GB etc.
        return None
    if re.match(r"^\(.*\)$", stripped):  # (G-1) etc.
        return None

    return None


def split_into_sections(text: str) -> list[dict]:
    """Split email body into sections using newsletter markers."""
    sections = []
    current_title = "intro"
    current_content = []

    for line in text.split("\n"):
        header = is_section_header(line)
        if header:
            content = "\n".join(current_content).strip()
            if content:
                sections.append({"title": current_title, "content": content})
            current_title = header
            current_content = []
        else:
            current_content.append(line)

    content = "\n".join(current_content).strip()
    if content:
        sections.append({"title": current_title, "content": content})

    return sections


def deduplicate_links(links: list[dict]) -> list[dict]:
    """Deduplicate by decoded URL, skip footer/promo links."""
    seen = set()
    unique = []
    for link in links:
        text = link.get("text", "").strip()
        url = link["url"]

        # Skip footer link texts
        if text.lower() in FOOTER_LINK_TEXTS:
            continue

        # Skip tracking / social domains
        if any(d in url.lower() for d in SKIP_LINK_DOMAINS):
            continue

        decoded = decode_convertkit_url(url)

        # Skip empty-text links (tracking pixels in <a> tags)
        if not text:
            continue

        if decoded in seen:
            continue
        seen.add(decoded)

        # Clean arrow artifacts from link text
        clean_text_val = text.rstrip(" →").strip()
        if not clean_text_val:
            continue

        entry = {"url": decoded, "text": clean_text_val}
        if decoded != url:
            entry["original_url"] = url
        unique.append(entry)

    return unique


def filter_images(images: list[dict]) -> list[dict]:
    """Remove tracking pixels and decorative images."""
    filtered = []
    for img in images:
        url = img.get("url", "")
        local_path = img.get("local_path", "")

        # Skip tracking domains
        if any(d in url for d in TRACKING_IMAGE_DOMAINS):
            continue

        # Skip tiny files (tracking pixels)
        if local_path:
            p = Path(local_path)
            if p.exists() and p.stat().st_size < 200:
                continue

        filtered.append(img)
    return filtered


def clean_article(article: dict) -> dict | None:
    """Clean scraped article: remove nav/cookie remnants, validate length."""
    content = article.get("content", "")
    if not content:
        return None

    for pattern in ARTICLE_NOISE_RE:
        content = pattern.sub("", content)

    content = clean_text(content)
    content = strip_leading_emoji(content)

    # Too short after cleaning → not useful
    if len(content) < 200:
        return None

    url = article.get("url", "")
    link_text = article.get("link_text", "").strip().rstrip(" →").strip()

    return {
        "url": decode_convertkit_url(url),
        "link_text": link_text,
        "content": content,
    }


def detect_categories(subject: str, text: str) -> list[str]:
    """Detect topic categories with precise, non-overlapping keywords.

    Returns categories sorted by match confidence (most keyword hits first).
    """
    combined = f" {subject} {text} ".lower()
    scores = {}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in combined)
        if hits > 0:
            scores[cat] = hits

    if not scores:
        return ["general"]

    # Return sorted by hit count descending
    return [cat for cat, _ in sorted(scores.items(), key=lambda x: -x[1])]


def process_email(email_path: Path) -> dict | None:
    """Process a single raw email JSON → cleaned, structured format."""
    raw = json.loads(email_path.read_text())

    email_text = raw.get("email_text", "")
    if not email_text or len(email_text) < 100:
        return None

    cleaned = clean_text(email_text)
    cleaned = strip_footer(cleaned)
    cleaned = strip_leading_emoji(cleaned)

    if len(cleaned) < 50:
        return None

    sections = split_into_sections(cleaned)
    links = deduplicate_links(raw.get("links", []))
    images = filter_images(raw.get("images", []))

    articles = []
    for article in raw.get("articles", []):
        cleaned_article = clean_article(article)
        if cleaned_article:
            articles.append(cleaned_article)

    subject = raw.get("subject", "")
    categories = detect_categories(subject, cleaned)

    return {
        "id": raw["id"],
        "subject": subject,
        "date": raw.get("date", ""),
        "categories": categories,
        "full_text": cleaned,
        "sections": sections,
        "links": links,
        "images": images,
        "articles": articles,
    }


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    email_files = sorted(EMAILS_DIR.glob("*.json"))
    print(f"Found {len(email_files)} raw emails")

    stats = {"total": 0, "cleaned": 0, "skipped": 0, "categories": {}}
    for email_file in email_files:
        stats["total"] += 1
        result = process_email(email_file)
        if result is None:
            stats["skipped"] += 1
            continue

        stats["cleaned"] += 1
        for cat in result["categories"]:
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

        out_path = CLEAN_DIR / email_file.name
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\nCleaning complete:")
    print(f"  Total: {stats['total']}")
    print(f"  Cleaned: {stats['cleaned']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"\nCategories:")
    for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    (CLEAN_DIR / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
