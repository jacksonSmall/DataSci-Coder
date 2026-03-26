#!/usr/bin/env python3
"""Step 2: Generate high-quality training data from cleaned emails.

Improvements over v1:
- Strips emoji artifacts from responses
- Cleans instruction text (no →, link junk)
- Better category detection → real stratified split
- Quality filters: min/max length, coherence checks
- Multi-turn conversation pairs
- Distinct templates per content type
- Article pairs only when article text is educational (not promo)
"""

import json
import re
import random
import hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
CLEAN_DIR = BASE_DIR / "data" / "clean_emails"
TRAIN_DIR = BASE_DIR / "data" / "training"

SYSTEM_PROMPT = (
    "You are an expert data science coding assistant. "
    "Respond ONLY with clean, runnable Python code. "
    "Use inline comments for explanation. "
    "No text outside code blocks."
)

# ── Response cleaning ───────────────────────────────────────────────
LEADING_EMOJI_RE = re.compile(
    r"^[\U0001f300-\U0001f9ff\u2600-\u26ff\u2700-\u27bf"
    r"\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    r"\u2192\u2190-\u21ff"
    r"]+\s*"
)

# Math-formatted Unicode chars (bold/italic sans-serif) → ASCII
MATH_UNICODE_RE = re.compile(r"[\U0001d400-\U0001d7ff]")

# ── Instruction templates ──────────────────────────────────────────
EXPLAIN_TEMPLATES = [
    "Explain {topic}.",
    "What is {topic}?",
    "Can you explain {topic} in simple terms?",
    "Tell me about {topic}.",
    "I want to understand {topic}. Can you explain it?",
    "Give me an overview of {topic}.",
    "Break down {topic} for me.",
]

SECTION_QA_TEMPLATES = [
    "What is {topic} and why does it matter?",
    "Explain how {topic} works.",
    "Can you break down {topic} for me?",
    "Describe {topic} and its key concepts.",
    "What should I know about {topic}?",
]

HOWTO_TEMPLATES = [
    "How do you implement {topic}?",
    "Walk me through implementing {topic}.",
    "What's the step-by-step approach to {topic}?",
    "How would you build {topic} in practice?",
    "Show me how to set up {topic}.",
]

CODE_TEMPLATES = [
    "Show me how to implement {topic} with code.",
    "Write code for {topic}.",
    "Give me a code example for {topic}.",
    "How do I code {topic}?",
]

COMPARE_TEMPLATES = [
    "Compare {topic1} and {topic2}.",
    "What's the difference between {topic1} and {topic2}?",
    "When should I use {topic1} vs {topic2}?",
    "{topic1} vs {topic2} — what are the trade-offs?",
]

SUMMARIZE_TEMPLATES = [
    "Summarize {topic}.",
    "What are the key takeaways of {topic}?",
    "Give me the main points about {topic}.",
]

FOLLOWUP_TEMPLATES = [
    "Can you give a practical example?",
    "What are the common pitfalls?",
    "How does this compare to alternatives?",
    "When would you not use this approach?",
    "What are the prerequisites to understand this?",
]


def clean_response(text: str) -> str:
    """Clean a response: strip emoji, math unicode, normalize whitespace."""
    # Strip leading emoji from each line
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = LEADING_EMOJI_RE.sub("", line)
        # Convert math unicode to regular text (approximate)
        line = MATH_UNICODE_RE.sub(lambda m: chr((ord(m.group()) - 0x1D400) % 52 + (65 if (ord(m.group()) - 0x1D400) % 52 < 26 else 71)), line)
        cleaned.append(line)
    text = "\n".join(cleaned)

    # Remove "Reading time: X minutes." lines
    text = re.sub(r"Reading time:.*?minutes?\.", "", text)

    # Remove "Together with X" sponsor intro blocks
    text = re.sub(
        r"Together with \w+.*?(?=\n[A-Z]|\n\n)",
        "", text, count=1, flags=re.DOTALL
    )

    # Remove "Today's daily dose of data science" intro and self-references
    text = re.sub(r"Today's daily dose of data science\n*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"daily dose of (?:data science|ds)\s*", "", text, flags=re.IGNORECASE)

    # Remove "In today's newsletter:" blocks
    text = re.sub(r"In today[\u2019']s newsletter:.*?(?=\n[A-Z])", "", text, flags=re.DOTALL)

    # Remove stray "In today's newsletter:" lines
    text = re.sub(r"In today's newsletter:.*?(?=\n[A-Z])", "", text, flags=re.DOTALL)

    # Remove "TODAY'S ISSUE" header blocks
    text = re.sub(r"\*?\*?TODAY[''']?S ISSUE\*?\*?\s*\n*", "", text, flags=re.IGNORECASE)

    # Remove stray promotional fragments (handle curly/straight apostrophes)
    promo_patterns = [
        r".*don[\u2018\u2019'']t forget to star[^)]*\)?\s*$",
        r"\(and\s+don[\u2018\u2019'']t forget to star[^)]*\)\.?",
        r"\(don[\u2018\u2019'']t forget to star[^)]*\)\.?",
        r"GitHub Repo:?\s*→?\s*$",
        r"GitHub repo\s*→?\s*$",
        r"Star the repo.*$",
        r"Try it here\s*→?\s*$",
        r"Get started.*?→\s*$",
        r"You can find the \w+ repo here\s*→?\s*$",
        r"^\s*Open-source\s*$",
        r"check it out.*→?\s*$",
    ]
    for pat in promo_patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_topic(text: str) -> str:
    """Clean a topic string for use in instructions."""
    text = text.strip()
    # Remove trailing arrows, special chars
    text = re.sub(r"\s*[→←↑↓]\s*$", "", text)
    # Remove leading/trailing special chars and zero-width spaces
    text = text.strip("​ \t\u200b")
    # Remove "this" prefix from newsletter self-references
    text = re.sub(r"^this\s+", "", text, flags=re.IGNORECASE)
    # Remove trailing question marks (avoid "What is X??")
    text = text.rstrip("?")
    # Remove leading "Read " or "Try " link artifacts
    text = re.sub(r"^(?:Read|Try|Check out|Visit|See)\s+", "", text)
    return text.strip()


def has_code(text: str) -> bool:
    """Check if text contains code-like content."""
    indicators = [
        "import ", "def ", "class ", "from ", "pip install",
        "```", "print(", ".fit(", ".predict(", ".transform(",
        "model.", "torch.", "tf.", "np.", "pd.",
        "self.", "__init__", "return ", "for i in ",
        ">>> ", "$ ", "lambda ", ".cuda()",
    ]
    return sum(1 for i in indicators if i in text) >= 2


def is_promotional(text: str) -> bool:
    """Check if text is primarily promotional / course-selling."""
    promo_signals = [
        "crash course", "we cover", "we have covered",
        "we discussed", "course with", "sign up",
        "subscribe", "premium", "membership",
        "advertise to", "sponsor us",
    ]
    text_lower = text.lower()
    hits = sum(1 for s in promo_signals if s in text_lower)
    # If more than 30% of the content is promo-like
    return hits >= 3 or (hits >= 2 and len(text) < 300)


def is_article_educational(content: str) -> bool:
    """Check if scraped article content is educational, not junk."""
    if len(content) < 300:
        return False

    # Too much noise
    noise_ratio = (
        content.count("cookie") +
        content.count("subscribe") +
        content.count("newsletter") +
        content.count("sign up")
    ) / max(len(content.split()), 1)
    if noise_ratio > 0.05:
        return False

    # Should have some educational content
    edu_signals = [
        "algorithm", "model", "data", "function", "method",
        "implement", "example", "code", "train", "predict",
        "feature", "parameter", "layer", "network", "learn",
        "accuracy", "loss", "performance", "optimization",
        "architecture", "approach", "technique", "pipeline",
    ]
    text_lower = content.lower()
    return sum(1 for s in edu_signals if s in text_lower) >= 3


def make_chat_example(
    instruction: str,
    response: str,
    category: str,
    source_id: str,
) -> dict:
    """Create a ChatML/ShareGPT format training example with metadata."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ],
        "category": category,
        "source_id": source_id,
    }


def make_multiturn_example(
    turns: list[tuple[str, str]],
    category: str,
    source_id: str,
) -> dict:
    """Create a multi-turn conversation example."""
    conversations = [{"from": "system", "value": SYSTEM_PROMPT}]
    for user_msg, asst_msg in turns:
        conversations.append({"from": "human", "value": user_msg})
        conversations.append({"from": "gpt", "value": asst_msg})
    return {
        "conversations": conversations,
        "category": category,
        "source_id": source_id,
    }


def generate_pairs(email: dict) -> list[dict]:
    """Generate training examples from a single cleaned email."""
    examples = []
    subject = clean_topic(email["subject"])
    full_text = email["full_text"]
    sections = email["sections"]
    categories = email["categories"]
    primary_cat = categories[0] if categories else "general"
    email_id = email["id"]

    # ── 1. Subject → overview explanation ───────────────────────────
    # Use only the main content sections (not intro/promo)
    # Section names that are newsletter structure, not real topics
    STRUCTURAL_SECTIONS = {
        "TODAY'S ISSUE", "TODAY'S TOPICS", "OPEN SOURCE",
        "VISUAL EXPLAINERS", "intro",
    }
    STRUCTURAL_UPPER = {s.upper() for s in STRUCTURAL_SECTIONS}

    def section_topic(title: str) -> str:
        """Convert section title to a proper topic name."""
        if title in STRUCTURAL_SECTIONS or title.upper() in STRUCTURAL_UPPER:
            return subject
        return clean_topic(title)

    content_sections = [
        s for s in sections
        if s["title"].lower() != "intro"
        and len(s["content"]) > 100
        and not is_promotional(s["content"])
    ]

    if content_sections:
        # Combine main content sections for a comprehensive response
        main_content = "\n\n".join(
            f"**{s['title']}**\n{s['content']}"
            if len(content_sections) > 1 else s["content"]
            for s in content_sections[:3]  # cap at 3 sections
        )
        main_content = clean_response(main_content)

        if 100 < len(main_content) < 6000:
            template = random.choice(EXPLAIN_TEMPLATES)
            instruction = template.format(topic=subject)
            examples.append(make_chat_example(
                instruction, main_content, primary_cat, email_id
            ))

    # ── 2. Section-level Q&A ────────────────────────────────────────
    for section in content_sections:
        title = section["title"]
        content = clean_response(section["content"])

        if len(content) < 100 or is_promotional(content):
            continue

        topic = section_topic(title)

        if len(topic) < 3:
            continue

        template = random.choice(SECTION_QA_TEMPLATES)
        instruction = template.format(topic=topic)
        examples.append(make_chat_example(
            instruction, content, primary_cat, email_id
        ))

        # Code-focused pair if section has code
        if has_code(content):
            template = random.choice(CODE_TEMPLATES)
            instruction = template.format(topic=topic)
            examples.append(make_chat_example(
                instruction, content, primary_cat, email_id
            ))

        # How-to pair for practical sections
        practical_keywords = [
            "hands-on", "implement", "build", "deploy",
            "tutorial", "step-by-step", "set up", "create",
        ]
        if any(kw in title.lower() for kw in practical_keywords):
            template = random.choice(HOWTO_TEMPLATES)
            instruction = template.format(topic=topic)
            examples.append(make_chat_example(
                instruction, content, primary_cat, email_id
            ))

    # ── 3. Comparison pairs ─────────────────────────────────────────
    # Only compare sections with real topic names (not structural)
    if len(content_sections) >= 2:
        for i in range(min(len(content_sections) - 1, 2)):
            s1, s2 = content_sections[i], content_sections[i + 1]
            t1 = section_topic(s1["title"])
            t2 = section_topic(s2["title"])
            # Skip if both resolve to subject (no meaningful comparison)
            if t1 == t2 or len(t1) < 3 or len(t2) < 3:
                continue

            c1 = clean_response(s1["content"])[:2000]
            c2 = clean_response(s2["content"])[:2000]

            template = random.choice(COMPARE_TEMPLATES)
            instruction = template.format(topic1=t1, topic2=t2)
            response = f"**{t1}:**\n{c1}\n\n**{t2}:**\n{c2}"

            examples.append(make_chat_example(
                instruction, response, primary_cat, email_id
            ))

    # ── 4. Article summarization (only educational content) ─────────
    for article in email.get("articles", []):
        content = article.get("content", "")
        link_text = clean_topic(article.get("link_text", ""))

        if not is_article_educational(content):
            continue

        topic = link_text if link_text and len(link_text) > 5 else subject
        content = clean_response(content[:5000])

        if len(content) < 200:
            continue

        template = random.choice(SUMMARIZE_TEMPLATES)
        instruction = template.format(topic=topic)
        examples.append(make_chat_example(
            instruction, content, primary_cat, email_id
        ))

    # ── 5. Multi-turn conversations ─────────────────────────────────
    if len(content_sections) >= 2:
        s1 = content_sections[0]
        s2 = content_sections[1] if len(content_sections) > 1 else None

        c1 = clean_response(s1["content"])
        t1 = section_topic(s1["title"])

        if len(c1) > 100 and len(t1) > 3 and s2:
            c2 = clean_response(s2["content"])
            t2 = section_topic(s2["title"])

            if len(c2) > 100 and len(t2) > 3:
                turn1_q = random.choice(EXPLAIN_TEMPLATES).format(topic=t1)
                turn2_q = random.choice(FOLLOWUP_TEMPLATES)

                # Second response connects to first topic
                turn2_resp = f"Building on {t1}:\n\n{c2[:1500]}"

                examples.append(make_multiturn_example(
                    [(turn1_q, c1[:2000]), (turn2_q, turn2_resp)],
                    primary_cat, email_id
                ))

    return examples


def dedup_examples(examples: list[dict]) -> list[dict]:
    """Remove near-duplicate examples.

    Dedup on instruction + first 200 chars of response, so the same question
    with genuinely different content is kept.
    """
    seen = set()
    unique = []
    for ex in examples:
        instr = ex["conversations"][1]["value"]
        resp_prefix = ex["conversations"][2]["value"][:200]
        h = hashlib.md5(f"{instr}|||{resp_prefix}".encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique


def code_only_filter(examples: list[dict]) -> list[dict]:
    """Keep only examples where the response contains clean, well-structured code."""
    import re as _re
    code_indicators = [
        "import ", "def ", "class ", "from ", "```",
        "print(", ".fit(", ".predict(", ".transform(",
        "np.", "pd.", "torch.", "self.", "return ",
        "model.", "plt.", "sklearn", "DataFrame",
        "for i in ", "lambda ", ".cuda()", "__init__",
        ".read_csv", ".to_csv", ".groupby(", ".merge(",
        "optimizer", "criterion", "scheduler",
    ]
    filtered = []
    for ex in examples:
        response = ex["conversations"][-1]["value"]
        hits = sum(1 for i in code_indicators if i in response)
        if hits < 2:
            continue

        # Skip very short responses (teaches truncation)
        if len(response) < 150:
            continue

        # Must have code fences OR substantial bare code (3+ code indicators)
        has_fences = "```" in response
        if not has_fences and hits < 3:
            continue

        # Wrap bare code in fences for consistency
        if not has_fences:
            response = "```python\n" + response.strip() + "\n```"
            ex["conversations"][-1]["value"] = response

        # Remove notebook output artifacts that cause hallucination loops
        if _re.search(r"Output:|Out\[|In\[|\boutput:", response):
            # Strip output blocks and keep if still substantial
            cleaned = _re.sub(
                r"(?:Output:|output:).*?(?=```|\Z)", "", response, flags=_re.DOTALL
            )
            if len(cleaned) < 200 or "```" not in cleaned:
                continue
            ex["conversations"][-1]["value"] = cleaned.strip()

        # Skip responses with emoji artifacts
        if "🎯" in response or "→" in response:
            response = _re.sub(r"[🎯→←↑↓]", "", response)
            ex["conversations"][-1]["value"] = response

        filtered.append(ex)
    return filtered


def quality_filter(examples: list[dict]) -> list[dict]:
    """Filter out low-quality training examples."""
    filtered = []
    for ex in examples:
        convos = ex["conversations"]
        instruction = convos[1]["value"]
        response = convos[2]["value"]

        # Skip very short or empty
        if len(instruction) < 10 or len(response) < 80:
            continue

        # Skip if response is mostly URLs or has too many
        url_count = response.count("http")
        word_count = len(response.split())
        if url_count > 5:
            continue
        if word_count > 0 and url_count / word_count > 0.15:
            continue

        # Skip if instruction contains URL or arrow artifacts
        if "http" in instruction or "→" in instruction:
            continue

        # Skip if response still has footer/sponsor remnants
        footer_signals = [
            "update your profile", "unsubscribe",
            "advertise to", "sponsor us",
            "brought to you by", "© 20",
            "together with ",
            "today's issue",
        ]
        # Check first 150 chars for sponsor leaks, full text for footer
        if any(s in response.lower()[:150] for s in ["together with "]):
            # Try to salvage by stripping the sponsor block
            lines = response.split("\n")
            clean_lines = []
            skip = False
            for line in lines:
                if "together with" in line.lower() and not skip:
                    skip = True
                    continue
                if skip and line.strip() == "":
                    skip = False
                    continue
                if not skip:
                    clean_lines.append(line)
            response = "\n".join(clean_lines).strip()
            ex["conversations"][2]["value"] = response
            if len(response) < 80:
                continue
        # Re-check after sponsor stripping
        resp_lower = response.lower()
        if any(s in resp_lower for s in footer_signals[:-1]):
            continue

        # Cap total length to avoid truncation waste (increased for 2048 seq len)
        total_len = len(instruction) + len(response)
        if total_len > 10000:
            # Trim response, keeping instruction intact
            response = response[:10000 - len(instruction)]
            # Try to trim at sentence boundary
            last_period = response.rfind(".")
            if last_period > len(response) * 0.7:
                response = response[:last_period + 1]
            ex["conversations"][2]["value"] = response

        filtered.append(ex)

    return filtered


def stratified_split(
    examples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split by category."""
    rng = random.Random(seed)

    buckets = defaultdict(list)
    for ex in examples:
        buckets[ex["category"]].append(ex)

    train, val = [], []
    for cat, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def strip_metadata(examples: list[dict]) -> list[dict]:
    """Remove category/source_id before writing (training doesn't need them)."""
    out = []
    for ex in examples:
        clean = {"conversations": ex["conversations"]}
        out.append(clean)
    return out


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    clean_files = sorted(CLEAN_DIR.glob("*.json"))
    clean_files = [f for f in clean_files if f.name != "stats.json"]
    print(f"Found {len(clean_files)} cleaned emails")

    all_examples = []
    for email_file in clean_files:
        email = json.loads(email_file.read_text())
        pairs = generate_pairs(email)
        all_examples.extend(pairs)

    total_raw = len(all_examples)
    print(f"Raw newsletter examples: {total_raw}")

    # Merge class material examples if available
    class_file = TRAIN_DIR / "class_examples.jsonl"
    if class_file.exists():
        class_examples = []
        with open(class_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    class_examples.append(json.loads(line))
        print(f"Merging {len(class_examples)} class material examples")
        all_examples.extend(class_examples)
        total_raw = len(all_examples)
    else:
        print("No class_examples.jsonl found — run format_class_data.py first to include class materials")

    # Merge public dataset examples if available
    public_file = TRAIN_DIR / "public_examples.jsonl"
    if public_file.exists():
        public_examples = []
        with open(public_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    public_examples.append(json.loads(line))
        print(f"Merging {len(public_examples)} public dataset examples")
        all_examples.extend(public_examples)
        total_raw = len(all_examples)

    # Merge curated examples (high-priority, always included)
    for curated_name in ["curated_examples.jsonl", "curated_v2_examples.jsonl", "curated_v3_examples.jsonl"]:
        curated_file = TRAIN_DIR / curated_name
        if curated_file.exists():
            curated = []
            with open(curated_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        curated.append(json.loads(line))
            # Add curated examples multiple times for emphasis (3x weight)
            print(f"Merging {len(curated)} curated examples from {curated_name} (3x weighted)")
            for _ in range(3):
                all_examples.extend(curated)
            total_raw = len(all_examples)

    print(f"Total raw examples: {total_raw}")

    # Deduplicate
    all_examples = dedup_examples(all_examples)
    print(f"After dedup: {len(all_examples)}")

    # Quality filter
    all_examples = quality_filter(all_examples)
    print(f"After quality filter: {len(all_examples)}")

    # Code-only filter (keep only examples with actual code in response)
    all_examples = code_only_filter(all_examples)
    print(f"After code-only filter: {len(all_examples)}")

    # Category distribution
    cat_counts = defaultdict(int)
    for ex in all_examples:
        cat_counts[ex["category"]] += 1

    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(all_examples)*100:.1f}%)")

    # Cap overrepresented categories to reduce imbalance
    MAX_PER_CAT = 3000
    capped = []
    cap_buckets = defaultdict(list)
    for ex in all_examples:
        cap_buckets[ex["category"]].append(ex)
    for cat, items in cap_buckets.items():
        if len(items) > MAX_PER_CAT:
            print(f"  Capping {cat}: {len(items)} → {MAX_PER_CAT}")
            random.shuffle(items)
            items = items[:MAX_PER_CAT]
        capped.extend(items)
    all_examples = capped
    print(f"After category cap ({MAX_PER_CAT}): {len(all_examples)}")

    # Stratified split
    train, val = stratified_split(all_examples)

    # Write output (strip metadata for training files)
    def write_jsonl(path, examples):
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_DIR / "train_chat.jsonl", strip_metadata(train))
    write_jsonl(TRAIN_DIR / "val_chat.jsonl", strip_metadata(val))

    # Also write with metadata for analysis
    write_jsonl(TRAIN_DIR / "train_full.jsonl", train)
    write_jsonl(TRAIN_DIR / "val_full.jsonl", val)

    # Write MLX format (messages with role/content)
    def to_mlx_format(examples):
        mlx_examples = []
        for ex in examples:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            messages = [
                {"role": role_map[c["from"]], "content": c["value"]}
                for c in ex["conversations"]
            ]
            mlx_examples.append({"messages": messages})
        return mlx_examples

    write_jsonl(TRAIN_DIR / "train.jsonl", to_mlx_format(train))
    write_jsonl(TRAIN_DIR / "valid.jsonl", to_mlx_format(val))

    # Stats
    stats = {
        "total_raw": total_raw,
        "total_after_filter": len(all_examples),
        "train_count": len(train),
        "val_count": len(val),
        "category_distribution": dict(cat_counts),
        "emails_processed": len(clean_files),
        "class_examples_merged": class_file.exists(),
    }
    (TRAIN_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nTrain: {len(train)}, Val: {len(val)}")
    print(f"Output: {TRAIN_DIR}")


if __name__ == "__main__":
    main()
