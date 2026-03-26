#!/usr/bin/env python3
"""Extract code training pairs from class materials and newsletter links.

Focuses on producing high-quality instruction → code pairs for fine-tuning
a data science code LLM. Sources:
  1. Jupyter notebooks (class materials) — code cells with markdown context
  2. Python files (class materials) — functions/classes with docstrings
  3. GitHub repos linked in newsletters — README code examples

Output: data/training/class_examples.jsonl (ChatML format)

Usage:
    python format_class_data.py                            # Process all
    python format_class_data.py --source ~/Desktop/school/sta4241
    python format_class_data.py --dry-run
    python format_class_data.py --scrape-github            # Also scrape GitHub links
"""

import argparse
import base64
import json
import re
import random
import hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "data" / "training"
OUTPUT_FILE = TRAIN_DIR / "class_examples.jsonl"
CLEAN_DIR = BASE_DIR / "data" / "clean_emails"

SYSTEM_PROMPT = (
    "You are an expert data science coding assistant. "
    "Respond ONLY with clean, runnable Python code. "
    "Use inline comments for explanation. "
    "No text outside code blocks."
)

DEFAULT_SOURCES = [
    Path.home() / "Desktop" / "school",
    Path.home() / "Downloads",
]

SKIP_DIRS = {
    "cs1", "cis3100C", "cis3360", "cis4340",
    "cop 2500", "cop3223", "cop3502C", "cop4283",
    "venv", ".venv", "env", "__pycache__", ".ipynb_checkpoints",
    ".virtual_documents", "site-packages", "node_modules",
    ".git", "lib", "Lib",
}

COURSE_CATEGORIES = {
    "sta4241": "statistics",
    "sta4724": "machine_learning",
    "sta4365": "statistics",
    "sta4173": "statistics",
    "sta4164": "statistics",
    "sta4321": "statistics",
    "sta4322": "statistics",
    "sta4852": "statistics",
    "isc4551": "machine_learning",
    "isc4241": "machine_learning",
    "isc4242": "machine_learning",
    "ISLP": "statistics",
    "MLpy": "machine_learning",
    "kaggle": "machine_learning",
    "HUT": "deep_learning",
}

CATEGORY_KEYWORDS = {
    "deep_learning": [
        "neural network", "cnn", "rnn", "lstm", "transformer", "pytorch",
        "tensorflow", "keras", "deep learning", "backpropagation",
        "convolutional", "autoencoder", "gan", "attention", "torch",
    ],
    "statistics": [
        "hypothesis test", "p-value", "confidence interval", "anova",
        "chi-square", "t-test", "bayesian", "regression", "glm",
        "statsmodels", "statistical", "multivariate", "linear model",
    ],
    "machine_learning": [
        "random forest", "gradient boosting", "xgboost", "svm",
        "cross-validation", "feature selection", "classification",
        "clustering", "k-means", "decision tree", "sklearn",
        "train_test_split", "pipeline", "hyperparameter",
    ],
}

# ── Instruction templates ─────────────────────────────────────────────

CODE_INSTRUCTION_TEMPLATES = [
    "Write Python code to {task}.",
    "Show me how to {task} in Python.",
    "Write a Python script that {task}.",
    "Implement {task} using Python.",
    "Give me Python code for {task}.",
]

FUNCTION_TEMPLATES = [
    "Write a Python function that {desc}.",
    "Implement a function to {desc}.",
    "Create a Python function for {desc}.",
]

ANALYSIS_TEMPLATES = [
    "Write code to analyze {topic} using Python.",
    "Show me a data analysis pipeline for {topic}.",
    "How would you analyze {topic} with pandas and matplotlib?",
]

# ── Helpers ───────────────────────────────────────────────────────────


def should_skip_dir(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def detect_category(text: str, path: Path) -> str:
    for part in path.parts:
        for course, cat in COURSE_CATEGORIES.items():
            if part.lower() == course.lower():
                return cat
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower)
              for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "machine_learning"


def make_example(instruction: str, response: str, category: str, source_id: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ],
        "category": category,
        "source_id": source_id,
    }


def is_substantial_code(code: str) -> bool:
    """Check if code is worth training on (not just imports/prints)."""
    lines = [l.strip() for l in code.strip().split("\n")
             if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 3:
        return False
    import_lines = sum(1 for l in lines if l.startswith(("import ", "from ", "%", "!")))
    if import_lines >= len(lines) - 1:
        return False
    # Must have some real operations
    has_ops = any(kw in code for kw in [
        "=", "(", "def ", "class ", "for ", "if ", "return ",
        ".fit", ".predict", ".transform", ".plot", ".read_",
        "print(", "plt.", "pd.", "np.", "torch.", "model.",
    ])
    return has_ops


def clean_code(code: str) -> str:
    """Clean code for training: remove ANSI, normalize whitespace."""
    code = re.sub(r"\x1b\[[0-9;]*m", "", code)
    # Remove trailing whitespace per line
    lines = [l.rstrip() for l in code.split("\n")]
    # Remove excessive blank lines
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append(line)
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()


def extract_task_from_markdown(md: str) -> str:
    """Extract a task description from markdown context."""
    # Try header first
    header = re.search(r"^#{1,3}\s+(.+)$", md, re.MULTILINE)
    if header:
        task = header.group(1).strip()
        task = re.sub(r"\*+", "", task).strip()
        task = re.sub(r"^\d+[\.\)]\s*", "", task).strip()
        task = re.sub(r"^(?:Lab|Chapter|Ch)\s*\d*:?\s*", "", task, flags=re.IGNORECASE).strip()
        if 3 < len(task) < 80:
            return task.lower().rstrip(".")

    # Try bold text
    bold = re.search(r"\*\*(.+?)\*\*", md)
    if bold:
        task = bold.group(1).strip()
        if 3 < len(task) < 80:
            return task.lower().rstrip(".")

    # Try first meaningful sentence
    for line in md.split("\n"):
        line = line.strip().lstrip("#").strip()
        if len(line) > 10 and not re.match(r"^(homework|hw|assignment|due|name|student)", line, re.I):
            return line[:70].lower().rstrip(".")

    return ""


def is_assignment_header(text: str) -> bool:
    patterns = [
        r"^homework\s*\d", r"^hw\s*\d", r"^assignment\s*\d",
        r"^due\s*date", r"^due:\s*", r"^name:\s*", r"^student",
        r"^course:", r"^instructor:",
    ]
    text_lower = text.strip().lower()
    return any(re.match(p, text_lower) for p in patterns)


def strip_student_info(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(
        l for l in lines
        if not re.match(r"^(name|student|date|due|instructor|professor|course|class)\s*:", l.strip(), re.I)
    )


def clean_output(output: str) -> str:
    if not output:
        return ""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    if len(output) > 500:
        output = output[:500] + "\n..."
    return output.strip()


# ── Notebook processing ───────────────────────────────────────────────


def parse_notebook(path: Path) -> list[dict]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    cells = nb.get("cells", [])
    parsed = []
    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        outputs_text = ""
        if cell_type == "code":
            for out in cell.get("outputs", []):
                if "text" in out:
                    t = out["text"]
                    outputs_text += "".join(t) if isinstance(t, list) else t
                elif "text/plain" in out.get("data", {}):
                    t = out["data"]["text/plain"]
                    outputs_text += "".join(t) if isinstance(t, list) else t

        parsed.append({
            "type": cell_type,
            "source": source.strip(),
            "output": clean_output(outputs_text),
        })
    return parsed


def generate_notebook_code_pairs(path: Path) -> list[dict]:
    """Extract code training pairs from a notebook."""
    cells = parse_notebook(path)
    if not cells:
        return []

    all_text = " ".join(c["source"] for c in cells)
    category = detect_category(all_text, path)
    source_id = f"nb:{path.stem}"
    examples = []

    # Walk through cells looking for markdown → code patterns
    i = 0
    while i < len(cells):
        cell = cells[i]

        # Pattern 1: Markdown followed by code cell(s)
        if cell["type"] == "markdown" and not is_assignment_header(cell["source"]):
            md = strip_student_info(cell["source"])
            task = extract_task_from_markdown(md)

            # Collect following code cells
            code_cells = []
            j = i + 1
            while j < len(cells) and cells[j]["type"] == "code":
                if is_substantial_code(cells[j]["source"]):
                    code_cells.append(cells[j])
                j += 1

            if task and code_cells:
                # Build the code response
                code_parts = []
                for cc in code_cells[:3]:  # max 3 code cells per block
                    cleaned = clean_code(cc["source"])
                    code_parts.append(cleaned)
                    if cc["output"]:
                        code_parts.append(f"# Output:\n# {cc['output'].replace(chr(10), chr(10) + '# ')}")

                code_response = "\n\n".join(code_parts)

                if 50 < len(code_response) < 4000:
                    # Add brief context from markdown if helpful
                    context = ""
                    if len(md) > 30 and len(md) < 300:
                        context = f"# {md.strip().split(chr(10))[0]}\n\n"

                    template = random.choice(CODE_INSTRUCTION_TEMPLATES)
                    instruction = template.format(task=task)
                    response = f"```python\n{context}{code_response}\n```"

                    examples.append(make_example(
                        instruction, response, category, source_id
                    ))

            i = j
            continue

        # Pattern 2: Standalone substantial code cell (no markdown context)
        if cell["type"] == "code" and is_substantial_code(cell["source"]):
            code = clean_code(cell["source"])
            if 80 < len(code) < 3000:
                # Infer task from code content
                task = infer_task_from_code(code)
                if task:
                    response = f"```python\n{code}\n```"
                    if cell["output"]:
                        response += f"\n\nOutput:\n```\n{cell['output']}\n```"

                    template = random.choice(CODE_INSTRUCTION_TEMPLATES)
                    instruction = template.format(task=task)
                    examples.append(make_example(
                        instruction, response, category, source_id
                    ))

        i += 1

    return examples


def infer_task_from_code(code: str) -> str:
    """Infer a task description from code content."""
    # Check for common patterns
    patterns = [
        (r"\.fit\(", "train a {model} model"),
        (r"train_test_split", "split data into train and test sets"),
        (r"\.plot\(|plt\.", "create a plot/visualization"),
        (r"pd\.read_csv|pd\.read_", "load and process a dataset"),
        (r"\.predict\(", "make predictions with a trained model"),
        (r"cross_val_score|GridSearchCV", "perform cross-validation"),
        (r"StandardScaler|MinMaxScaler", "scale/normalize features"),
        (r"confusion_matrix|accuracy_score", "evaluate model performance"),
        (r"PCA\(|\.fit_transform", "perform dimensionality reduction"),
        (r"KMeans|DBSCAN|AgglomerativeClustering", "cluster the data"),
        (r"LinearRegression|Ridge|Lasso", "perform regression analysis"),
        (r"RandomForest|GradientBoosting|XGB", "train an ensemble model"),
        (r"torch\.nn|nn\.Module", "build a neural network"),
        (r"\.corr\(\)|heatmap", "compute correlations"),
        (r"groupby|pivot_table", "aggregate and summarize data"),
        (r"fillna|dropna|isna", "handle missing values"),
        (r"get_dummies|LabelEncoder", "encode categorical variables"),
    ]

    for pattern, desc in patterns:
        if re.search(pattern, code):
            # Try to identify the specific model/data
            model_match = re.search(
                r"(LinearRegression|LogisticRegression|RandomForest\w*|"
                r"GradientBoosting\w*|SVM|KNN|DecisionTree\w*|"
                r"XGB\w*|LightGBM|CatBoost|KMeans|DBSCAN|PCA|"
                r"Ridge|Lasso|ElasticNet|NaiveBayes|AdaBoost)\s*\(",
                code
            )
            if "{model}" in desc and model_match:
                desc = desc.replace("{model}", model_match.group(1))
            elif "{model}" in desc:
                desc = desc.replace("{model}", "machine learning")
            return desc

    # Check for function definitions
    func_match = re.search(r"def\s+(\w+)\s*\(", code)
    if func_match:
        name = func_match.group(1)
        # Convert function name to description
        name = name.replace("_", " ")
        return name

    return ""


# ── Python file processing ────────────────────────────────────────────


def generate_python_code_pairs(path: Path) -> list[dict]:
    """Extract code pairs from Python files."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    if len(content) < 100 or len(content) > 50000:
        return []

    # Skip if mostly imports
    lines = [l for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return []
    if sum(1 for l in lines if l.strip().startswith(("import ", "from "))) / max(len(lines), 1) > 0.7:
        return []

    category = detect_category(content, path)
    source_id = f"py:{path.stem}"
    examples = []

    # Extract functions with docstrings
    func_pattern = re.compile(
        r"^(def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*\S+)?\s*:)\s*\n"
        r'\s+(?:"""([\s\S]*?)"""|\'\'\'([\s\S]*?)\'\'\')',
        re.MULTILINE
    )

    for match in func_pattern.finditer(content):
        func_name = match.group(2)
        docstring = (match.group(3) or match.group(4) or "").strip()

        if len(docstring) < 10:
            continue

        # Get function body
        start = match.start()
        rest = content[match.end():]
        func_indent = len(match.group(0)) - len(match.group(0).lstrip())
        end_offset = len(rest)
        for m in re.finditer(r"^(?:def |class )", rest, re.MULTILINE):
            line_start = rest.rfind("\n", 0, m.start()) + 1
            if m.start() - line_start <= func_indent:
                end_offset = m.start()
                break

        func_code = clean_code(content[start:match.end() + end_offset])
        if len(func_code) > 3000:
            func_code = func_code[:3000]

        desc = docstring.split("\n")[0].strip().rstrip(".").lower()
        template = random.choice(FUNCTION_TEMPLATES)
        instruction = template.format(desc=desc)
        response = f"```python\n{func_code}\n```"

        if 80 < len(response) < 4000:
            examples.append(make_example(instruction, response, category, source_id))

    # Complete scripts (few functions, mostly procedural)
    func_count = content.count("\ndef ")
    if func_count <= 2 and len(content) > 200 and is_substantial_code(content):
        task = path.stem.replace("_", " ").replace("-", " ").lower()
        if len(task) > 3:
            code = clean_code(content[:3000])
            template = random.choice(CODE_INSTRUCTION_TEMPLATES)
            instruction = template.format(task=task)
            response = f"```python\n{code}\n```"
            if len(response) < 4000:
                examples.append(make_example(instruction, response, category, source_id))

    return examples


# ── Newsletter GitHub link scraping ───────────────────────────────────


def extract_github_urls_from_emails() -> list[str]:
    """Decode base64-encoded GitHub URLs from newsletter links."""
    if not CLEAN_DIR.exists():
        return []

    github_urls = set()
    for f in sorted(CLEAN_DIR.glob("*.json")):
        if f.name == "stats.json":
            continue
        try:
            email = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue

        for link in email.get("links", []):
            url = link.get("url", "")
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2:
                candidate = parts[-1]
                padding = 4 - len(candidate) % 4
                if padding != 4:
                    candidate += "=" * padding
                try:
                    decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
                    if decoded.startswith("http") and "github.com" in decoded:
                        github_urls.add(decoded)
                except Exception:
                    pass

    return sorted(github_urls)


def scrape_github_readme(url: str) -> list[dict]:
    """Scrape code examples from a GitHub repo README."""
    import urllib.request
    import urllib.error

    # Convert github.com URL to raw README
    # e.g., https://github.com/user/repo → raw README
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)/(.+))?", url)
    if not match:
        return []

    user, repo = match.group(1), match.group(2)
    branch = match.group(3) or "main"
    subpath = match.group(4) or ""

    # Try to fetch README
    readme_urls = []
    if subpath:
        readme_urls.append(f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{subpath}/README.md")
    readme_urls.append(f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/README.md")
    readme_urls.append(f"https://raw.githubusercontent.com/{user}/{repo}/master/README.md")

    content = None
    for readme_url in readme_urls:
        try:
            req = urllib.request.Request(readme_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            continue

    if not content or len(content) < 100:
        return []

    examples = []
    source_id = f"gh:{user}/{repo}"
    category = detect_category(content, Path(url))

    # Extract code blocks with surrounding context
    # Find ```python ... ``` blocks
    code_blocks = re.finditer(
        r"(?:^|\n)(#{1,3}\s+.+?\n|.{10,80}\n)?```(?:python|py)\n(.*?)```",
        content, re.DOTALL
    )

    for match in code_blocks:
        context = (match.group(1) or "").strip()
        code = match.group(2).strip()

        if not is_substantial_code(code) or len(code) < 50 or len(code) > 3000:
            continue

        # Get task from context or infer from code
        task = ""
        if context:
            task = context.lstrip("#").strip().lower().rstrip(".")
            task = re.sub(r"^\d+[\.\)]\s*", "", task)

        if not task or len(task) < 5:
            task = infer_task_from_code(code)

        if not task:
            continue

        template = random.choice(CODE_INSTRUCTION_TEMPLATES)
        instruction = template.format(task=task)
        response = f"```python\n{clean_code(code)}\n```"

        examples.append(make_example(instruction, response, category, source_id))

    return examples


# ── Quality filtering ─────────────────────────────────────────────────


def quality_filter(examples: list[dict]) -> list[dict]:
    """Filter for high-quality code examples."""
    filtered = []
    for ex in examples:
        instruction = ex["conversations"][1]["value"]
        response = ex["conversations"][2]["value"]

        # Must have actual code
        if "```python" not in response and "```" not in response:
            continue

        # Extract code from response
        code_match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if not code_match:
            continue
        code = code_match.group(1)

        # Min code length
        if len(code.strip()) < 50:
            continue

        # Instruction quality — must be coherent
        if len(instruction) < 20:
            continue

        # Skip truncated/nonsensical instructions (auto-extracted artifacts)
        words = instruction.split()
        if len(words) < 4:
            continue
        # Check for sentences that end abruptly (missing object/verb)
        last_word = words[-1].rstrip(".")
        # Common signs of truncation: ends with article, preposition, or "the"
        truncation_endings = {"the", "a", "an", "to", "for", "with", "in", "on", "of", "that", "this", "and", "or", "by", "from", "using", "can"}
        if last_word.lower() in truncation_endings:
            continue

        # Skip if instruction has weird artifacts
        if instruction.count("`") > 2 or instruction.count("$") > 2:
            continue

        # Skip instructions with incomplete fragments
        if re.search(r"\bthe\s+$", instruction) or re.search(r"\ba\s+$", instruction):
            continue

        # Max total length
        if len(instruction) + len(response) > 5000:
            response = response[:5000 - len(instruction)]
            # Try to end at a code block boundary
            last_close = response.rfind("```")
            if last_close > len(response) * 0.5:
                response = response[:last_close + 3]
            ex["conversations"][2]["value"] = response

        # Skip if code is mostly comments
        code_lines = [l for l in code.split("\n") if l.strip()]
        comment_lines = sum(1 for l in code_lines if l.strip().startswith("#"))
        if code_lines and comment_lines / len(code_lines) > 0.7:
            continue

        filtered.append(ex)

    return filtered


def dedup_examples(examples: list[dict]) -> list[dict]:
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


# ── File collection ───────────────────────────────────────────────────


def collect_files(sources: list[Path]) -> dict[str, list[Path]]:
    files = {"notebooks": [], "python": []}

    for source in sources:
        source = source.expanduser().resolve()
        if not source.exists():
            print(f"  Warning: {source} does not exist, skipping")
            continue

        if source.is_file():
            if source.suffix == ".ipynb":
                files["notebooks"].append(source)
            elif source.suffix == ".py":
                files["python"].append(source)
            continue

        for p in source.rglob("*"):
            if should_skip_dir(p):
                continue
            if p.suffix == ".ipynb":
                files["notebooks"].append(p)
            elif p.suffix == ".py":
                if not any(skip in str(p) for skip in ["node_modules", "venv", ".venv"]):
                    files["python"].append(p)

    return files


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract code training pairs")
    parser.add_argument("--source", type=str, nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scrape-github", action="store_true",
                        help="Also scrape GitHub repos from newsletter links")
    args = parser.parse_args()

    random.seed(42)

    sources = [Path(s) for s in args.source] if args.source else DEFAULT_SOURCES

    print("Collecting files...")
    files = collect_files(sources)
    print(f"  Notebooks: {len(files['notebooks'])}")
    print(f"  Python:    {len(files['python'])}")

    all_examples = []

    # Process notebooks
    print("\nProcessing notebooks...")
    nb_count = 0
    for path in files["notebooks"]:
        pairs = generate_notebook_code_pairs(path)
        if pairs:
            nb_count += 1
            all_examples.extend(pairs)
    print(f"  {len(all_examples)} code pairs from {nb_count} notebooks")

    # Process Python files
    print("Processing Python files...")
    py_before = len(all_examples)
    py_count = 0
    for path in files["python"]:
        pairs = generate_python_code_pairs(path)
        if pairs:
            py_count += 1
            all_examples.extend(pairs)
    print(f"  {len(all_examples) - py_before} code pairs from {py_count} Python files")

    # Scrape GitHub repos
    if args.scrape_github:
        print("\nExtracting GitHub URLs from newsletters...")
        github_urls = extract_github_urls_from_emails()
        print(f"  Found {len(github_urls)} GitHub URLs")

        gh_before = len(all_examples)
        for url in github_urls:
            print(f"  Scraping {url}...", end=" ", flush=True)
            pairs = scrape_github_readme(url)
            all_examples.extend(pairs)
            print(f"{len(pairs)} pairs")
        print(f"  {len(all_examples) - gh_before} total from GitHub")

    total_raw = len(all_examples)
    print(f"\nTotal raw: {total_raw}")

    all_examples = dedup_examples(all_examples)
    print(f"After dedup: {len(all_examples)}")

    all_examples = quality_filter(all_examples)
    print(f"After quality filter: {len(all_examples)}")

    cat_counts = defaultdict(int)
    for ex in all_examples:
        cat_counts[ex["category"]] += 1

    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/max(len(all_examples),1)*100:.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_examples)} examples to {OUTPUT_FILE}")

    stats = {
        "total_raw": total_raw,
        "total_after_filter": len(all_examples),
        "category_distribution": dict(cat_counts),
        "files_processed": {"notebooks": nb_count, "python": py_count},
    }
    (TRAIN_DIR / "class_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
