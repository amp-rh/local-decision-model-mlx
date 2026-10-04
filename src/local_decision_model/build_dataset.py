"""Sample and normalize public classification datasets into the unified Jev schema.

Unified record: {"state", "question", "options": [{label, key, description}], "answer", "source"}.
Run on a machine with `pip install datasets` (not needed on the Mac training box).
"""

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset  # type: ignore[import-untyped]
except ImportError:  # only needed for HF downloads, not for synthetic generators
    load_dataset = None

OUT = Path(__file__).resolve().parent.parent / "data"
LETTERS = "ABCDEFGH"


def opt(label: str, key: str, description: str) -> dict[str, str]:
    return {"label": label, "key": key, "description": description}


def rec(
    state: str, question: str, options: list[dict[str, str]], answer: str, source: str
) -> dict[str, Any]:
    return {
        "state": state,
        "question": question,
        "options": options,
        "answer": answer,
        "source": source,
    }


# --- HF dataset adapters: yield unified records -----------------------------


def multinli(n: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Yield n three-way entailment records from MultiNLI."""
    ds = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=42)
    for ex in ds.select(range(n)):
        opts = [
            opt("A", "entailment", "The hypothesis is true given the premise."),
            opt("B", "contradiction", "The hypothesis is false given the premise."),
            opt("C", "neutral", "The hypothesis cannot be judged from the premise."),
        ]
        yield rec(
            f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}",
            "Does the premise entail, contradict, or neither support the hypothesis?",
            opts,
            ["entailment", "neutral", "contradiction"][ex["label"]],
            "multi_nli",
        )


def boolq(n: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Yield n binary yes/no reading-comprehension records from BoolQ."""
    ds = load_dataset("google/boolq", split="train").shuffle(seed=42)
    for ex in ds.select(range(n)):
        opts = [
            opt("A", "yes", "The answer is yes."),
            opt("B", "no", "The answer is no."),
        ]
        yield rec(
            f"Passage: {ex['passage']}\nQuestion: {ex['question']}",
            "Based on the passage, is the answer to the question yes or no?",
            opts,
            "yes" if ex["answer"] else "no",
            "boolq",
        )


def banking77(n: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Yield n 10-way banking-intent classification records from Banking77."""
    ds = load_dataset("PolyAI/banking77", split="train").shuffle(seed=42)
    # Build a fixed option set of 10 intents, sampled per-example from the 77
    intents = sorted(set(ds["label"]))
    names = ds.features["label"].names
    for ex in ds.select(range(n)):
        correct = ex["label"]
        distractors = rng.sample([i for i in intents if i != correct], 9)
        pool = [correct] + distractors
        rng.shuffle(pool)
        opts = [opt(LETTERS[i], f"intent_{p}", names[p]) for i, p in enumerate(pool)]
        answer = next(o["key"] for o in opts if o["key"] == f"intent_{correct}")
        yield rec(
            ex["text"],
            "Which banking intent best matches this customer message?",
            opts,
            answer,
            "banking77",
        )


def agnews(n: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Yield n 4-way news-topic classification records from AG News."""
    ds = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=42)
    opts = [
        opt(LETTERS[i], k, f"The article is about {v.lower()}.")
        for i, (k, v) in enumerate(
            [
                ("world", "world"),
                ("sports", "sports"),
                ("business", "business"),
                ("tech", "science/tech"),
            ]
        )
    ]
    mapping = {0: "world", 1: "sports", 2: "business", 3: "tech"}
    for ex in ds.select(range(n)):
        yield rec(
            ex["text"],
            "Which category does this news article belong to?",
            opts,
            mapping[ex["label"]],
            "ag_news",
        )


def sst5(n: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Yield n 5-way sentiment records from SST-5."""
    ds = load_dataset("stanfordnlp/sst5", split="train").shuffle(seed=42)
    opts = [
        opt("A", "very_negative", "Very negative sentiment."),
        opt("B", "negative", "Negative sentiment."),
        opt("C", "neutral", "Neutral sentiment."),
        opt("D", "positive", "Positive sentiment."),
        opt("E", "very_positive", "Very positive sentiment."),
    ]
    keys = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    for ex in ds.select(range(n)):
        yield rec(
            ex["text"], "What is the sentiment of this text?", opts, keys[ex["label"]], "sst5"
        )


# --- Synthetic generators (no download needed) -------------------------------


def programmatic_policies(n: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Refund-policy gate: apply a written rule to a refund request."""
    reasons = [
        "arrived damaged",
        "wrong item shipped",
        "found it cheaper elsewhere",
        "no longer needed",
        "never arrived",
        "stopped working within warranty",
        "accidental purchase",
        "allergic reaction to product",
    ]
    tiers = [
        (0, 7, "approve", "Purchases within 7 days are approved."),
        (7, 30, "review", "Purchases aged 7-30 days require manual review."),
        (30, 10**6, "deny", "Purchases older than 30 days are denied."),
    ]
    for _ in range(n):
        age = rng.randint(0, 120)
        damaged = rng.random() < 0.2
        tier = next(t for t in tiers if t[0] <= age < t[1])
        answer = "approve" if (tier[2] == "approve" or damaged) else tier[2]
        state = (
            f"Refund request: item {rng.choice(reasons)}. "
            f"Days since delivery: {age}. Item reported damaged: {'yes' if damaged else 'no'}.\n"
            f"Policy: {tiers[0][2]} purchases within 7 days. Manual review for 7-30 days. "
            f"Deny purchases older than 30 days. Always approve damaged items regardless of age."
        )
        opts = [
            opt("A", "approve", "Approve the refund automatically."),
            opt("B", "review", "Route to a human for review."),
            opt("C", "deny", "Deny the refund."),
        ]
        yield rec(
            state,
            "Per the policy, should this refund be approved, reviewed, or denied?",
            opts,
            answer,
            "policies",
        )


def routing(n: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Route a support ticket to one of four queues."""
    routes = [
        ("billing", "payments, charges, invoices, refunds"),
        ("technical", "bugs, errors, outages, integration problems"),
        ("account", "login, password, profile, permissions"),
        ("sales", "pricing questions, upgrades, new purchases"),
    ]
    msgs = {
        "billing": [
            "I was charged twice this month",
            "How do I update my card?",
            "Need a copy of my invoice",
        ],
        "technical": [
            "The app crashes on export",
            "Webhook returns 500 errors",
            "API latency is terrible today",
        ],
        "account": [
            "I can't log in anymore",
            "How do I reset my password?",
            "Need to add a teammate to my workspace",
        ],
        "sales": [
            "What does the enterprise plan cost?",
            "Can I upgrade to annual billing?",
            "Do you offer volume discounts?",
        ],
    }
    for _ in range(n):
        key = rng.choice([r[0] for r in routes])
        state = f'Support ticket: "{rng.choice(msgs[key])}"'
        opts = [opt(LETTERS[i], k, d) for i, (k, d) in enumerate(routes)]
        yield rec(
            state, "Which support queue should this ticket be routed to?", opts, key, "routing"
        )


# --- Main --------------------------------------------------------------------

BUILDERS = {
    "multi_nli": (multinli, 5000),
    "boolq": (boolq, 3000),
    "banking77": (banking77, 3000),
    "ag_news": (agnews, 1500),
    "sst5": (sst5, 2000),
    "policies": (programmatic_policies, 13500),
    "routing": (routing, 6000),
}


def main() -> None:
    rng = random.Random(42)
    OUT.mkdir(parents=True, exist_ok=True)
    all_recs = []
    for source, (fn, count) in BUILDERS.items():
        recs = list(fn(count, rng))
        print(f"{source}: {len(recs)} examples")
        all_recs.extend(recs)

    rng.shuffle(all_recs)
    n_dev = max(500, len(all_recs) // 50)
    dev, train = all_recs[:n_dev], all_recs[n_dev:]

    for split, recs in [("train", train), ("dev", dev)]:
        path = OUT / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{path}: {len(recs)}")


if __name__ == "__main__":
    main()
