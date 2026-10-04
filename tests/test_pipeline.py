"""Unit tests for the synthetic dataset generators and chat-format conversion.

These run offline: the synthetic generators and the format converters use only
the standard library. HF-backed adapters are not tested here (they need network
and the `datasets` package).
"""

import json
import random

from local_decision_model.build_dataset import opt, programmatic_policies, rec, routing
from local_decision_model.to_mlx_chat import SYSTEM_PROMPT, format_options, to_example

SCHEMA_KEYS = {"state", "question", "options", "answer", "source"}


def test_opt_record_shape() -> None:
    o = opt("A", "approve", "Approve it.")
    assert o == {"label": "A", "key": "approve", "description": "Approve it."}

    r = rec("state text", "question?", [o], "approve", "policies")
    assert set(r) == SCHEMA_KEYS
    assert r["answer"] == "approve"


def test_programmatic_policies_schema_and_answer_validity() -> None:
    rng = random.Random(42)
    recs = list(programmatic_policies(200, rng))
    assert len(recs) == 200
    valid = {"approve", "review", "deny"}
    for r in recs:
        assert set(r) == SCHEMA_KEYS
        assert r["source"] == "policies"
        assert r["answer"] in valid
        keys = [o["key"] for o in r["options"]]
        assert r["answer"] in keys
        assert len({o["label"] for o in r["options"]}) == 3


def test_programmatic_policies_respects_damage_override() -> None:
    """Damaged items are always approved regardless of age (policy invariant)."""
    rng = random.Random(0)
    for r in programmatic_policies(500, rng):
        damaged = "Item reported damaged: yes" in r["state"]
        if damaged:
            assert r["answer"] == "approve"


def test_routing_schema_and_labels() -> None:
    rng = random.Random(42)
    recs = list(routing(100, rng))
    assert len(recs) == 100
    for r in recs:
        assert r["source"] == "routing"
        assert r["answer"] in {"billing", "technical", "account", "sales"}
        assert len(r["options"]) == 4
        assert [o["label"] for o in r["options"]] == ["A", "B", "C", "D"]


def test_to_example_bakes_system_prompt_and_contract() -> None:
    r = rec(
        "state text",
        "question?",
        [opt("A", "yes", "It is yes."), opt("B", "no", "It is no.")],
        "yes",
        "boolq",
    )
    ex = to_example(r)
    msgs = ex["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "state: state text" in msgs[1]["content"]
    parsed = json.loads(msgs[2]["content"])
    assert parsed == {"label": "A", "key": "yes"}


def test_format_options_rendering() -> None:
    block = format_options([opt("A", "yes", "It is yes."), opt("B", "no", "It is no.")])
    assert block == "A. yes - It is yes.\nB. no - It is no."
