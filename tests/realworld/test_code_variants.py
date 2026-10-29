"""App. D.4 corpus builders: benchmark scanner, pinned receipts, id resolution (CPU only, synthetic data)."""

import json
import sys
from pathlib import Path

import pytest

CODE_VARIANTS = Path(__file__).resolve().parents[2] / "experiments" / "realworld" / "prepare" / "code_variants"
sys.path.insert(0, str(CODE_VARIANTS))

import common  # noqa: E402


def test_receipts_are_consistent():
    excluded = common.load_receipt("excluded_source_episode_ids.json")
    ids = excluded["excluded_source_episode_ids"]
    assert len(ids) == len(set(ids)) == 514 and ids == sorted(ids)
    drops = common.load_receipt("oci_decontamination_drops.json")["rows"]
    fingerprint = common.load_receipt("oci_fingerprint_exclusions.json")
    for rows, n in ((drops, 19), (fingerprint["rows"], 11)):
        assert len(rows) == n == len({r["source_row_index"] for r in rows})
        assert all(len(r["conversation_sha256"]) == 64 for r in rows)
        assert all(r["source_id"] == f"opencodeinstruct:{r['source_row_index']}" for r in rows)
    assert not {r["source_row_index"] for r in drops} & {r["source_row_index"] for r in fingerprint["rows"]}
    assert len(fingerprint["patterns"][0]["token_ids"]) == 16
    expected = common.load_receipt("expected_artifacts.json")
    assert expected["code_removed"]["n_supervised_tokens"] == 180_391_606
    replaced = expected["code_replaced"]
    assert replaced["n_supervised_tokens"] == 180_391_606 + replaced["coding_supervised_tokens"]
    assert 0 <= replaced["coding_supervised_tokens"] - replaced["target_supervised_tokens"] < 2049


def test_opencodeinstruct_contract_prompt():
    import oci_records

    record = {"average_test_score": 1.0, "input": " Add two numbers. ",
              "output": "```python\ndef add(a, b):\n    return a + b\n```",
              "unit_tests": json.dumps(["assert add(1, 2) == 3", "def helper(): pass", "assert other(1)",
                                        "assert add(0, 0) == 0", "assert add(2, 2) == 4", "assert add(3, 3) == 6"])}
    user, assistant = oci_records.opencodeinstruct_contract_messages(record)
    assert assistant == {"role": "assistant", "content": "def add(a, b):\n    return a + b"}
    assert user["content"] == ("Add two numbers.\n\nWrite a Python solution that passes these tests:\n```python\n"
                               "assert add(1, 2) == 3\nassert add(0, 0) == 0\nassert add(2, 2) == 4\n```")
    prompt, response = oci_records.simple_chatml_chunks([user, assistant])
    assert prompt.startswith("<|im_start|>user\n") and prompt.endswith("<|im_start|>assistant\n")
    assert response.endswith("<|im_end|>\n")
    for bad in ({**record, "average_test_score": 0.9}, {**record, "unit_tests": json.dumps(["assert other(1)"])},
                {**record, "output": "x = 1"}):
        with pytest.raises(ValueError):
            oci_records.opencodeinstruct_contract_messages(bad)


def test_prompt_deduplicator_exact_and_near():
    import oci_records

    dedup = oci_records.PromptDeduplicator(near_jaccard_threshold=0.92)
    base = "Write a function that returns the sum of all even numbers in a list of integers, ignoring negatives"
    assert dedup.find_duplicate(base) is None
    dedup.add(base, component="c", source_id="0")
    assert dedup.find_duplicate("  WRITE a function that returns the sum of all even numbers in a list of integers, "
                                "ignoring negatives ")[0] == "normalized_exact_prompt"
    assert dedup.find_duplicate("Reverse a linked list in place and return the new head node of the list") is None
    with pytest.raises(ValueError):
        dedup.add(base.upper(), component="c", source_id="1")


def test_fingerprint_ids_resolve_from_source_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("unified_training.episodes")
    code_replaced = pytest.importorskip("code_replaced")
    rows = common.load_receipt("oci_fingerprint_exclusions.json")["rows"]
    episodes = list(range(len(rows) + 3))
    table = {"episode": episodes,
             "source_row_index": [r["source_row_index"] for r in rows] + [1, 2, 3],
             "conversation_sha256": [r["conversation_sha256"] for r in rows] + ["x", "y", "z"]}
    pq.write_table(pa.table(table), tmp_path / "provenance.parquet")
    assert code_replaced.fingerprint_exclusion_ids(tmp_path) == list(range(len(rows)))
    table["conversation_sha256"][0] = "changed"
    pq.write_table(pa.table(table), tmp_path / "provenance.parquet")
    with pytest.raises(ValueError):
        code_replaced.fingerprint_exclusion_ids(tmp_path)


def test_excluded_ids_require_the_source_corpus(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        common.excluded_original_ids(tmp_path, None, None, None)
