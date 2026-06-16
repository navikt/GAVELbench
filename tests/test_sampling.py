"""Unit tests for resampling of unanswered questions.

Verifies that:
  * ``answered_questions`` returns the intersection of answered questions
    across all active models (a question is answered only when every active
    model has answered it).
  * ``sample_by_overkategori`` excludes already-answered questions so the
    pipeline resamples a fresh batch instead of redrawing the same set.

These tests use temporary files only — no external services are required.

Usage::

    uv run python tests/test_sampling.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
from fetch_data import sample_by_overkategori  # noqa: E402
from generate import answered_questions  # noqa: E402

_MAPPING = [
    {"kategori": "kat-a", "overkategori": ["Arbeid"]},
    {"kategori": "kat-b", "overkategori": ["Helse"]},
]


def _write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_answered_questions_intersection() -> None:
    """A question is answered only when every active model has answered it."""
    with tempfile.TemporaryDirectory() as tmp:
        gen_dir = Path(tmp)
        _write_json(
            gen_dir / "generated_answers_model-a.json",
            [
                {"question": "q1", "answer": "a"},
                {"question": "q2", "answer": "a"},
            ],
        )
        _write_json(
            gen_dir / "generated_answers_model-b.json",
            [
                {"question": "q1", "answer": "b"},
            ],
        )

        # Both models active: only q1 is answered by all.
        answered = answered_questions(str(gen_dir), {"model-a", "model-b"})
        assert answered == {"q1"}, answered

        # Only model-a active: q1 and q2 are answered.
        answered_a = answered_questions(str(gen_dir), {"model-a"})
        assert answered_a == {"q1", "q2"}, answered_a

        # Active model with no file → nothing counts as answered by all.
        answered_missing = answered_questions(str(gen_dir), {"model-a", "model-c"})
        assert answered_missing == set(), answered_missing

        # No active models → empty.
        assert answered_questions(str(gen_dir), set()) == set()

    print("  ✅  answered_questions intersection OK")


def test_sampling_excludes_answered() -> None:
    """Sampling skips already-answered questions and draws only available ones."""
    qa = [
        {
            "contextualized_question": f"q{i}",
            "answer_content": "ref",
            "data_categories": ["kat-a"],
        }
        for i in range(5)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "kb.json"
        mapping = Path(tmp) / "mapping.json"
        _write_json(src, qa)
        _write_json(mapping, _MAPPING)

        excluded = {"q0", "q1", "q2"}
        sampled = sample_by_overkategori(
            str(src), str(mapping), n_per_overkategori=10, exclude_questions=excluded
        )
        drawn = {r["contextualized_question"] for r in sampled}
        assert drawn == {"q3", "q4"}, drawn
        assert all(q not in drawn for q in excluded)

        # No exclusion → all five available.
        all_sampled = sample_by_overkategori(
            str(src), str(mapping), n_per_overkategori=10
        )
        assert len(all_sampled) == 5, len(all_sampled)

        # Everything answered → empty result (terminal "no more questions" state).
        empty = sample_by_overkategori(
            str(src),
            str(mapping),
            n_per_overkategori=10,
            exclude_questions={f"q{i}" for i in range(5)},
        )
        assert empty == [], empty

    print("  ✅  sampling excludes answered questions OK")


def main() -> None:
    """Runs all sampling unit tests."""
    failures: list[str] = []
    for test_fn in [
        test_answered_questions_intersection,
        test_sampling_excludes_answered,
    ]:
        try:
            test_fn()
        except Exception as exc:
            print(f"  ❌  {test_fn.__name__}: {type(exc).__name__}: {exc}")
            failures.append(test_fn.__name__)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All sampling tests passed.")


if __name__ == "__main__":
    main()
