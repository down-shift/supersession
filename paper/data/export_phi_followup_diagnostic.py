#!/usr/bin/env python3
"""Export strict and exploratory output-prefix counts for Phi follow-up runs.

The prefix categories are a post hoc text diagnostic. They do not replace the
frozen complete-answer parser or its exact-match accuracy.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/followups"
DEST = Path(__file__).resolve().parent / "phi_followup_output_diagnostic.csv"
VALUES = re.compile(r"(?i)^(amber|coral|jade|pearl|slate|teal|violet|ivory)")

rows = []
for n in (2, 4, 6):
    scores = [json.loads(line) for line in (SOURCE / f"phi4_mini_development_n{n}_scores.jsonl").open()]
    counts = {"strict_exact_correct": sum(row["answer_category"] == "correct" for row in scores),
              "correct_candidate_prefix": 0, "stale_candidate_prefix": 0,
              "other_candidate_prefix": 0, "no_candidate_prefix": 0}
    for row in scores:
        output = (row.get("generated_answer") or "").lstrip()
        match = VALUES.match(output)
        if not match:
            counts["no_candidate_prefix"] += 1
        elif match.group(1).lower() == row["answer"].lower():
            counts["correct_candidate_prefix"] += 1
        elif match.group(1).lower() == (row.get("stale_value") or "").lower():
            counts["stale_candidate_prefix"] += 1
        else:
            counts["other_candidate_prefix"] += 1
    rows.append({"n_distractors": n, "n_members": len(scores), **counts})

DEST.parent.mkdir(parents=True, exist_ok=True)
with DEST.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
