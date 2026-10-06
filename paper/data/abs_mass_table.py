#!/usr/bin/env python3
"""LaTeX rows of Table tab:b-abs-mass from paper/data/review_round1_estimates.json (reviewer item C1)."""
import json
from pathlib import Path

DATA = json.loads((Path(__file__).resolve().parent / "review_round1_estimates.json").read_text())
CONDITIONS = (("live", "Live"), ("superseded", "Superseded"), ("other_attribute", "Other attribute"),
              ("entity_mention", "Entity mention"), ("early_unassigned", "Early unassigned"),
              ("late_unassigned", "Late unassigned"))


def fmt(x):
    return f"{x:.3f}".replace("-", "−")


for model in ("Qwen3-8B", "Gemma 3 4B"):
    d = DATA["C1_absolute_log_masses"][model]
    print(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{model}}}}} \\")
    for key, label in CONDITIONS:
        cells = []
        for role in ("source", "donor", "current"):
            for side in ("baseline", "edited"):
                c = d[f"{key}|{side}|{role}"]
                cells.append(rf"\shortstack[r]{{{fmt(c['mean'])}\\[-1pt]\tiny({fmt(c['q25'])}, {fmt(c['q75'])})}}")
        print(f"{label} & " + " & ".join(cells) + r" \\")
