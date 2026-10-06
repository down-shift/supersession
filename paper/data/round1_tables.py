#!/usr/bin/env python3
"""LaTeX rows for the reviewer-round-1 appendix tables, from review_round1_estimates.json.

    --table accuracy   Table tab:c-accuracy  (C2: current-candidate rank accuracy, every model x experiment)
    --table errors     Table tab:f-olmo-errors (B7: what OLMo-2's generated answers named)
    --table olmo-r     Table tab:f-olmo-r    (B7: score-based R beside the generation contrasts)
"""
import argparse
import json
from pathlib import Path

DATA = json.loads((Path(__file__).resolve().parent / "review_round1_estimates.json").read_text())
CONDITIONS = (("live", "Live"), ("superseded", "Superseded"), ("other_attribute", "Other attribute"),
              ("entity_mention", "Entity mention"), ("early_unassigned", "Early unassigned"),
              ("late_unassigned", "Late unassigned"))
NAMES = dict(CONDITIONS) | {"marker": "", "superseded_marker0": "superseded, no marker"}


def fmt(x):
    return f"{x:.3f}".replace("-", "−")


def ci(e):
    return f"{fmt(e['mean'])} [{fmt(e['ci95'][0])}, {fmt(e['ci95'][1])}]"


def count(c):
    return f"{c['correct']:,}/{c['total']:,}"


SHORT = {"OLMo-2-7B (excluded, generation run)": "OLMo-2-7B (excluded)"}


def accuracy():
    for model, a in DATA["C2_exp1_accuracy"].items():
        model = SHORT.get(model, model)
        low = a["lowest_group"]
        print(f"{model} & Experiment 1 & {count(a)} & {NAMES.get(low, low.replace('_', ' '))} {count(a['by_group'][low])} \\\\")
    for model, exps in DATA["C2_followup_accuracy"].items():
        for exp, a in exps.items():
            low = a["lowest_group"]
            exp = exp.replace(" (96-history score run)", ", score run")
            print(f"{model} & {exp} & {count(a)} & {low.replace('_', ' ')} {count(a['by_group'][low])} \\\\")


def errors():
    e = DATA["B7_olmo"]["error_breakdown"]
    keys = ("correct", "source_value", "donor", "other_entity_current", "other_entity_earlier", "other_candidate", "invalid_format")
    for c, label in CONDITIONS:
        for side in ("baseline", "edited"):
            d = e[f"{c}|{side}"]
            print(f"{label if side == 'baseline' else ''} & {side} & " + " & ".join(f"{d.get(k, 0):,}" for k in keys) + r" \\")


def olmo_r():
    o = DATA["B7_olmo"]
    for c, label in CONDITIONS:
        g = o["generation_donor_minus_source"]
        print(f"{label} & {ci(o['score_R'][c])} & {ci(g[c + '|all'])} & {fmt(g[c + '|aligned']['mean'])} & {fmt(g[c + '|reversed']['mean'])} \\\\")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table", choices=("accuracy", "errors", "olmo-r"), required=True)
    {"accuracy": accuracy, "errors": errors, "olmo-r": olmo_r}[p.parse_args().table]()
