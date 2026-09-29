#!/usr/bin/env python3
"""Fail closed on history/query/value/edit imbalance in the new design."""
import argparse,collections
from src.data.io import read_jsonl

p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--exclude-dataset",help="fail if a concrete history also occurs in this dataset"); p.add_argument("--role-tolerance",type=int,default=1); p.add_argument("--transition-tolerance",type=int,default=1); p.add_argument("--min-targets-per-source",type=int,default=2); a=p.parse_args()
rows=read_jsonl(a.dataset); by_history=collections.defaultdict(list)
if not rows: raise SystemExit("dataset is empty")
for r in rows: by_history[r["history_id"]].append(r)
is_pairs="pair_id" in rows[0] if rows else False
def history_key(r):
    return (tuple(r["variables"]),tuple(r["order"]),*(r[k] for k in ("old_x","old_z","current_x","current_z")))
keys={}
for hist,members in by_history.items():
    baseline=next((r for r in members if not is_pairs or r["pair_direction"]==0),None)
    if baseline is None: raise SystemExit(f"{hist} has no baseline record")
    key=history_key(baseline)
    if key in keys: raise SystemExit(f"duplicate concrete histories: {keys[key]} and {hist}")
    keys[key]=hist
if a.exclude_dataset:
    excluded=read_jsonl(a.exclude_dataset)
    excluded_keys={history_key(r) for r in excluded if "pair_id" not in r or r["pair_direction"]==0}
    overlap=keys.keys() & excluded_keys
    if overlap: raise SystemExit(f"{len(overlap)} concrete histories overlap {a.exclude_dataset}")
if is_pairs:
    by_pair=collections.defaultdict(list); sources=collections.Counter(); targets=collections.Counter(); transitions=collections.Counter(); target_sets=collections.defaultdict(set)
    seen_edit=set()
    for r in rows:
        by_pair[r["pair_id"]].append(r)
        edit_key=(r["history_id"],r["edited_binding"])
        if r["pair_direction"]==0 and edit_key not in seen_edit:
            sources[(r["edited_binding"],r["source_value"])]+=1
            targets[(r["edited_binding"],r["replacement_value"])]+=1
            transitions[(r["edited_binding"],r["source_value"],r["replacement_value"])]+=1
            target_sets[(r["edited_binding"],r["source_value"])].add(r["replacement_value"])
            seen_edit.add(edit_key)
    for pid,members in by_pair.items():
        if len(members)!=2 or {r["pair_direction"] for r in members}!={0,1}: raise SystemExit(f"invalid pair members: {pid}")
    for hist,members in by_history.items():
        if len(members)!=32: raise SystemExit(f"{hist} has {len(members)} pair rows; expected 32")
        expected={(binding,query,direction) for binding in ("old_x","old_z","current_x","current_z") for query in ("current_x","initial_x","current_z","initial_z") for direction in (0,1)}
        observed={(r["edited_binding"],r["query_id"],r["pair_direction"]) for r in members}
        if observed!=expected: raise SystemExit(f"{hist} has incomplete binding/query/direction coverage")
    for binding in ("old_x","old_z","current_x","current_z"):
        counts=[n for (role,_),n in sources.items() if role==binding]
        if counts and max(counts)-min(counts)>a.role_tolerance: raise SystemExit(f"unbalanced source role {binding}: {counts}")
        counts=[n for (role,_),n in targets.items() if role==binding]
        if counts and max(counts)-min(counts)>a.role_tolerance: raise SystemExit(f"unbalanced target role {binding}: {counts}")
    trans=[n for (_,_,_),n in transitions.items()]
    if trans and max(trans)-min(trans)>a.transition_tolerance: raise SystemExit(f"edit transition imbalance exceeds tolerance: range {min(trans)}..{max(trans)}")
    deficient={k:sorted(v) for k,v in target_sets.items() if len(v)<a.min_targets_per_source}
    if deficient: raise SystemExit(f"each source must use at least {a.min_targets_per_source} replacement targets; failures: {deficient}")
else:
    for hist,members in by_history.items():
        if len(members)!=4 or {r["query_id"] for r in members}!={"current_x","initial_x","current_z","initial_z"}:
            raise SystemExit(f"{hist} does not contain exactly the four fixed queries")
    roles=collections.Counter((r["query_id"],r["answer"]) for r in rows)
    values={r["answer"] for r in rows}
    for query in ("current_x","initial_x","current_z","initial_z"):
        counts=[roles[(query,v)] for v in values]
        if counts and max(counts)-min(counts)>a.role_tolerance: raise SystemExit(f"unbalanced answer values for {query}: {counts}")
    if any(len(set(r[k] for k in ("old_x","old_z","current_x","current_z")))!=4 for r in rows):
        raise SystemExit("history contains colliding assignment values")
print({"rows":len(rows),"histories":len(by_history),"paired":is_pairs,"audit":"passed"})
