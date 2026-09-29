from collections import Counter,defaultdict
import re

import pytest

from src.analysis.metrics import trimmed_mean
from src.data.generate import (
    expand_history_queries,
    legal_orders,
    make_histories,
    matched_history_pairs,
    query_variable_span,
    render_example,
)
from src.data.token_validation import validate_assignment_patching
from src.experiments.patching import assert_aligned,partition_history_ids,patch_effect_metrics

VALUES=("amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac")
VARIABLES=(("x","z"),("p","q"),("u","v"),("m","n"))
QUERIES={"current_x":"current_x","initial_x":"old_x","current_z":"current_z","initial_z":"old_z"}

def dataset():
    histories=make_histories(144,seed=41,values=VALUES,variables=VARIABLES)
    for h in histories: h["candidate_values"]=list(VALUES)
    return histories

def test_four_queries_orientation_order_value_balance_and_unique_assignments():
    histories=dataset()
    assert len(histories)==144
    assert Counter(tuple(h["order"]) for h in histories)==Counter({order:24 for order in legal_orders()})
    assert Counter((tuple(h["variable_pair"]),h["orientation"]) for h in histories)==Counter({(pair,o):18 for pair in VARIABLES for o in (0,1)})
    for h in histories:
        assert len({h[k] for k in ("old_x","old_z","current_x","current_z")})==4
        queries=expand_history_queries(h)
        assert {q["query_id"] for q in queries}==set(QUERIES)
        assert len(queries)==4
    for key in ("old_x","old_z","current_x","current_z"):
        assert Counter(h[key] for h in histories)==Counter({v:12 for v in VALUES})

def test_matched_pairs_have_independent_roles_and_correct_answers_for_every_cell():
    for h in dataset()[:12]:
        pairs=matched_history_pairs(h)
        by_key=defaultdict(dict)
        for ex in pairs: by_key[(ex["edited_binding"],ex["query_id"])][ex["pair_direction"]]=ex
        assert len(by_key)==16
        for (binding,query),members in by_key.items():
            assert set(members)=={0,1}
            base,edit=members[0],members[1]
            base_answer=h[QUERIES[query]]
            edited_value=edit[binding]
            expected_edit_answer=edited_value if (
                (binding==f"current_{query[-1]}" and query.startswith("current_")) or
                (binding==f"old_{query[-1]}" and query.startswith("initial_"))
            ) else base_answer
            assert base["roles"]["target"]==base["answer"]==base_answer
            assert edit["roles"]["target"]==edit["answer"]==expected_edit_answer
            assert base[binding]==h[binding]
            assert edit[binding]==edit["replacement_value"]
            assert base["roles"][binding]==base[binding]
            assert edit["roles"][binding]==edit[binding]
            for other in ("old_x","old_z","current_x","current_z"):
                if other!=binding: assert base[other]==edit[other]
            assert base["roles"]["target"]==base_answer  # editing must not mutate its sibling

def test_each_source_uses_multiple_replacement_targets_and_pair_changes_one_assignment():
    targets=defaultdict(set)
    for h in dataset():
        for binding in ("old_x","old_z","current_x","current_z"):
            pair=matched_history_pairs(h)
            edit=next(r for r in pair if r["edited_binding"]==binding and r["query_id"]=="current_x" and r["pair_direction"]==1)
            targets[(binding,h[binding])].add(edit["replacement_value"])
            assert edit["replacement_value"] not in {h[k] for k in ("old_x","old_z","current_x","current_z") if k!=binding}
    assert all(len(v)>=2 for v in targets.values())

def test_query_variable_span_tracks_x_and_z_questions():
    h=dataset()[0]
    for query in ("current_x","initial_x","current_z","initial_z"):
        ex=next(q for q in expand_history_queries(h) if q["query_id"]==query)
        prompt=render_example(ex,chat=False)
        start,end=query_variable_span(ex,prompt)
        expected=ex["variables"][0 if ex["query"]=="x" else 1]
        assert prompt[start:end]==expected

def test_assignment_token_alignment_audit_and_single_position_contract():
    class WordTokenizer:
        def __call__(self,text,add_special_tokens=False):
            words=re.findall(r"\w+|[^\w\s]",text)
            return {"input_ids":[hash(w) for w in words]}
    audit=validate_assignment_patching(WordTokenizer(),dataset()[:2],VALUES,chat=False)
    assert audit["status"]=="passed"
    assert_aligned([1,2,3],[1,9,3],[1])
    with pytest.raises(ValueError): assert_aligned([1,2,3],[1,9,8],[1])

def test_trimmed_mean_sorts_before_trimming():
    assert trimmed_mean([100,1,3,2,4,5,6,7,8,9],.1)==pytest.approx(5.5)

def test_discovery_and_heldout_history_sets_are_disjoint_and_reproducible():
    ids=[f"h{i}" for i in range(144)]
    discovery,heldout=partition_history_ids(ids,24,seed=9)
    assert len(discovery)==24 and len(heldout)==120
    assert set(discovery).isdisjoint(heldout)
    assert set(discovery+heldout)==set(ids)
    assert partition_history_ids(ids,24,seed=9)==(discovery,heldout)

def test_patch_effect_orientation_is_donor_relative_in_both_directions():
    forward=patch_effect_metrics(donor_margin=8,recipient_margin=-2,patched_margin=3)
    reverse=patch_effect_metrics(donor_margin=2,recipient_margin=-8,patched_margin=-3)
    assert forward["patch_delta_toward_donor"]==5
    assert reverse["patch_delta_toward_donor"]==5
    assert forward["normalized_recovery"]==pytest.approx(.5)
    assert patch_effect_metrics(1,1+1e-9,2)["normalized_recovery"] is None
