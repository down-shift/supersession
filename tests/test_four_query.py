from collections import Counter,defaultdict
import json
import os
import re
import subprocess
import sys
from pathlib import Path

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
from src.data.progress import append_jsonl_record,prepare_jsonl_progress
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

def test_calibration_and_confirmatory_histories_are_unique_and_disjoint():
    confirm=dataset()
    calibration=make_histories(192,seed=10041,values=VALUES,variables=VARIABLES,partition="calibration")
    def identity(h):
        return (tuple(h["variables"]),tuple(h["order"]),h["old_x"],h["old_z"],h["current_x"],h["current_z"])
    confirm_keys={identity(h) for h in confirm}; calibration_keys={identity(h) for h in calibration}
    assert len(confirm_keys)==144
    assert len(calibration_keys)==192
    assert confirm_keys.isdisjoint(calibration_keys)
    index={value:i for i,value in enumerate(VALUES)}
    offsets=lambda rows:{(index[h["old_z"]]-index[h["old_x"]])%len(VALUES) for h in rows}
    assert offsets(confirm)==offsets(calibration)==set(range(2,len(VALUES),2))
    for role in ("old_x","old_z","current_x","current_z"):
        assert Counter(h[role] for h in calibration)==Counter({v:16 for v in VALUES})

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
            queried=edit["query"]
            other="z" if queried=="x" else "x"
            assert edit["roles"]["query_old"]==edit[f"old_{queried}"]
            assert edit["roles"]["other_old"]==edit[f"old_{other}"]
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

def test_scoring_checkpoint_resumes_and_discards_truncated_final_record(tmp_path):
    dataset=tmp_path/"data.jsonl"; dataset.write_text('{"example_id":"a"}\n{"example_id":"b"}\n')
    token_ids=tmp_path/"tokens.json"; token_ids.write_text('{"token_ids":{"elm":1}}')
    output=tmp_path/"scores.jsonl"; config={"model":{"id":"test"}}
    rows=[{"example_id":"a"},{"example_id":"b"}]
    assert prepare_jsonl_progress(output,dataset,token_ids,config,rows)==set()
    append_jsonl_record(output,{"example_id":"a","score":1})
    with output.open("ab") as f: f.write(b'{"example_id":"b"')
    assert prepare_jsonl_progress(output,dataset,token_ids,config,rows,resume=True)=={"a"}
    assert output.read_text()=='{"example_id": "a", "score": 1}\n'
    with pytest.raises(ValueError,match="fingerprint differs in config"):
        prepare_jsonl_progress(output,dataset,token_ids,{"model":{"id":"changed"}},rows,resume=True)

def test_checkpoint_token_metadata_migration_requires_explicit_verified_match(tmp_path):
    dataset=tmp_path/"data.jsonl"; dataset.write_text('{"example_id":"a"}\n')
    token_ids=tmp_path/"tokens.json"; token_ids.write_text('{"token_ids":{"elm":1},"timestamp":"new"}')
    output=tmp_path/"scores.jsonl"; config={"model":{"id":"test"}}; rows=[{"example_id":"a"}]
    prepare_jsonl_progress(output,dataset,token_ids,config,rows)
    manifest=tmp_path/"scores.jsonl.run.json"
    saved=json.loads(manifest.read_text()); saved["token_ids_sha256"]="old-metadata-hash"; manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match="token_ids_sha256"):
        prepare_jsonl_progress(output,dataset,token_ids,config,rows,resume=True)
    assert prepare_jsonl_progress(output,dataset,token_ids,config,rows,resume=True,allow_token_ids_rehash=True)==set()

def test_patch_analyzer_supports_x_only_discovery_and_full_heldout(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts/analyze_four_query_patching.py"
    rows=[]
    for hid in ("h1","h2"):
        for binding in ("old_x","current_x","old_z","current_z"):
            for query in ("current_x","initial_x","current_z","initial_z"):
                for layer in (0,1,2):
                    for direction in ("donor_to_recipient","recipient_to_donor"):
                        rows.append({"history_id":hid,"edited_binding":binding,"query_id":query,"direction":direction,"site":"final_preanswer","layer":layer,"patch_delta_toward_donor":10. if (binding,query) in {("old_x","initial_x"),("current_x","current_x"),("old_z","initial_z"),("current_z","current_z")} else 1.})
    patches=tmp_path/"patches.jsonl"
    patches.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    patches_x=tmp_path/"patches_x.jsonl"
    patches_x.write_text("".join(json.dumps(r)+"\n" for r in rows if r["edited_binding"] in ("old_x","current_x")),encoding="utf8")
    env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1])}
    discovery=tmp_path/"discovery"; discovery.mkdir()
    subprocess.run([sys.executable,str(script),"--patches",str(patches_x),"--output-dir",str(discovery),"--stage","discovery"],check=True,env=env,capture_output=True,text=True)
    d=json.loads((discovery/"four_query_patch_summary.json").read_text())
    assert set(d["cells_present"])=={f"{b}/{q}" for b in ("old_x","current_x") for q in ("current_x","initial_x","current_z","initial_z")}
    assert "S_x_control" in d["results"]["final_preanswer/layer_0"]
    assert d["selection"]["selection_statistic"]["name"]=="R_x_patch"
    assert d["selection"]["selection_statistic"]["selected_layers"]==[0,1,2]
    heldout=tmp_path/"heldout"; heldout.mkdir()
    subprocess.run([sys.executable,str(script),"--patches",str(patches),"--output-dir",str(heldout),"--stage","heldout"],check=True,env=env,capture_output=True,text=True)
    h=json.loads((heldout/"heldout_patch_summary.json").read_text())
    result=h["results"]["final_preanswer/layer_0"]
    assert "symmetric_R_patch" in result and "S_x_control" in result and "S_z_control" in result

def test_all_position_patch_analysis_keeps_absolute_positions_separate(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts/analyze_four_query_patching.py"
    rows=[]
    for hid in ("h1","h2"):
        for query,value in (("current_x",4.),("current_z",1.)):
            for layer in (0,1,2):
                for position,role in ((7,"other_position"),(8,"other_position"),(9,"final_preanswer")):
                    for direction in ("baseline_to_edited","edited_to_baseline"):
                        rows.append({"history_id":hid,"edited_binding":"old_x","query_id":query,"direction":direction,"site_role":role,"site":role,"position":position,"token_id":position+100,"token_text":f"tok{position}","layer":layer,"patch_delta_toward_donor":value+position/100})
    patches=tmp_path/"positions.jsonl"; patches.write_text("".join(json.dumps(r)+"\n" for r in rows))
    out=tmp_path/"analysis"; out.mkdir(); env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1]),"MPLBACKEND":"Agg","MPLCONFIGDIR":str(tmp_path/"mpl")}
    subprocess.run([sys.executable,str(script),"--patches",str(patches),"--output-dir",str(out),"--stage","discovery"],check=True,env=env,capture_output=True,text=True)
    import pandas as pd
    matrix=pd.read_csv(out/"all_positions_Rx.csv")
    assert set(matrix.position)=={7,8,9}
    assert len(matrix)==9  # three layers × three absolute positions
    assert set(matrix['count'])=={2}
    assert set(pd.read_csv(out/"all_positions_token_legend.csv").site_role)=={"other_position","final_preanswer"}
