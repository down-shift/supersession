from collections import defaultdict
import json, re, subprocess, sys
from pathlib import Path
import pytest
from src.data.generate import PROMPT_VARIANTS, make_histories, expand_history_queries, render_example, query_variable_span, legal_orders, matched_history_pairs
from src.analysis.prompt_development import classify_prediction, temporal_gap, select_prompt
from src.data.progress import prepare_jsonl_progress

VALUES=("amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac")
def histories(partition="prompt_dev", n=96):
    hs=make_histories(n,seed=2026,values=VALUES,variables=[["x","z"],["p","q"],["u","v"],["m","n"]],partition=partition)
    return [{**h,"candidate_values":list(VALUES)} for h in hs]

def test_canonical_first_latest_and_all_variants_have_query_spans():
    h=histories()[0]
    for variant in PROMPT_VARIANTS:
        for ex in expand_history_queries({**h,"prompt_variant":variant}):
            prompt=render_example(ex,chat=False)
            assert "\n\n" not in prompt and ":=" not in prompt and "Set " not in prompt
            start,end=query_variable_span(ex,prompt)
            assert prompt[start:end]==ex["variables"][0 if ex["query"]=="x" else 1]
            if variant=="first_latest":
                assert all(re.fullmatch(r"[a-z]+ = [a-z]+",line) for line in prompt.splitlines()[:4])

def test_prompt_variants_are_paired_and_only_change_temporal_framing():
    h=histories()[0]
    prompts={v:render_example(expand_history_queries({**h,"prompt_variant":v})[0],chat=False) for v in PROMPT_VARIANTS}
    assert all(h["old_x"] in prompts[v] and h["current_x"] in prompts[v] for v in PROMPT_VARIANTS)
    assert "Initial assignment:" in prompts["initial_update"] and "Update:" in prompts["initial_update"]
    assert "t0:" in prompts["timestamped"] and "t1:" in prompts["timestamped"]
    assert prompts["first_latest"]!=prompts["initial_update"]!=prompts["timestamped"]

def test_three_partition_concrete_histories_are_disjoint():
    sets=[]
    for part,n in (("prompt_dev",96),("gate",192),("confirmatory",288)):
        sets.append({(tuple(h["variables"]),tuple(h["order"]),*(h[k] for k in ("old_x","old_z","current_x","current_z"))) for h in histories(part,n)})
    assert all(not (sets[i]&sets[j]) for i in range(3) for j in range(i))

def test_selection_prefers_first_and_never_timestamped():
    passing={v:{"meets_development_criterion":True} for v in PROMPT_VARIANTS}
    assert select_prompt(passing)["selected_variant"]=="first_latest"
    passing["first_latest"]["meets_development_criterion"]=False
    assert select_prompt(passing)["selected_variant"]=="initial_update"
    assert select_prompt({"timestamped":{"meets_development_criterion":True}})["confirmatory_permitted"] is False

def test_error_taxonomy_five_classes():
    base={"query_id":"initial_x","answer":"amber","old_x":"amber","current_x":"birch","old_z":"coral","current_z":"denim","generated_first_token":"","order":["O_x","O_z","C_x","C_z"]}
    cases={"birch":"same_variable_other_time","coral":"other_variable_same_time","denim":"other_variable_other_time","frost":"other_history_candidate","zzz":"off_candidate"}
    for pred,expected in cases.items(): assert classify_prediction({**base,"generated_first_token":pred},VALUES)==expected

def test_temporal_gap_matches_all_legal_orders():
    for order in legal_orders():
        for var in "xz":
            expected=order.index("C_"+var)-order.index("O_"+var)-1
            assert temporal_gap({"query_id":"current_"+var,"order":order})==expected

def test_checkpoint_fingerprint_contains_prompt_variant_via_dataset(tmp_path):
    tok=tmp_path/"tok.json"; tok.write_text("{}")
    cfg={"model":{"id":"test"}}; rows=[{"example_id":"a"}]
    d1=tmp_path/"a.jsonl"; d1.write_text('{"prompt_variant":"first_latest"}\n')
    d2=tmp_path/"b.jsonl"; d2.write_text('{"prompt_variant":"initial_update"}\n')
    out=tmp_path/"scores.jsonl"; prepare_jsonl_progress(out,d1,tok,cfg,rows)
    with pytest.raises(ValueError,match="fingerprint differs in dataset_sha256"):
        prepare_jsonl_progress(out,d2,tok,cfg,rows,resume=True)

def test_matched_pair_roles_remain_independent():
    h=histories()[0]
    pairs=matched_history_pairs(h)
    assert len(pairs)==32
    for p in pairs:
        assert p["roles"]["target"]==p["answer"]

def test_gate_requires_full_vocab_accuracy_for_all_four_queries():
    from src.analysis.prompt_development import gate_allows_confirmatory
    passing={q:{"full_vocab_next_token_accuracy":1.0,"candidate_accuracy":1.0} for q in ("current_x","initial_x","current_z","initial_z")}
    assert gate_allows_confirmatory(passing,.99)
    passing["initial_x"]["full_vocab_next_token_accuracy"]=.98
    assert not gate_allows_confirmatory(passing,.99)

def test_candidate_validation_uses_shared_vocabulary_across_variants():
    from src.data.token_validation import validate_candidate_vocabulary
    class WordTokenizer:
        def __call__(self,text,add_special_tokens=False): return {"input_ids":[hash(w) for w in re.findall(r"\w+|[^\w\s]",text)]}
    examples=expand_history_queries(histories()[0])
    valid,_=validate_candidate_vocabulary(WordTokenizer(),examples,list(VALUES),chat=False,prompt_variants=PROMPT_VARIANTS)
    assert set(valid)==set(VALUES)

def test_gate_failure_does_not_authorize_confirmatory_workflow():
    from src.analysis.prompt_development import gate_allows_confirmatory
    result={q:{"full_vocab_next_token_accuracy":.98} for q in ("current_x","initial_x","current_z","initial_z")}
    if not gate_allows_confirmatory(result,.99): confirmatory_generated=False
    else: confirmatory_generated=True
    assert not confirmatory_generated

def test_assignment_substitutions_align_under_every_variant():
    from src.data.token_validation import validate_assignment_patching
    class WordTokenizer:
        def __call__(self,text,add_special_tokens=False): return {"input_ids":[hash(w) for w in re.findall(r"\w+|[^\w\s]",text)]}
    audit=validate_assignment_patching(WordTokenizer(),histories()[:1],VALUES,chat=False,prompt_variants=PROMPT_VARIANTS)
    assert audit["status"]=="passed"

def test_audit_accepts_multiple_exclusion_datasets(tmp_path):
    from src.data.io import write_jsonl
    root=Path(__file__).resolve().parents[1]
    paths=[]
    for part in ("prompt_dev","gate","confirmatory"):
        hs=make_histories(48,seed=73,values=VALUES,variables=[["x","z"],["p","q"],["u","v"],["m","n"]],partition=part)
        rows=[q for h in hs for q in expand_history_queries(h)]
        path=tmp_path/f"{part}.jsonl"; write_jsonl(rows,path); paths.append(path)
    proc=subprocess.run([sys.executable,str(root/"scripts/audit_four_query.py"),str(paths[2]),"--exclude-dataset",str(paths[0]),"--exclude-dataset",str(paths[1])],cwd=root,env={**__import__("os").environ,"PYTHONPATH":str(root)},capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert "'audit': 'passed'" in proc.stdout
