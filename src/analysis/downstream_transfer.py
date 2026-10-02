"""History-level derived-code transfer analysis."""
from collections import defaultdict
import numpy as np
from src.analysis.query_reactivation import summarize_histories as _summarize_histories

def summarize_histories(values,seed=20261033,n_boot=10000):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite history statistic; histories must not be silently excluded")
    if n_boot < 1:
        raise ValueError("bootstrap draws must be positive")
    return _summarize_histories(x, seed, n_boot)

def sequence_logprob(model,tokenizer,prompt,candidate,chat=False):
    """Teacher-force every token in a complete candidate continuation."""
    from src.cross_model.scoring import score_prompt
    prefix = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full = list(tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"])
    if not prefix or len(full) <= len(prefix) or full[:len(prefix)] != prefix:
        raise ValueError("candidate is not a stable continuation at prompt boundary")
    events = {candidate: [{"text": candidate, "ids": full[len(prefix):]}]}
    masses, _, _, _ = score_prompt(model, tokenizer, prompt, events)
    return masses[candidate]

def score_protocol_record(row,scorer,cache,stage):
    if row["stage"] != stage:
        raise ValueError("score stage differs from dataset stage")
    scored=scorer(row)
    if stage!="confirmatory":
        for k in ("candidate_logprobs", "candidate_logits", "derived_transfer", "candidate_accuracy",
                  "identity_transfer", "matched_edit_effect", "candidate_probabilities"):
            scored.pop(k, None)
        scored["causal_effects_computed"]=False; return scored
    if row["pair_direction"]==0:
        scored["causal_effects_computed"] = False
        cache[row["pair_id"]]=scored
        return scored
    base=cache.get(row["pair_id"])
    if base is None: raise ValueError("edited pair member lacks baseline")
    if base["codebook"]!=row["codebook"] or base["query_id"]!=row["query_id"]: raise ValueError("pair codebook/query mismatch")
    cs=base["codebook"][row["source_value"]]; cr=base["codebook"][row["replacement_value"]]
    lb,le=base["candidate_logprobs"],scored["candidate_logprobs"]
    scored["derived_transfer"]=(le[cr]-le[cs])-(lb[cr]-lb[cs]); scored["causal_effects_computed"]=True
    return scored

def derive_cells(rows):
    pairs=defaultdict(dict)
    for r in rows:
        direction = r["pair_direction"]
        if direction not in (0, 1) or direction in pairs[r["pair_id"]]:
            raise ValueError("invalid/duplicate pair direction")
        pairs[r["pair_id"]][direction] = r
    cells=defaultdict(dict)
    for pid,m in pairs.items():
        if set(m)!={0,1}: raise ValueError(f"incomplete pair {pid}")
        b,e=m[0],m[1]
        for k in ("history_id","edited_binding","query_id","source_value","replacement_value","codebook"):
            if b[k]!=e[k]: raise ValueError("matched pair metadata differs")
        cs=b["codebook"][b["source_value"]]; cr=b["codebook"][b["replacement_value"]]
        val=(e["candidate_logprobs"][cr]-e["candidate_logprobs"][cs])-(b["candidate_logprobs"][cr]-b["candidate_logprobs"][cs])
        if not np.isfinite(val):
            raise ValueError("nonfinite paired derived effect")
        key=(b["edited_binding"],b["query_id"])
        if key in cells[b["history_id"]]: raise ValueError("duplicate history cell")
        cells[b["history_id"]][key]=float(val)
    return cells

def history_contrasts(rows):
    cells=derive_cells(rows); out=[]
    for hid,e in sorted(cells.items()):
        required={(b,q) for b in ("old_x","old_z","current_x","current_z") for q in ("current_x","current_z")}
        if set(e)!=required: raise ValueError("history missing binding/query cells")
        stale=.5*((e["old_x","current_x"]-e["old_x","current_z"])+(e["old_z","current_z"]-e["old_z","current_x"]))
        live=.5*((e["current_x","current_x"]-e["current_x","current_z"])+(e["current_z","current_z"]-e["current_z","current_x"]))
        out.append({"history_id":hid,"R_stale_derived":stale,"R_live_derived":live})
    return out

def evaluate_competence_gate(rows,scores,threshold=.97):
    from src.data.downstream_transfer import audit
    audit(rows)
    if rows[0]["stage"] != "frozen_gate" or threshold != .97:
        raise ValueError("frozen gate requires the frozen stage and 0.97 threshold")
    exp={r["example_id"]:r for r in rows}; got={r["example_id"]:r for r in scores}
    if len(exp)!=len(rows) or set(exp)!=set(got) or len(got)!=len(scores): raise ValueError("gate scores must exactly cover dataset")
    groups=defaultdict(list); unrestricted=defaultdict(list); diag=defaultdict(list)
    for eid,r in exp.items():
        s=got[eid]
        if any(s.get(k)!=v for k,v in r.items()): raise ValueError("score metadata differs")
        if s.get("causal_effects_computed") is not False or any(k in s for k in (
                "candidate_logprobs", "candidate_logits", "derived_transfer", "identity_transfer", "matched_edit_effect")):
            raise ValueError("competence score exposes causal output")
        acc=s.get("current_code_accuracy")
        if acc not in (0,1): raise ValueError("invalid current code accuracy")
        groups["current"].append(acc); diag[(r["query_id"],r["orientation"])].append(acc)
        generated=s.get("unrestricted_code_accuracy")
        if generated not in (0,1) or not isinstance(s.get("unrestricted_generated_text"),str):
            raise ValueError("missing unrestricted greedy-generation competence diagnostic")
        unrestricted["current"].append(generated)
    candidate_accuracy=float(np.mean(groups["current"])) if groups["current"] else 0.0
    generated_accuracy=float(np.mean(unrestricted["current"])) if unrestricted["current"] else 0.0
    return {"pass":len(groups["current"])>0 and candidate_accuracy>=threshold and generated_accuracy>=threshold,"threshold":threshold,
      "n":len(groups["current"]),"current_derived_code_accuracy":float(np.mean(groups["current"])),
      "current_unrestricted_generation_accuracy":generated_accuracy,
      "competence_rule":"candidate sequence rank and greedy unrestricted generation must each meet threshold",
      "query_orientation_diagnostics":{"|".join(map(str,k)):{"n":len(v),"accuracy":float(np.mean(v))} for k,v in sorted(diag.items())},
      "causal_effects_computed":False}

def current_answer_stability(rows,scores):
    # Uses baseline/edit stale cells only; all observations remain included.
    byid={r["example_id"]:r for r in scores}; pairs=defaultdict(dict)
    for r in rows:
        if r.get("edited_binding") not in ("old_x","old_z"): continue
        pairs[r["pair_id"]][r["pair_direction"]]=byid[r["example_id"]]
    out=[]
    for pid,m in pairs.items():
        b,e=m[0],m[1]; code=b["answer_code"]
        blp, elp = b["candidate_logprobs"], e["candidate_logprobs"]
        bm = blp[code] - max(v for k, v in blp.items() if k != code)
        em = elp[code] - max(v for k, v in elp.items() if k != code)
        out.append({"history_id":b["history_id"], "pair_id":pid,
          "edited_binding":b["edited_binding"], "query_id":b["query_id"],
          "baseline_accuracy":b["current_code_accuracy"],"edited_accuracy":e["current_code_accuracy"],
          "baseline_correct_logprob":blp[code],"edited_correct_logprob":elp[code],
          "baseline_margin":bm,"edited_margin":em,
          "delta_correct_logprob":elp[code]-blp[code],"delta_margin":em-bm})
    return out


def validate_scores(rows, scores, stage, complete=True):
    """Check exact metadata and stage separation, including resumed records."""
    expected = {r["example_id"]: r for r in rows}
    ids = set()
    for score in scores:
        eid = score.get("example_id")
        if eid not in expected or eid in ids:
            raise ValueError("unexpected/duplicate score record")
        ids.add(eid)
        row = expected[eid]
        if any(score.get(k) != value for k, value in row.items()):
            raise ValueError(f"score metadata differs from dataset: {eid}")
        if score.get("current_code_accuracy") not in (0, 1):
            raise ValueError("invalid current-code accuracy")
        if stage != "confirmatory":
            if score.get("causal_effects_computed") is not False or any(k in score for k in (
                    "candidate_logprobs", "candidate_logits", "derived_transfer", "identity_transfer")):
                raise ValueError("competence score exposes causal data")
        else:
            lp = score.get("candidate_logprobs", {})
            if set(lp) != set(row["code_vocabulary"]) or not all(np.isfinite(v) for v in lp.values()):
                raise ValueError("invalid/nonfinite candidate sequence scores")
            # Ties do not count as competence, avoiding vocabulary-order bias.
            accuracy = int(all(lp[row["answer_code"]] > v for k, v in lp.items() if k != row["answer_code"]))
            if score["current_code_accuracy"] != accuracy:
                raise ValueError("saved accuracy differs from candidate sequence scores")
            if score.get("causal_effects_computed") is not bool(row["pair_direction"]):
                raise ValueError("pair direction/causal scoring flag mismatch")
    if complete and ids != set(expected):
        raise ValueError("scores do not exactly cover the dataset")
    return True
