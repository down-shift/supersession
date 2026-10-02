"""Frozen downstream_transfer_v1 histories and matched interventions."""
import hashlib, json, random
from pathlib import Path
from collections import defaultdict

SCHEMA = "downstream_transfer_v1"
SEEDS = {"development": 20261030, "frozen_gate": 20261031, "confirmatory": 20261032}
COUNTS = {"development": 24, "frozen_gate": 24, "confirmatory": 96}
QUERIES = ("current_x", "current_z")
VALUES = ("amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac","maple","navy","pearl","rust")
CODES = ("K7","M2","R5","T9","B4","C8","D3","F6","G1","H9","J2","L5","N8","P3","Q6","V4")

def template_hash():
    files = (Path(__file__), Path("src/data/supersession_behavior.py"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()

def signature(row):
    mv=row.get("matching_values", {})
    return (mv.get("initial_x",row.get("old_x")), mv.get("initial_z",row.get("old_z")),
            mv.get("current_x",row.get("current_x")), mv.get("current_z",row.get("current_z")))

def render(row, tokenizer=None, chat=True):
    vals=dict(row["matching_values"])
    if row.get("stage")=="confirmatory" and row.get("pair_direction")==1:
        binding=row["edited_binding"]
        target=("initial_"+binding[-1]) if binding.startswith("old_") else binding
        vals[target]=row["replacement_value"]
    x,z=row["variables"]; cb=row["codebook"]
    lines=[f"The badge assigned to {x} was {vals['initial_x']}.",
           f"The badge assigned to {z} was {vals['initial_z']}.",
           f"Later, {x}'s badge was changed to {vals['current_x']}.",
           f"Later, {z}'s badge was changed to {vals['current_z']}."]
    entity=x if row["query_id"]=="current_x" else z
    lines += [f"Codebook: "+", ".join(f"{v} -> {cb[v]}" for v in sorted(cb)),
              f"Which code corresponds to {entity}'s current badge?"]
    from src.data.supersession_behavior import _answer_prefix
    return _answer_prefix("\n".join(lines), tokenizer, chat)

def generate(stage,n,values=VALUES,codes=CODES,seed=None,excluded=()):
    if stage not in COUNTS or n!=COUNTS[stage]: raise ValueError("invalid stage or frozen history count")
    seed=SEEDS[stage] if seed is None else seed
    if seed!=SEEDS[stage]: raise ValueError("stage seeds are frozen")
    if any(r.get("seed")==seed for r in excluded): raise ValueError("stage seed overlaps an excluded dataset")
    if len(values)!=len(codes) or len(set(values))!=len(values) or len(set(codes))!=len(codes): raise ValueError("value/code vocabularies must be equal-length and unique")
    rng=random.Random(seed); seen={signature(r) for r in excluded}; rows=[]
    for i in range(n):
        while True:
            a,b,c,d=rng.sample(list(values),4)
            perm=list(codes); rng.shuffle(perm); cb=dict(zip(values,perm))
            probe={"matching_values":{"initial_x":a,"initial_z":b,"current_x":c,"current_z":d},"codebook":cb}
            sig=signature(probe)
            if sig not in seen: seen.add(sig); break
        mv={"initial_x":a,"initial_z":b,"current_x":c,"current_z":d}
        # Independent replacement draws; replacement is distinct from all active bindings.
        repl=rng.sample([v for v in values if v not in mv.values()],4)
        hid=f"downstream_transfer_{stage}_{seed}_{i:06d}"
        base={"schema":SCHEMA,"stage":stage,"history_id":hid,"history_index":i,"seed":seed,
              "variables":["Nora","Owen"] if i%2==0 else ["Owen","Nora"],"variable_pair":["x","z"],"attribute":"badge",
              "matching_values":mv,"codebook":cb,"value_vocabulary":list(values),"code_vocabulary":list(codes),
              "orientation":i%2}
        if stage!="confirmatory":
            for q in QUERIES:
                val=mv[q]; rows.append({**base,"record_type":"competence","query_id":q,"answer_code":cb[val],
                  "correct_value":val,"roles":{"target":cb[val]},"example_id":hid+":"+q})
        else:
            for binding,source,replacement in (("old_x",a,repl[0]),("old_z",b,repl[1]),("current_x",c,repl[2]),("current_z",d,repl[3])):
                for q in QUERIES:
                    answer=cb[mv[q]]
                    pid=f"{hid}:{binding}:{q}"
                    for direction in (0,1):
                        pair_answer=answer
                        if binding.startswith("current_") and q==binding:
                            pair_answer=cb[replacement if direction else source]
                        rows.append({**base,"record_type":"matched_edit","query_id":q,"pair_id":pid,
                          "pair_direction":direction,"edited_binding":binding,"source_value":source,
                          "replacement_value":replacement,"answer_code":pair_answer,"correct_value":pair_answer,
                          "roles":{"target":pair_answer},"example_id":f"{pid}:{direction}"})
    audit(rows)
    return rows

def audit(rows):
    if not rows: raise ValueError("empty dataset")
    stage=rows[0]["stage"]; expect=set(QUERIES) if stage!="confirmatory" else {(b,q,d) for b in ("old_x","old_z","current_x","current_z") for q in QUERIES for d in (0,1)}
    histories=defaultdict(dict)
    for r in rows:
        if r["schema"]!=SCHEMA or r["stage"]!=stage: raise ValueError("mixed protocol/stage")
        if len(set(r["codebook"].values()))!=len(r["codebook"]): raise ValueError("codebook is not one-to-one")
        if len(r["codebook"])!=len(r["value_vocabulary"]) or set(r["codebook"])!=set(r["value_vocabulary"]): raise ValueError("codebook vocabulary mismatch")
        if stage!="confirmatory" and any(k in r for k in ("pair_id","edited_binding","replacement_value","candidate_logits","derived_transfer")): raise ValueError("causal metadata leaked into competence stage")
        key=(r["edited_binding"],r["query_id"],r["pair_direction"]) if stage=="confirmatory" else r["query_id"]
        if key in histories[r["history_id"]]: raise ValueError("duplicate cell")
        histories[r["history_id"]][key]=r
    if len(histories)!=COUNTS[stage]: raise ValueError("wrong history count")
    for hid,cells in histories.items():
        if set(cells)!=expect: raise ValueError(f"incomplete cells: {hid}")
        if stage=="confirmatory":
            for b in ("old_x","old_z","current_x","current_z"):
                for q in QUERIES:
                    base,edit=cells[b,q,0],cells[b,q,1]
                    if (base["query_id"],base["codebook"],base["matching_values"]) != (edit["query_id"],edit["codebook"],edit["matching_values"]): raise ValueError("matched pair context mismatch")
                    if base["source_value"]==base["replacement_value"] or base["codebook"][base["source_value"]]==base["codebook"][base["replacement_value"]]: raise ValueError("source/replacement codes must differ")
                    stale=b.startswith("old_")
                    if stale:
                        if base["answer_code"]!=edit["answer_code"] or base["matching_values"]["current_"+b[-1]]!=edit["matching_values"]["current_"+b[-1]]: raise ValueError("stale edit changes current answer/binding")
                    elif q==b:
                        if base["answer_code"]!=base["codebook"][base["source_value"]] or edit["answer_code"]!=edit["codebook"][edit["replacement_value"]]: raise ValueError("live intervention answer orientation error")
    return True

def validated_stage(rows):
    audit(rows); return rows[0]["stage"]

def seal_artifact(payload):
    b=dict(payload); b.pop("seal",None)
    return {**b,"seal":hashlib.sha256(json.dumps(b,sort_keys=True,separators=(",",":")).encode()).hexdigest()}

def verify_sealed_artifact(doc):
    b=dict(doc); seal=b.pop("seal",None)
    if seal!=hashlib.sha256(json.dumps(b,sort_keys=True,separators=(",",":")).encode()).hexdigest(): raise ValueError("sealed artifact mismatch")
    return b
