"""Frozen downstream_transfer_v1 histories and matched interventions."""
import hashlib, json, random
import logging
from pathlib import Path
from collections import defaultdict
from src.cross_model.progress import progress

logger = logging.getLogger(__name__)

SCHEMA = "downstream_transfer_v1"
SEEDS = {"development": 20261030, "frozen_gate": 20261031, "confirmatory": 20261032}
COUNTS = {"development": 24, "frozen_gate": 24, "confirmatory": 96}
QUERIES = ("current_x", "current_z")
VALUES = ("amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac","maple","navy","pearl","rust")
CODES = ("K7","M2","R5","T9","B4","C8","D3","F6","G1","H9","J2","L5","N8","P3","Q6","V4")

def template_hash():
    files = (Path(__file__), Path(__file__).with_name("supersession_behavior.py"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()

def signature(row):
    mv = row.get("matching_values") or row.get("semantic_values") or {}
    sig = (mv.get("initial_x", row.get("old_x")),
           mv.get("initial_z", row.get("old_z")),
           mv.get("current_x", mv.get("proposed_x", row.get("current_x"))),
           mv.get("current_z", mv.get("proposed_z", row.get("current_z"))))
    if any(value is None for value in sig):
        raise ValueError("prior dataset lacks a recognized two-entity history signature")
    return sig

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
              f"Which code corresponds to {entity}'s current badge?",
              "Respond with only the code, with no explanation."]
    from src.data.supersession_behavior import _answer_prefix
    return _answer_prefix("\n".join(lines), tokenizer, chat)

def generate(stage,n,values=VALUES,codes=CODES,seed=None,excluded=(),show_progress=False):
    if stage not in COUNTS or n!=COUNTS[stage]: raise ValueError("invalid stage or frozen history count")
    seed=SEEDS[stage] if seed is None else seed
    if seed!=SEEDS[stage]: raise ValueError("stage seeds are frozen")
    if any(r.get("seed")==seed for r in excluded): raise ValueError("stage seed overlaps an excluded dataset")
    if len(values)<8 or len(values)!=len(codes) or len(set(values))!=len(values) or len(set(codes))!=len(codes): raise ValueError("value/code vocabularies must be equal-length and unique (at least eight)")
    rng=random.Random(seed); seen=set(); rows=[]
    for record in excluded:
        sig = signature(record)
        seen.update((sig, (sig[1], sig[0], sig[3], sig[2])))
    logger.info("Generating %s: %d histories, seed=%d, %d excluded signatures", stage, n, seed, len(seen))
    indices = progress(range(n), desc=f"Generating {stage}", unit="history") if show_progress else range(n)
    for i in indices:
        while True:
            a,b,c,d=rng.sample(list(values),4)
            perm=list(codes); rng.shuffle(perm); cb=dict(zip(values,perm))
            probe={"matching_values":{"initial_x":a,"initial_z":b,"current_x":c,"current_z":d},"codebook":cb}
            sig=signature(probe)
            if sig not in seen:
                seen.update((sig, (sig[1], sig[0], sig[3], sig[2])))
                break
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
                          "replacement_value":replacement,"answer_code":pair_answer,
                          "correct_value":replacement if direction and binding==q else mv[q],
                          "roles":{"target":pair_answer},"example_id":f"{pid}:{direction}"})
    audit(rows)
    logger.info("Validated %d %s records", len(rows), stage)
    return rows

def audit(rows):
    if not rows or rows[0].get("stage") not in COUNTS:
        raise ValueError("empty dataset or invalid stage")
    stage = rows[0]["stage"]
    bindings = ("old_x", "old_z", "current_x", "current_z")
    expected = set(QUERIES) if stage != "confirmatory" else {
        (b, q, d) for b in bindings for q in QUERIES for d in (0, 1)
    }
    values, codes = rows[0]["value_vocabulary"], rows[0]["code_vocabulary"]
    if (len(values) < 8 or len(values) != len(codes) or
            len(set(values)) != len(values) or len(set(codes)) != len(codes)):
        raise ValueError("invalid fixed value/code vocabularies")
    histories = defaultdict(dict)
    ids, seen = set(), set()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("stage") != stage or row.get("seed") != SEEDS[stage]:
            raise ValueError("mixed protocol/stage or unfrozen seed")
        if row["value_vocabulary"] != values or row["code_vocabulary"] != codes:
            raise ValueError("vocabularies change within dataset")
        if set(row["codebook"]) != set(values) or set(row["codebook"].values()) != set(codes):
            raise ValueError("codebook is not a bijection of the frozen vocabularies")
        if row["query_id"] not in QUERIES:
            raise ValueError("only current queries are allowed")
        if row["example_id"] in ids:
            raise ValueError("duplicate example ID")
        ids.add(row["example_id"])
        if stage != "confirmatory" and any(k in row for k in (
                "pair_id", "pair_direction", "edited_binding", "replacement_value",
                "source_value", "candidate_logits", "candidate_logprobs", "derived_transfer")):
            raise ValueError("causal metadata leaked into competence stage")
        key = (row["edited_binding"], row["query_id"], row["pair_direction"]) if stage == "confirmatory" else row["query_id"]
        if key in histories[row["history_id"]]:
            raise ValueError("duplicate cell")
        histories[row["history_id"]][key] = row
    if len(histories) != COUNTS[stage]:
        raise ValueError("wrong history count")
    for hid, cells in histories.items():
        if set(cells) != expected:
            raise ValueError(f"incomplete cells: {hid}")
        reference = next(iter(cells.values()))
        mv = reference["matching_values"]
        if (set(mv) != {"initial_x", "initial_z", "current_x", "current_z"} or
                len(set(mv.values())) != 4 or not set(mv.values()) <= set(values)):
            raise ValueError("invalid concrete history bindings")
        sig = signature(reference)
        if sig in seen:
            raise ValueError("duplicate concrete history, including x/z reversal")
        seen.update((sig, (sig[1], sig[0], sig[3], sig[2])))
        index = reference["history_index"]
        if not isinstance(index, int) or not 0 <= index < COUNTS[stage]:
            raise ValueError("invalid history index")
        if hid != f"downstream_transfer_{stage}_{SEEDS[stage]}_{index:06d}":
            raise ValueError("invalid history identifier")
        replacements = {}
        for row in cells.values():
            for key in ("matching_values", "codebook", "variables", "orientation", "history_index", "attribute"):
                if row[key] != reference[key]:
                    raise ValueError(f"history context changed: {key}")
            if row["orientation"] != index % 2 or row["variables"] != (
                    ["Nora", "Owen"] if index % 2 == 0 else ["Owen", "Nora"]):
                raise ValueError("entity orientation mismatch")
            correct = mv[row["query_id"]]
            expected_id = hid + ":" + row["query_id"]
            if stage == "confirmatory":
                binding = row["edited_binding"]
                field = "initial_" + binding[-1] if binding.startswith("old_") else binding
                if row["source_value"] != mv[field] or row["replacement_value"] not in values or row["replacement_value"] in mv.values():
                    raise ValueError("invalid binding source/replacement")
                replacement = row["replacement_value"]
                if replacements.setdefault(binding, replacement) != replacement:
                    raise ValueError("replacement changes across the two queries")
                pid = f"{hid}:{binding}:{row['query_id']}"
                expected_id = f"{pid}:{row['pair_direction']}"
                if row["pair_id"] != pid:
                    raise ValueError("matched pair identifier mismatch")
                if row["pair_direction"] == 1 and binding == row["query_id"]:
                    correct = replacement
            if row["example_id"] != expected_id:
                raise ValueError("example identifier mismatch")
            if row["answer_code"] != row["codebook"][correct] or row["correct_value"] != correct or row["roles"] != {"target": row["answer_code"]}:
                raise ValueError("current answer/edit orientation mismatch")
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
