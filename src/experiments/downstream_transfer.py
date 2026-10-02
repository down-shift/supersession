"""Shared artifact and candidate checks for the isolated downstream experiment."""
import hashlib
import json
import re
from pathlib import Path

from src.cross_model.scoring import score_prompt
from src.data.downstream_transfer import (
    SCHEMA, audit, signature, template_hash, verify_sealed_artifact,
)
from src.data.io import read_jsonl, sha256_file

SCORING_VERSION = "complete_sequence_logprob_v1"
ROOT = Path(__file__).resolve().parents[2]


def scoring_hash():
    paths = (
        "scripts/run_downstream_transfer.py", "src/analysis/downstream_transfer.py",
        "src/experiments/downstream_transfer.py", "src/cross_model/scoring.py",
        "src/cross_model/tokens.py", "src/data/progress.py", "src/models/loader.py",
    )
    return hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in paths)
                          + SCORING_VERSION.encode()).hexdigest()


def frozen_metadata(config, config_path, values_path, codes_path):
    model = config["model"]
    for key in ("revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(model.get(key))):
            raise ValueError(f"frozen experiment requires immutable {key}")
    if config.get("thresholds", {}).get("gate_accuracy") != .97:
        raise ValueError("the frozen competence threshold is 0.97")
    return {
        "config_sha256": sha256_file(config_path),
        "values_sha256": sha256_file(values_path),
        "codes_sha256": sha256_file(codes_path),
        "model_revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "template_sha256": template_hash(),
        "scoring_version": SCORING_VERSION,
        "scoring_code_sha256": scoring_hash(),
    }


def write_json_create(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")


def resolve_artifact_path(recorded_path, current_artifact):
    """Resolve a saved absolute path, then a same-directory relocated artifact."""
    recorded = Path(recorded_path)
    if recorded.exists():
        return recorded
    local = Path(current_artifact).resolve().parent / recorded.name
    if local.exists():
        return local
    raise FileNotFoundError(f"saved artifact is unavailable: {recorded_path} (also checked {local})")


def candidate_events(tokenizer, prompt, codes):
    """One fixed spaced continuation per code, with complete token sequences."""
    prefix = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    events = {}
    sequences = []
    for code in codes:
        text = " " + code
        full = list(tokenizer(prompt + text, add_special_tokens=False)["input_ids"])
        if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(f"code {code!r} is not a stable continuation")
        ids = full[len(prefix):]
        for other in sequences:
            if ids[:len(other)] == other or other[:len(ids)] == ids:
                raise ValueError("code candidate token sequences collide or overlap")
        sequences.append(ids)
        events[code] = [{"text": text, "ids": ids}]
    return events


def score_codes(model, tokenizer, prompt, codes, token_audit, progress_callback=None):
    events = candidate_events(tokenizer, prompt, codes)
    if {code: members[0]["ids"] for code, members in events.items()} != {
        code: entry["token_ids"] for code, entry in token_audit["codes"].items()
    }:
        raise ValueError("actual candidate sequences differ from the tokenizer audit")
    # Reuse the existing sequence scorer, including embedding-device selection
    # and nonfinite-logit rejection. No extra surface variants are introduced.
    probabilities, _, _, _ = score_prompt(
        model, tokenizer, prompt, events, progress_callback=progress_callback,
    )
    return probabilities


def generate_unrestricted_code(model, tokenizer, prompt, codes, token_audit, progress_callback=None):
    """Greedily generate a short answer and accept only an exact code label."""
    import torch
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prefix_length = encoded["input_ids"].shape[1]
    max_new_tokens = max(4, max(entry["n_tokens"] for entry in token_audit["codes"].values()) + 2)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                   do_sample=False, num_beams=1)
    if progress_callback:
        progress_callback()
    continuation = generated[0, prefix_length:]
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    parsed = text.strip()
    code = parsed if parsed in codes else None
    return {"unrestricted_generated_text": text,
            "unrestricted_generated_code": code}


def validate_token_audit(path, config, metadata, codes):
    doc = verify_sealed_artifact(json.loads(Path(path).read_text()))
    model = config["model"]
    expected = {
        "protocol": SCHEMA, "audit": "tokenizer_only_no_model_inference",
        "tokenizer_id": model.get("tokenizer_id") or model["id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "chat_template": model.get("chat_template", True),
        "continuation_prefix_stable": True,
    }
    expected.update(metadata)
    if any(doc.get(key) != value for key, value in expected.items()):
        raise ValueError("tokenizer audit differs from frozen tokenizer/config/vocabularies")
    if set(doc.get("codes", {})) != set(codes) or doc.get("prompts_checked", 0) < 48:
        raise ValueError("tokenizer audit lacks code or prompt coverage")
    sequences = []
    for entry in doc["codes"].values():
        ids = entry.get("token_ids")
        if not ids or entry.get("n_tokens") != len(ids) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("invalid tokenizer audit candidate sequence")
        if any(ids[:len(other)] == other or other[:len(ids)] == ids for other in sequences):
            raise ValueError("tokenizer audit code sequences overlap")
        sequences.append(ids)
    return doc


def validate_dataset(path, metadata=None):
    rows = read_jsonl(path)
    audit(rows)
    side = Path(str(path) + ".provenance.json")
    doc = json.loads(side.read_text())
    if (doc.get("protocol") != SCHEMA or doc.get("stage") != rows[0]["stage"] or
            doc.get("dataset_sha256") != sha256_file(path) or
            doc.get("template_sha256") != template_hash()):
        raise ValueError("dataset provenance/hash mismatch")
    if metadata:
        for key in ("config_sha256", "values_sha256", "codes_sha256",
                    "model_revision", "tokenizer_revision"):
            if doc.get(key) != metadata[key]:
                raise ValueError(f"dataset frozen artifact mismatch: {key}")
    paths, hashes = doc.get("prior_dataset_paths", []), doc.get("prior_dataset_sha256", [])
    if len(paths) != len(hashes):
        raise ValueError("invalid prior dataset provenance")
    own = {signature(r) for r in rows}
    has_development = False
    for prior, expected in zip(paths, hashes):
        resolved_prior = resolve_artifact_path(prior, path)
        if sha256_file(resolved_prior) != expected:
            raise ValueError("supplied prior dataset changed")
        for record in read_jsonl(resolved_prior):
            has_development |= record.get("schema") == SCHEMA and record.get("stage") == "development"
            sig = signature(record)
            if sig in own or (sig[1], sig[0], sig[3], sig[2]) in own:
                raise ValueError("concrete history overlaps a supplied prior dataset")
    if rows[0]["stage"] in ("frozen_gate", "confirmatory") and not has_development:
        raise ValueError("frozen stages lack development dataset exclusion provenance")
    return rows, doc


def validate_gate(path, metadata, token_audit_path):
    gate = verify_sealed_artifact(json.loads(Path(path).read_text()))
    for key, value in metadata.items():
        if gate.get(key) != value:
            raise ValueError(f"frozen {key} changed")
    if (gate.get("protocol") != SCHEMA or gate.get("stage") != "frozen_gate" or
            gate.get("pass") is not True or gate.get("threshold") != .97 or
            gate.get("n") != 48 or not .97 <= gate.get("current_derived_code_accuracy", -1) <= 1):
        raise ValueError("invalid/failed frozen gate")
    for stem in ("dataset", "scores", "token_audit"):
        if gate.get(stem + "_sha256") != sha256_file(gate[stem + "_path"]):
            raise ValueError(f"frozen gate {stem} changed")
    if gate["token_audit_sha256"] != sha256_file(token_audit_path):
        raise ValueError("tokenizer audit differs from frozen gate")
    for stem in ("dataset", "scores"):
        if gate.get(stem + "_provenance_sha256") != sha256_file(
                gate[stem + "_path"] + ".provenance.json"):
            raise ValueError(f"frozen gate {stem} provenance changed")
    gate_rows, gate_prov = validate_dataset(gate["dataset_path"], metadata)
    if gate_rows[0]["stage"] != "frozen_gate":
        raise ValueError("gate artifact references the wrong dataset stage")
    from src.analysis.downstream_transfer import evaluate_competence_gate
    result = evaluate_competence_gate(gate_rows, read_jsonl(gate["scores_path"]))
    for key, value in result.items():
        if gate.get(key) != value:
            raise ValueError("sealed gate decision differs from its competence scores")
    score_prov = json.loads(Path(gate["scores_path"] + ".provenance.json").read_text())
    if (score_prov.get("protocol") != SCHEMA or score_prov.get("stage") != "frozen_gate" or
            score_prov.get("causal_effects_computed") is not False or
            score_prov.get("dataset_sha256") != gate["dataset_sha256"] or
            score_prov.get("token_audit_sha256") != gate["token_audit_sha256"]):
        raise ValueError("invalid gate score provenance")
    for key, value in metadata.items():
        if score_prov.get(key) != value:
            raise ValueError(f"frozen gate score provenance differs: {key}")
    return gate, gate_prov
