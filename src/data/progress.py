"""Small JSONL checkpoint helpers for long scoring jobs."""
import hashlib
import json
from pathlib import Path


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_jsonl_progress(output, dataset, token_ids, config, rows, resume=False, allow_token_ids_rehash=False):
    """Validate/create a run manifest and return IDs already checkpointed."""
    output = Path(output)
    manifest = Path(str(output) + ".run.json")
    expected = {
        "dataset_sha256": _sha256(dataset),
        "token_ids_sha256": _sha256(token_ids),
        "config": config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if not manifest.exists():
            raise ValueError("--resume requires the original .run.json manifest")
        saved = json.loads(manifest.read_text(encoding="utf8"))
        if saved != expected:
            mismatched = [key for key in expected if saved.get(key) != expected[key]]
            if allow_token_ids_rehash and mismatched == ["token_ids_sha256"]:
                # A caller may enable this only after verifying the current
                # token map against a frozen selection artifact. Older run
                # manifests hashed the entire JSON file, including volatile
                # provenance metadata.
                manifest.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n", encoding="utf8")
                saved = expected
                mismatched = []
            if not mismatched:
                pass
            else:
                raise ValueError(
                    "cannot resume: checkpoint fingerprint differs in "
                    + ", ".join(mismatched)
                    + ". Preserve the old records/manifest and start a fresh output if this is a new run."
                )
    else:
        if output.exists() or manifest.exists():
            raise FileExistsError(f"{output} already exists; choose a new output path or pass --resume")
        manifest.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n", encoding="utf8")

    expected_ids = [str(r["example_id"]) for r in rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("input dataset contains duplicate example_id values")
    expected_id_set = set(expected_ids)
    completed = {}
    if resume and output.exists():
        # Ignore and remove a potentially truncated final line from an interrupted write.
        raw = output.read_bytes()
        last_newline = raw.rfind(b"\n")
        if raw and last_newline != len(raw) - 1:
            raw = raw[:last_newline + 1] if last_newline >= 0 else b""
            output.write_bytes(raw)
        for line in raw.splitlines():
            record = json.loads(line)
            example_id = str(record["example_id"])
            if example_id not in expected_id_set:
                raise ValueError(f"checkpoint contains unexpected example_id: {example_id}")
            if example_id in completed:
                raise ValueError(f"checkpoint contains duplicate example_id: {example_id}")
            completed[example_id] = True
    return set(completed)


def append_jsonl_record(output, record):
    with Path(output).open("a", encoding="utf8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
