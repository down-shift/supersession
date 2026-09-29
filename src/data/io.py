import json, hashlib
from pathlib import Path

def write_jsonl(rows,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf8") as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+"\n")
def read_jsonl(path):
    with Path(path).open(encoding="utf8") as f: return [json.loads(x) for x in f if x.strip()]
def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
