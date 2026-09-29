#!/usr/bin/env python3
"""Summarize paired prompt-development scores and freeze the pre-specified choice."""
import argparse,hashlib,json
from pathlib import Path
from src.data.io import read_jsonl
from src.analysis.prompt_development import summarize_variant,summarize_errors,select_prompt
from src.utils import load_config
p=argparse.ArgumentParser(); p.add_argument("--records",action="append",required=True,help="VARIANT=per-example.jsonl; repeat for variants"); p.add_argument("--token-ids",required=True); p.add_argument("--config",required=True); p.add_argument("--provenance",help="scoring summary JSON containing resolved model/tokenizer revisions"); p.add_argument("--output",required=True); p.add_argument("--criterion",type=float); a=p.parse_args()
config=load_config(a.config); criterion=a.criterion if a.criterion is not None else float(config["thresholds"].get("prompt_dev_min_accuracy",.98)); summaries={}
for item in a.records:
 variant,path=item.split("=",1); rows=read_jsonl(path)
 summary=summarize_variant(rows,criterion); summary["error_taxonomy"]=summarize_errors(rows,json.load(open(a.token_ids))["token_ids"].keys()); summaries[variant]=summary
choice=select_prompt(summaries)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
tok_meta=json.load(open(a.token_ids))
score_provenance=json.load(open(a.provenance)).get("provenance",{}) if a.provenance else {}
artifact={**choice,"variants":summaries,"development_criterion":criterion,"dataset_hashes":{k:sha(v) for k,v in (x.split("=",1) for x in a.records)},"config_hash":sha(a.config),"model_id":config["model"]["id"],"model_revision":score_provenance.get("model_revision") or config["model"].get("revision"),"tokenizer_id":config["model"].get("tokenizer_id") or config["model"]["id"],"tokenizer_revision":score_provenance.get("tokenizer_revision") or tok_meta.get("tokenizer_revision"),"token_ids":tok_meta["token_ids"]}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(artifact,indent=2)+"\n")
print(json.dumps(artifact,indent=2))
if not choice["confirmatory_permitted"]: raise SystemExit("no admissible prompt variant met development criterion")
