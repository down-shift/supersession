#!/usr/bin/env python3
"""Score independent calibration queries by full-vocabulary next-token accuracy."""
import argparse,json,collections
import numpy as np
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.data.progress import prepare_jsonl_progress,append_jsonl_record
from src.experiments.behavior import score_example
from src.models.loader import load_model
from src.utils import load_config,provenance,save_json

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/four_query_pilot.yaml"); p.add_argument("--dataset",required=True); p.add_argument("--token-ids",required=True); p.add_argument("--output",required=True); p.add_argument("--records-output",help="per-example checkpoint JSONL (default: OUTPUT.records.jsonl)"); p.add_argument("--resume",action="store_true",help="continue matching calibration scoring from its last complete record"); a=p.parse_args()
c=load_config(a.config); ids=json.load(open(a.token_ids))["token_ids"]
rows=read_jsonl(a.dataset); records_output=a.records_output or a.output+".records.jsonl"
completed=prepare_jsonl_progress(records_output,a.dataset,a.token_ids,c,rows,resume=a.resume)
model,tok=load_model(c); todo=[r for r in rows if str(r["example_id"]) not in completed]
for row in tqdm(todo,desc=f"Calibration scoring ({len(completed)} already saved)"):
    scored=score_example(model,tok,row,ids,chat=c["model"].get("chat_template",True))
    append_jsonl_record(records_output,scored)
scored=read_jsonl(records_output)
by_query=collections.defaultdict(list)
for r in scored: by_query[r["query_id"]].append(r)
summary={q:{"n":len(v),"candidate_accuracy":float(np.mean([x["accuracy"] for x in v])),"full_vocab_next_token_accuracy":float(np.mean([x["full_vocab_next_token_accuracy"] for x in v]))} for q,v in sorted(by_query.items())}
threshold=float(c["thresholds"].get("direct_accuracy",.99)); passed=len(by_query)==4 and all(x["full_vocab_next_token_accuracy"]>=threshold for x in summary.values())
result={"query_results":summary,"required_full_vocab_next_token_accuracy":threshold,"pass":passed,"data_role":"independent calibration only; exclude from confirmatory analyses","provenance":provenance(c,a.dataset)}
save_json(result,a.output); print(json.dumps(result,indent=2));
if not passed: raise SystemExit("calibration failed the configured full-vocabulary accuracy gate; choose/fix the prompt or model before generating confirmatory results")
