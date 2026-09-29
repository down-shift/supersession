#!/usr/bin/env python3
"""Score independent calibration queries by full-vocabulary next-token accuracy."""
import argparse,json,collections
import numpy as np
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.data.progress import prepare_jsonl_progress,append_jsonl_record
from src.experiments.behavior import score_example
from src.analysis.prompt_development import summarize_errors,gate_allows_confirmatory
from src.models.loader import load_model
from src.utils import load_config,provenance,save_json

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/four_query_288.yaml"); p.add_argument("--dataset",required=True); p.add_argument("--token-ids",required=True); p.add_argument("--output",required=True); p.add_argument("--records-output",help="per-example checkpoint JSONL (default: OUTPUT.records.jsonl)"); p.add_argument("--resume",action="store_true"); p.add_argument("--diagnostic-only",action="store_true",help="score and summarize without applying the held-out gate"); a=p.parse_args()
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
threshold=float(c["thresholds"].get("gate_accuracy",.99)); passed=gate_allows_confirmatory(summary,threshold)
result={"query_results":summary,"error_taxonomy":summarize_errors(scored,ids.keys()),"required_full_vocab_next_token_accuracy":threshold,"pass":None if a.diagnostic_only else passed,"data_role":"prompt-development diagnostic only" if a.diagnostic_only else "held-out competence gate; exclude from prompt tuning and confirmatory analyses","provenance":provenance(c,a.dataset)}
save_json(result,a.output); print(json.dumps(result,indent=2));
if not a.diagnostic_only and not passed: raise SystemExit("held-out gate failed full-vocabulary next-token accuracy; confirmatory generation is prohibited")
