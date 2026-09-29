#!/usr/bin/env python3
"""Score independent calibration queries by full-vocabulary next-token accuracy."""
import argparse,json,collections
import numpy as np
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.experiments.behavior import score_example
from src.models.loader import load_model
from src.utils import load_config,provenance,save_json

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/four_query_pilot.yaml"); p.add_argument("--dataset",required=True); p.add_argument("--token-ids",required=True); p.add_argument("--output",required=True); a=p.parse_args()
c=load_config(a.config); model,tok=load_model(c); ids=json.load(open(a.token_ids))["token_ids"]
rows=read_jsonl(a.dataset); scored=[score_example(model,tok,r,ids,chat=c["model"].get("chat_template",True)) for r in tqdm(rows,desc="Calibration scoring")]
by_query=collections.defaultdict(list)
for r in scored: by_query[r["query_id"]].append(r)
summary={q:{"n":len(v),"candidate_accuracy":float(np.mean([x["accuracy"] for x in v])),"full_vocab_next_token_accuracy":float(np.mean([x["full_vocab_next_token_accuracy"] for x in v]))} for q,v in sorted(by_query.items())}
threshold=float(c["thresholds"].get("direct_accuracy",.99)); passed=len(by_query)==4 and all(x["full_vocab_next_token_accuracy"]>=threshold for x in summary.values())
result={"query_results":summary,"required_full_vocab_next_token_accuracy":threshold,"pass":passed,"data_role":"independent calibration only; exclude from confirmatory analyses","provenance":provenance(c,a.dataset)}
save_json(result,a.output); print(json.dumps(result,indent=2));
if not passed: raise SystemExit("calibration failed the configured full-vocabulary accuracy gate; choose/fix the prompt or model before generating confirmatory results")
