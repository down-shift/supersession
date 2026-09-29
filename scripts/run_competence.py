#!/usr/bin/env python3
import argparse,json
from src.utils import load_config,save_json,provenance
from src.data.io import read_jsonl,write_jsonl
from src.experiments.behavior import score_example,competence_gate
from src.models.loader import load_model
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--direct",default="outputs/pilot/direct.jsonl"); p.add_argument("--overwrite",default="outputs/pilot/dataset.jsonl"); p.add_argument("--token-ids",default="outputs/pilot/token_ids.json"); p.add_argument("--output",default="outputs/pilot/competence.json"); a=p.parse_args(); c=load_config(a.config); model,tok=load_model(c); ids=json.load(open(a.token_ids))["token_ids"]
direct=[score_example(model,tok,x,ids,chat=c["model"].get("chat_template",True)) for x in read_jsonl(a.direct)]; overwrite=[score_example(model,tok,x,ids,chat=c["model"].get("chat_template",True)) for x in read_jsonl(a.overwrite)]; summary=competence_gate(direct,overwrite,c["thresholds"]); write_jsonl(direct,a.output+".direct.jsonl"); write_jsonl(overwrite,a.output+".overwrite.jsonl"); run=provenance(c,a.overwrite); run["candidate_token_ids"]=ids; save_json({"summary":summary,"direct_n":len(direct),"overwrite_n":len(overwrite),"provenance":run},a.output)
import pandas as pd
pd.DataFrame([{"condition":"direct","n":len(direct),"accuracy":summary["direct_accuracy"],"threshold":c["thresholds"]["direct_accuracy"]},{"condition":"overwrite","n":len(overwrite),"accuracy":summary["overwrite_accuracy"],"threshold":c["thresholds"]["overwrite_accuracy"]}]).to_csv(a.output+".csv",index=False); print(summary)
