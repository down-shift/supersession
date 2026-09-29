#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer
from src.utils import load_config,save_json,provenance
from src.data.generate import make_contexts
from src.data.token_validation import validate_candidate_vocabulary
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--family",choices=["symbolic","natural"],default="symbolic"); p.add_argument("--output",default="outputs/pilot/token_ids.json"); a=p.parse_args()
c=load_config(a.config); m=c["model"]; ident=m.get("tokenizer_id") or m["id"]; rev=m.get("tokenizer_revision") or m.get("revision"); tok=AutoTokenizer.from_pretrained(ident,revision=rev); d=c["dataset"]; examples=make_contexts(min(d["n_contexts"],max(12,len(d["variables"])*12)),c["seed"],d["values"],d["variables"],a.family); ids,rejected=validate_candidate_vocabulary(tok,examples,d["values"],chat=m.get("chat_template",True)); save_json({"tokenizer_id":ident,"tokenizer_revision":getattr(tok,"_commit_hash",None) or getattr(tok,"init_kwargs",{}).get("_commit_hash",rev),"chat_template":m.get("chat_template"),"family":a.family,"token_ids":ids,"rejected":rejected,"provenance":provenance(c)},a.output); print(f"validated {len(ids)} candidates; rejected {len(rejected)}; wrote {a.output}")
