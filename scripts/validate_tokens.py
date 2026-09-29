#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer
from src.utils import load_config,save_json,provenance
from src.data.generate import make_contexts,make_histories,expand_history_queries,PROMPT_VARIANTS
from src.data.token_validation import validate_candidate_vocabulary,validate_assignment_patching
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--family",choices=["symbolic","natural"],default="symbolic"); p.add_argument("--design",choices=["legacy","four-query"],default="legacy"); p.add_argument("--output",default="outputs/pilot/token_ids.json"); a=p.parse_args()
c=load_config(a.config); m=c["model"]; ident=m.get("tokenizer_id") or m["id"]; rev=m.get("tokenizer_revision") or m.get("revision"); tok=AutoTokenizer.from_pretrained(ident,revision=rev); d=c["dataset"]
if a.design=="four-query":
 if len(d["values"])<16: raise ValueError("four-query validation needs a raw proposal pool of at least 16 values")
 # History values here provide representative prompt structure only. Every
 # candidate is tested as a continuation, then selected values are used below
 # to regenerate the actual assignment histories before patch-alignment audit.
 histories=make_histories(min(d["n_histories"],max(48,len(d["variables"])*12)),c["seed"],d["values"][:16],d["variables"])
 examples=[q for h in histories for q in expand_history_queries(h)]
 assignment_audit=None
else:
 histories=[]; assignment_audit=None
 examples=make_contexts(min(d["n_contexts"],max(12,len(d["variables"])*12)),c["seed"],d["values"],d["variables"],a.family)
ids,rejected=validate_candidate_vocabulary(tok,examples,d["values"],chat=m.get("chat_template",True),prompt_variants=PROMPT_VARIANTS if a.design=="four-query" else None)
if a.design=="four-query":
 count=int(d.get("candidate_count",12))
 if count not in (12,16): raise ValueError("dataset.candidate_count must be 12 or 16")
 selected=[value for value in d["values"] if value in ids][:count]
 if len(selected)!=count: raise ValueError(f"candidate_count={count}, but only {len(selected)} validated candidates are available")
 ids={value:ids[value] for value in selected}
 histories=make_histories(min(d["n_histories"],max(48,len(d["variables"])*12)),c["seed"],selected,d["variables"])
 assignment_audit=validate_assignment_patching(tok,histories,selected,chat=m.get("chat_template",True),prompt_variants=PROMPT_VARIANTS)
save_json({"tokenizer_id":ident,"tokenizer_revision":getattr(tok,"_commit_hash",None) or getattr(tok,"init_kwargs",{}).get("_commit_hash",rev),"chat_template":m.get("chat_template"),"family":a.family,"design":a.design,"token_ids":ids,"rejected":rejected,"assignment_patch_audit":assignment_audit,"provenance":provenance(c)},a.output); print(f"validated {len(ids)} candidates; rejected {len(rejected)}; wrote {a.output}")
