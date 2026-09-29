#!/usr/bin/env python3
"""Fit matched value-identity linear probes from streamed activation chunks."""
import argparse,json
import numpy as np
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.experiments.probes import fit_probe_curves
from src.utils import save_json
p=argparse.ArgumentParser(); p.add_argument("--dataset",default="outputs/primary/dataset.jsonl"); p.add_argument("--activations",default="outputs/primary/activations"); p.add_argument("--token-ids",default="outputs/primary/token_ids.json"); p.add_argument("--output",default="outputs/primary/probes.json"); p.add_argument("--seed",type=int,default=20260929); p.add_argument("--n-seeds",type=int,default=3); a=p.parse_args()
data=read_jsonl(a.dataset); token_ids=json.load(open(a.token_ids))["token_ids"]; chunks=sorted(__import__("pathlib").Path(a.activations).glob("chunk_*.npz")); arrays=[np.load(f)["activations"] for f in chunks]; X=np.concatenate(arrays); meta=json.load(open(a.activations+"/metadata.json"))["examples"]
if len(X)!=len(data) or len(meta)!=len(data): raise ValueError("activation, metadata, and dataset row counts differ")
splits=np.array([r["split"] for r in data]); out={}
for role in tqdm(("C_q","O_q","C_d","O_d"),desc="Probe roles"):
 labels=np.array([token_ids[r["roles"][role]] for r in data]); train_classes=set(labels[splits=="train"]); unseen=set(labels)-train_classes
 if unseen: raise ValueError(f"probe role {role} has value classes absent from probe train data: {sorted(unseen)}")
 out[role]=[row for seed in range(a.seed,a.seed+a.n_seeds) for row in fit_probe_curves(X,labels,splits,seed)]
save_json({"decoding_performance_not_mutual_information":out,"seeds":list(range(a.seed,a.seed+a.n_seeds)),"roles":["C_q","O_q","C_d","O_d"],"provenance":json.load(open(a.activations+"/metadata.json"))["provenance"]},a.output); print(f"saved probe curves to {a.output}")
