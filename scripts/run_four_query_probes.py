#!/usr/bin/env python3
"""Grouped five-fold probes for four binding slots under each query."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss
from src.data.io import read_jsonl
from src.utils import save_json

p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--activations",required=True); p.add_argument("--token-ids",required=True); p.add_argument("--output",required=True); p.add_argument("--folds",type=int,default=5); p.add_argument("--seed",type=int,default=20260929); a=p.parse_args()
data=read_jsonl(a.dataset); chunks=sorted(Path(a.activations).glob("chunk_*.npz")); X=np.concatenate([np.load(f)["activations"] for f in chunks]); meta=json.load(open(Path(a.activations)/"metadata.json"))["examples"]
if len(X)!=len(data) or len(meta)!=len(data): raise ValueError("activation, metadata, and dataset row counts differ")
if any(not r.get("history_id") or not r.get("query_id") for r in data): raise ValueError("dataset must contain history_id and query_id")
tokens=json.load(open(a.token_ids))["token_ids"]; results={}; slots=("old_x","current_x","old_z","current_z"); queries=("current_x","initial_x","current_z","initial_z")
for query in queries:
    ix=np.asarray([i for i,r in enumerate(data) if r["query_id"]==query]); groups=np.asarray([data[i]["history_id"] for i in ix]);
    for slot in slots:
        labels=np.asarray([tokens[data[i]["roles"][slot]] for i in ix]); classes=np.unique(labels)
        if len(ix)<a.folds or len(classes)<2: raise ValueError(f"insufficient data for {query}/{slot}")
        key=f"{query}/{slot}"; layer_rows=[]
        for layer in range(X.shape[1]):
            truth=np.full(len(ix),-1,dtype=int); prob=np.zeros((len(ix),len(classes)),dtype=float); fold_rows=[]
            for fold,(train,test) in enumerate(GroupKFold(n_splits=a.folds).split(ix,labels,groups)):
                if set(labels[train])!=set(classes): raise ValueError(f"training fold misses a class for {key}")
                scaler=StandardScaler().fit(X[ix[train],layer]); xtrain=scaler.transform(X[ix[train],layer]); xtest=scaler.transform(X[ix[test],layer])
                model=LogisticRegression(C=1.,max_iter=1000,random_state=a.seed).fit(xtrain,labels[train]); pred=model.predict(xtest); ptest=model.predict_proba(xtest)
                truth[test]=pred; class_ix={v:j for j,v in enumerate(model.classes_)}
                for j,v in enumerate(classes):
                    if v in class_ix: prob[test,j]=ptest[:,class_ix[v]]
                fold_rows.append({"fold":fold,"n_histories":int(len(test)),"accuracy":float(accuracy_score(labels[test],pred)),"balanced_accuracy":float(balanced_accuracy_score(labels[test],pred)),"log_loss":float(log_loss(labels[test],ptest,labels=model.classes_))})
            # History-level nonparametric interval on out-of-fold accuracy.
            correct=(truth==labels).astype(float); rng=np.random.default_rng(a.seed+layer); boot=np.mean(rng.choice(correct,(2000,len(correct)),replace=True),axis=1)
            layer_rows.append({"layer":layer,"folds":fold_rows,"oof_accuracy":float(correct.mean()),"oof_balanced_accuracy":float(balanced_accuracy_score(labels,truth)),"oof_log_loss":float(log_loss(labels,prob,labels=classes)),"history_bootstrap_accuracy_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]})
        results[key]=layer_rows
save_json({"results":results,"slots":list(slots),"queries":list(queries),"folds":a.folds,"grouping":"history_id; all query/counterfactual descendants remain in one fold","interpretation":"descriptive decoding performance, not evidence of behavioral use"},a.output)
print(f"saved grouped four-query probe results to {a.output}")
