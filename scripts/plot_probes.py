#!/usr/bin/env python3
import argparse,json
import matplotlib.pyplot as plt
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("probes",default="outputs/primary/probes.json"); p.add_argument("--output-dir",default="outputs/primary/analysis"); a=p.parse_args(); d=json.load(open(a.probes))["decoding_performance_not_mutual_information"]; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(7,4)); means={}
chance_values=[]
for role,rows in d.items():
 by={}
 for x in rows: by.setdefault(x["layer"],[]).append(x["test_accuracy"])
 layers=sorted(by); vals=[sum(by[i])/len(by[i]) for i in layers]; sd=[(sum((v-vals[j])**2 for v in by[i])/len(by[i]))**.5 for j,i in enumerate(layers)]; means[role]=dict(zip(layers,vals)); ax.plot(layers,vals,label=role); ax.fill_between(layers,[v-s for v,s in zip(vals,sd)],[v+s for v,s in zip(vals,sd)],alpha=.12); chance_values.extend(x["chance"] for x in rows)
ax.axhline(sum(chance_values)/len(chance_values),color="black",linestyle="--",label="chance")
ax.set(xlabel="Block output layer",ylabel="Linear decoding accuracy",ylim=(0,1)); ax.legend(); fig.tight_layout(); fig.savefig(out/"four_role_decoding.png",dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for label,a,b in (("Selective current", "C_q","C_d"),("Selective obsolete","O_q","O_d")):
 aa=means[a]; bb=means[b]; layers=sorted(set(aa)&set(bb)); ax.plot(layers,[aa[i]-bb[i] for i in layers],label=label)
ax.axhline(0,color="black",linewidth=.8); ax.set(xlabel="Block output layer",ylabel="Decoding accuracy difference"); ax.legend(); fig.tight_layout(); fig.savefig(out/"selective_decoding.png",dpi=180); plt.close(fig)
