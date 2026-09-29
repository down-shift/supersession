#!/usr/bin/env python3
"""Write the fixed stage-1 mechanistic partitions from existing artifacts."""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

def ids_in(path, column='history_id'):
    f=pd.read_json(path,lines=True) if str(path).endswith('.jsonl') else pd.read_csv(path)
    return list(dict.fromkeys(map(str,f[column].dropna())))

def partition_sets(confirm, discovery, heldout):
    confirm=set(map(str,confirm)); discovery=set(map(str,discovery)); heldout=set(map(str,heldout))
    reserve=confirm-discovery-heldout
    if discovery&heldout or discovery&reserve or heldout&reserve:
        raise ValueError('mechanistic partition sets are not mutually disjoint')
    if discovery|heldout|reserve != confirm:
        raise ValueError('mechanistic partition union differs from confirmatory histories')
    return discovery,heldout,reserve

def main():
    p=argparse.ArgumentParser(); p.add_argument('--pairs',required=True); p.add_argument('--discovery',required=True); p.add_argument('--heldout',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    pairs=pd.read_json(a.pairs,lines=True); confirm=sorted(set(pairs.loc[pairs.get('partition','confirmatory').eq('confirmatory'),'history_id'].astype(str)))
    disc=ids_in(a.discovery); held=ids_in(a.heldout)
    ds,hs,rs=partition_sets(confirm,disc,held)
    if not ds|hs <= set(confirm): raise ValueError('stage-1 IDs are not all confirmatory histories')
    reserve=sorted(rs)
    doc={'source_artifacts':{'confirmatory_pairs':a.pairs,'stage1_discovery':a.discovery,'stage1_heldout':a.heldout},'counts':{'confirmatory':len(confirm),'stage1_discovery':len(disc),'stage1_heldout':len(held),'unused_mechanistic_reserve':len(reserve)},'sha256':{k:hashlib.sha256('\n'.join(v).encode()).hexdigest() for k,v in [('confirmatory',confirm),('stage1_discovery',disc),('stage1_heldout',held),('unused_mechanistic_reserve',reserve)]},'history_ids':{'confirmatory':confirm,'stage1_discovery':disc,'stage1_heldout':held,'unused_mechanistic_reserve':reserve}}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__': main()
