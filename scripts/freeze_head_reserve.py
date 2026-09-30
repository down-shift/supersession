#!/usr/bin/env python3
"""Freeze deterministic, exhaustive subpartitions of the unused head reserve."""
import argparse, hashlib, json, random
from pathlib import Path

def split_reserve(ids, seed=20260930):
    ids=sorted(map(str,ids))
    if len(ids)!=len(set(ids)): raise ValueError('unused reserve IDs must be unique')
    rng=random.Random(seed); shuffled=ids[:]; rng.shuffle(shuffled)
    n=len(ids); weights=(72,48,48); total=sum(weights)
    sizes=[n*w//total for w in weights]
    for i in range(n-sum(sizes)): sizes[i%3]+=1
    a,b,c=sizes
    result={'head_confirmation':sorted(shuffled[:a]),'path_confirmation':sorted(shuffled[a:a+b]),'final_validation':sorted(shuffled[a+b:])}
    sets=[set(v) for v in result.values()]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)) or set.union(*sets)!=set(ids): raise AssertionError('invalid reserve split')
    return result

def build_document(parent, seed=20260930):
    reserve=parent['history_ids']['unused_mechanistic_reserve']; subsets=split_reserve(reserve,seed)
    return {'schema':'head_reserve_subpartition_v1','seed':seed,'source_partition_sha256':hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest(),
      'counts':{k:len(v) for k,v in subsets.items()},'history_ids':subsets,
      'sha256':{k:hashlib.sha256('\n'.join(v).encode()).hexdigest() for k,v in subsets.items()},
      'exact_cover_of':'unused_mechanistic_reserve','immutable':True}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--partition-file',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=20260930); a=p.parse_args()
    out=Path(a.output)
    if out.exists(): raise FileExistsError(f'{out} already exists; frozen subpartitions are never overwritten')
    doc=build_document(json.loads(Path(a.partition_file).read_text()),a.seed); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__': main()
