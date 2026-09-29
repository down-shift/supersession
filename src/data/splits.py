"""Group-level splits to prevent value variants crossing data partitions."""
import random

def grouped_split(rows, fractions=(.7,.15,.15), seed=0):
    if len(fractions)!=3 or abs(sum(fractions)-1)>1e-8: raise ValueError("fractions must sum to one")
    groups={}
    for row in rows: groups.setdefault(tuple(row["split_group"]),[]).append(row)
    keys=list(groups); random.Random(seed).shuffle(keys); n=len(keys)
    a=round(n*fractions[0]); b=round(n*(fractions[0]+fractions[1]))
    out=[]
    for split,subset in zip(("train","validation","test"),(keys[:a],keys[a:b],keys[b:])):
        for k in subset:
            for row in groups[k]: out.append({**row,"split":split})
    return out
