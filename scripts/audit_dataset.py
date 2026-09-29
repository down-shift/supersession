#!/usr/bin/env python3
import argparse,collections
from src.data.io import read_jsonl
from src.data.generate import legal_orders
p=argparse.ArgumentParser(); p.add_argument("dataset",nargs="?",default="outputs/pilot/dataset.jsonl"); a=p.parse_args(); rows=read_jsonl(a.dataset)
counts=collections.Counter(tuple(r["order"]) for r in rows); pos=collections.Counter()
for r in rows:
 for role,key in (("O_x","old_x"),("O_z","old_z"),("C_x","current_x"),("C_z","current_z")): pos[(role,r["order"].index(role))]+=1
position_table={role:[pos[(role,index)] for index in range(4)] for role in ("O_x","O_z","C_x","C_z")}; order_values=list(counts.values()); balanced=max(order_values)-min(order_values)<=1
print({"n":len(rows),"legal_orders":len(counts),"order_counts":dict(counts),"order_counts_balanced_within_one":balanced,"role_position_counts":dict(pos),"role_position_table":position_table})
if set(counts)!=set(legal_orders()): raise SystemExit("incomplete legal-order coverage")
if not balanced: raise SystemExit("legal order counts are not balanced within one")
