from collections import Counter
from src.data.generate import make_contexts,legal_orders,direct_example
from src.data.splits import grouped_split

def test_precedence_distinct_roles_query_balance_and_order_balance():
    rows=make_contexts(48,seed=3)
    assert len(legal_orders())==6
    assert Counter(tuple(r["order"]) for r in rows)==Counter({o:8 for o in legal_orders()})
    assert Counter(r["query"] for r in rows)=={"x":24,"z":24}
    for r in rows:
        assert r["order"].index("O_x")<r["order"].index("C_x")
        assert r["order"].index("O_z")<r["order"].index("C_z")
        assert len(set(r["roles"].values()))==4
        q=r["query"]; d="z" if q=="x" else "x"
        assert r["roles"]=={"O_q":r[f"old_{q}"],"C_q":r[f"current_{q}"],"O_d":r[f"old_{d}"],"C_d":r[f"current_{d}"]}

def test_group_split_does_not_leak():
    rows=grouped_split(make_contexts(120,seed=7),seed=9)
    seen={s:set(tuple(r["split_group"]) for r in rows if r["split"]==s) for s in ("train","validation","test")}
    assert not (seen["train"]&seen["validation"] or seen["train"]&seen["test"] or seen["validation"]&seen["test"])

def test_direct_drops_old_assignments():
    x=direct_example(make_contexts(6)[0]); assert all(not k.startswith("O_") for k in x["order"])
