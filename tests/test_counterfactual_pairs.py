from src.data.generate import make_contexts,counterfactual_pair
from src.data.generate import render_example
import pytest

def alternate_for(base):
    return next(value for value in ("amber","birch","coral","denim","elm","frost","grape","hazel","indigo","jade","khaki","lilac","maple","navy","ochre","pearl") if value not in base["roles"].values())

def test_each_counterfactual_changes_only_requested_semantic_value():
    base=make_contexts(6,seed=1)[0]
    for role in ("O_q","C_q","O_d","C_d"):
        current=base["roles"][role]; alternate=alternate_for(base)
        a,b=counterfactual_pair(base,role,current,alternate)
        assert a["roles"][role]==current and b["roles"][role]==alternate
        for key in ("O_q","C_q","O_d","C_d"):
            if key!=role: assert a["roles"][key]==b["roles"][key]
        changed=[k for k in ("old_x","old_z","current_x","current_z") if a[k]!=b[k]]
        assert len(changed)==1
        assert a["order"]==b["order"] and a["query"]==b["query"] and a["variables"]==b["variables"]

def test_rendered_counterfactual_only_replaces_one_assignment_value():
    base=make_contexts(6,seed=2)[0]
    current=base["roles"]["O_q"]; alternate=alternate_for(base)
    a,b=counterfactual_pair(base,"O_q",current,alternate)
    ta,tb=render_example(a),render_example(b)
    assert ta!=tb
    assert ta.replace(current,alternate)==tb

def test_counterfactual_rejects_role_value_collision():
    base=make_contexts(6,seed=2)[0]
    with pytest.raises(ValueError,match="collide"):
        counterfactual_pair(base,"O_q",base["roles"]["O_q"],base["roles"]["C_q"])
