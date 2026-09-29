from src.data.generate import make_contexts,counterfactual_pair
from src.data.generate import render_example

def test_each_counterfactual_changes_only_requested_semantic_value():
    base=make_contexts(6,seed=1)[0]
    for role in ("O_q","C_q","O_d","C_d"):
        a,b=counterfactual_pair(base,role,"amber","birch")
        assert a["roles"][role]=="amber" and b["roles"][role]=="birch"
        for key in ("O_q","C_q","O_d","C_d"):
            if key!=role: assert a["roles"][key]==b["roles"][key]
        changed=[k for k in ("old_x","old_z","current_x","current_z") if a[k]!=b[k]]
        assert len(changed)==1
        assert a["order"]==b["order"] and a["query"]==b["query"] and a["variables"]==b["variables"]

def test_rendered_counterfactual_only_replaces_one_assignment_value():
    a,b=counterfactual_pair(make_contexts(6,seed=2)[0],"O_q","amber","birch")
    ta,tb=render_example(a),render_example(b)
    assert ta!=tb
    assert ta.replace("amber","birch")==tb
