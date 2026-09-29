import numpy as np
from src.analysis.metrics import role_metrics,js_divergence

def test_binding_residue_and_current_control_candidate_mapping():
    ids={"cedar":0,"tulip":1,"violet":2,"amber":3}
    roles={"C_q":"cedar","O_q":"tulip","C_d":"violet","O_d":"amber"}
    result=role_metrics(np.array([4.,2.,1.,0.]),ids,roles)
    assert result["B"]==2.
    assert result["R"]==3.
    assert result["M"]==2.
    assert result["accuracy"]==1
    assert np.isclose(sum(result["candidate_probabilities"].values()),1)

def test_js_is_symmetric_and_zero_for_equal_distributions():
    a=[.7,.1,.1,.1]; b=[.1,.7,.1,.1]
    assert js_divergence(a,a)==0
    assert np.isclose(js_divergence(a,b),js_divergence(b,a))
