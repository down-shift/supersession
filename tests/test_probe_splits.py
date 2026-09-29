import numpy as np
from src.experiments.probes import fit_probe_curves

def test_probe_protocol_uses_common_splits_and_reports_chance():
    rng=np.random.default_rng(4); y=np.tile(np.arange(3),30); x=rng.normal(size=(90,2,6)); x[np.arange(90),:,y]+=2
    splits=np.array(["train"]*45+["validation"]*21+["test"]*24)
    rows=fit_probe_curves(x,y,splits,seed=2,C_values=(.1,1))
    assert len(rows)==2 and all(r["chance"]==1/3 for r in rows)
    assert all(0<=r["test_accuracy"]<=1 for r in rows)
