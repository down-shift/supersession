from scripts.analyze_named_head_decisions import FIELDS,NAMED,summarize

def test_named_head_decision_summary_uses_head_column_and_symmetric_history_bootstrap():
    rows=[]
    for hid in ('h0','h1'):
        for (layer,head) in NAMED:
            for axis in ('x','z'):
                rows.append({'history_id':hid,'layer':layer,'head':head,'edited_binding':'old_'+axis,'direction':'baseline_to_edited',
                    'baseline_current_minus_old':2.,'patched_current_minus_old':3.,'patch_delta_current_minus_old':1.,
                    'baseline_current_logit':5.,'baseline_old_logit':3.,'patched_current_logit':6.,'patched_old_logit':3.})
    history,summary=summarize(rows,n_boot=100)
    assert len(history)==len(NAMED)*4
    assert len(summary)==len(NAMED)
    assert summary[0]['patch_delta_current_minus_old_symmetric_mean']==1.
    assert summary[0]['patched_current_change_x_median']==1.
    assert summary[0]['patched_obsolete_change_z_mean']==0.
