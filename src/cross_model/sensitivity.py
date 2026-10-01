"""Fixed precision sensitivity subset; uses a passing int8 gate, never model selection."""
import copy
import logging
from pathlib import Path

from src.cross_model.analysis import contrasts, summarize
from src.cross_model.mechanism import patch_pair, relevance, selected_pairs
from src.cross_model.protocol import CONTRACT, manifest, sealed, write_new
from src.cross_model.scoring import score_row
from src.cross_model.tokens import audit_pairs, check_tokenizer
from src.cross_model.workflow import score_info, verify_confirmation
from src.cross_model.progress import progress

logger = logging.getLogger(__name__)


def run(a, config, candidate):
    spec = CONTRACT['quantization_sensitivity']
    if config['model']['id'] not in spec['models'] or config['model']['quantization'] != 'int8':
        raise ValueError('fixed sensitivity panel requires Qwen8 or Phi4-mini int8 baseline')
    logger.info("Validating confirmatory artifacts for fixed precision-sensitivity panel")
    rows,_ = verify_confirmation(a.dataset,config,a.config,a.candidates)
    _,saved,_,_ = score_info(a.scores,a.dataset,config,a.config,a.candidates,'confirmatory')
    ids = sorted({r['history_id'] for r in rows})[:spec['histories']]
    subset = [r for r in rows if r['history_id'] in ids]
    baseline = [s for s in saved if s['history_id'] in ids]
    # Both models use the same interventions; no precision-dependent pair filtering.
    _,all_pairs = selected_pairs(rows)
    pairs = {pid:m for pid,m in all_pairs.items() if m[0]['history_id'] in ids and
             m[0]['condition']=='superseded' and m[0]['edited_field'].startswith('initial')}
    reference = copy.deepcopy(config)
    reference['model']['quantization'] = 'none'
    reference['resolved_quantization'] = 'none'
    from src.cross_model.runtime import load_pinned_model, hook_smoke
    from src.cross_model.adapters import get_decoder_blocks
    original_records = []
    with_precision = []
    for mode,c in progress([('int8',config),('float16',reference)], desc='Precision conditions', unit='condition'):
        logger.info("Loading %s model for precision sensitivity", mode)
        model,tok = load_pinned_model(c);check_tokenizer(tok,candidate)
        if not audit_pairs(subset,tok)['all_mechanism_aligned']:raise ValueError('precision subset alignment failed')
        from src.data.supersession_behavior import render_behavior_example
        smoke=hook_smoke(model,tok,render_behavior_example(subset[0],tok,True))
        n = len(get_decoder_blocks(model));records=[]
        tasks=[(pid,members,depth,site) for pid,members in sorted(pairs.items())
               for depth in spec['patch_depths'] for site in spec['sites']]
        for pid,members,depth,site in progress(tasks, desc=f'{mode} patch tasks', unit='task', leave=False):
            layer=round(depth*(n-1))
            records.append(patch_pair(model,tok,members,candidate,layer,'block_output',site))
        if mode=='int8':
            causal=baseline
        else:
            causal=[score_row(model,tok,r,candidate,False)
                    for r in progress(subset, desc='Scoring float16 behavior', unit='row', leave=False)]
        with_precision.append({'precision':mode,'behavior':summarize(contrasts(subset,causal)),
                               'patch_history_relevance':relevance(records),
                               'activation_dtypes':sorted({d for r in records for d in r['activation_dtypes']}),
                               'real_hook_smoke':smoke, 'actual_config':c, 'hf_device_map':{k:str(v) for k,v in getattr(model,'hf_device_map',{}).items()}})
        del model
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # This is sensitivity evidence, never an extra confirmatory replication.
    a_rows,b_rows = [p['patch_history_relevance'] for p in with_precision]
    keys=lambda r:(r['history_id'],r['condition'],r['binding'],r['component'],r['site'],r['layer'])
    by_key={keys(r):r for r in b_rows}
    deltas=[{**r,'float16_minus_int8_patch_relevance':by_key[keys(r)]['patch_relevance']-r['patch_relevance']}
            for r in a_rows]
    logger.info("Writing precision-sensitivity results to %s", a.output)
    write_new(a.output,sealed({'stage':'quantization_sensitivity','history_ids':ids,
        'reference_gate_unchanged':True,'precision_results':with_precision,'paired_patch_deltas':deltas,
        'provenance':manifest(config,a.config,a.candidates,a.dataset),
        'memory_policy':'Auto dispatch may offload unquantized Qwen; actual device map recorded; failure is a technical stop'}))
