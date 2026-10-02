"""Fixed, paired minimal localization battery. No effect-dependent site selection."""
from collections import defaultdict
import logging
from pathlib import Path

from src.cross_model.adapters import ActivationHook, get_decoder_blocks, head_dimensions, normalized_depth
from src.cross_model.protocol import CONTRACT, check_manifest, manifest, read_sealed, sealed, write_new
from src.cross_model.scoring import score_prompt
from src.cross_model.tokens import audit_pairs, check_tokenizer, encode, semantic_positions
from src.cross_model.workflow import score_info, verify_confirmation
from src.data.supersession_behavior import edited_member
from src.cross_model.progress import progress

logger = logging.getLogger(__name__)


def text_decoder_layer_count(model_config):
    """Read decoder depth from flat or multimodal AutoConfig dictionaries."""
    text_config = model_config.get('text_config')
    if isinstance(text_config, dict):
        count = text_config.get('num_hidden_layers')
        if count is not None:
            if type(count) is not int or count < 1:
                raise ValueError('invalid text decoder layer count in model_config.text_config')
            return count
    count = model_config.get('num_hidden_layers')
    if type(count) is not int or count < 1:
        raise ValueError('model config has no valid text decoder num_hidden_layers')
    return count


def selected_pairs(rows, *, include_current=True):
    """First 12 fixed histories: 6 head discovery + 6 head reserve, never selected on R."""
    ids = sorted({r['history_id'] for r in rows})[:CONTRACT['mechanism_histories']]
    pairs = defaultdict(dict)
    for row in rows:
        if row['history_id'] in ids: pairs[row['pair_id']][row['pair_direction']] = row
    # Additional current binding identity edits in superseded histories, fixed before results.
    for members in list(pairs.values()):
        b = members[0]
        if not include_current or b['condition'] != 'superseded': continue
        field = 'proposed_' + b['edited_variable']
        clone = {**b, 'edited_field':field, 'source_value':b['matching_values'][field],
                 'replacement_value':b['replacement_values'][field], 'edit_status':'accepted_current',
                 'pair_id': b['pair_id']+':current_binding'}
        clone['example_id'] = clone['pair_id']+':0'
        pairs[clone['pair_id']] = {0:clone, 1:edited_member(clone)}
    return ids, pairs


def site_map(row, tokenizer):
    _, raw = semantic_positions(row, tokenizer)
    variable = row['edited_variable']; other = 'z' if variable == 'x' else 'x'
    sites = {'edited_value_span':raw[row['edited_field']], 'queried_entity':raw['queried_entity'],
             'final_preanswer':raw['final_preanswer']}
    for name, field in [('historical_value_span','initial_'+variable),
                        ('current_value_span',row['current_fields'][variable]),
                        ('distractor_value_span',row['current_fields'][other])]:
        if field in raw: sites[name] = raw[field]
    return sites


def patch_pair(model, tokenizer, members, candidate, layer, component, site, head=None, cache=None):
    import torch
    source, replacement = members[0]['source_value'], members[0]['replacement_value']
    values = (source, replacement)
    prompts, positions, unpatched = {}, {}, {}
    for direction, row in members.items():
        prompts[direction], _ = semantic_positions(row, tokenizer)
        positions[direction] = site_map(row, tokenizer)[site]
        known = cache.setdefault(prompts[direction], {}) if cache is not None else {}
        missing = [v for v in values if v not in known]
        if missing:
            known.update(score_prompt(model, tokenizer, prompts[direction], candidate['events'], values=missing)[0])
        unpatched[direction] = {v:known[v] for v in values}
    deltas = []
    dtypes = []
    for donor, recipient in ((1,0), (0,1)):
        device = model.get_input_embeddings().weight.device
        with ActivationHook(model, layer, component, positions[donor], head=head) as capture:
            with torch.inference_mode(): model(input_ids=torch.tensor([encode(tokenizer, prompts[donor])],device=device), use_cache=False)
        activation = capture.activation
        if activation is None: raise RuntimeError('hook failed to capture')
        dtypes.append(str(activation.dtype))
        factory = lambda: ActivationHook(model, layer, component, positions[recipient], source=activation, head=head)
        patched = score_prompt(model, tokenizer, prompts[recipient], candidate['events'], factory, values=values)[0]
        source, replacement = members[0]['source_value'], members[0]['replacement_value']
        def margin(mass): return mass[replacement]-mass[source]
        sign = 1 if donor == 1 else -1
        deltas.append(sign*(margin(patched)-margin(unpatched[recipient])))
    b = members[0]
    return {'pair_id':b['pair_id'], 'history_id':b['history_id'], 'condition':b['condition'],
            'edited_field':b['edited_field'], 'edited_variable':b['edited_variable'], 'query':b['query'],
            'slot_order':b.get('unassigned_slot_order'), 'layer':layer,
            'normalized_depth':normalized_depth(layer,len(get_decoder_blocks(model))),
            'component':component, 'site':site, 'head':head,
            'donor_oriented_patch_delta':sum(deltas)/2, 'directions':deltas, 'activation_dtypes':dtypes}


def relevance(records):
    """Counterbalance slots before x/z query contrasts; reject incomplete query cells."""
    grouped = defaultdict(dict)
    for r in progress(records, desc='Summarizing patch relevance', unit='record'):
        key = (r['history_id'],r['condition'],r['edited_field'].split('_')[0],
               r['component'],r['site'],r['layer'],r['head'])
        cell = (r['edited_variable'],r['query'],r['slot_order'])
        if cell in grouped[key]: raise ValueError('duplicate mechanistic contrast member')
        grouped[key][cell] = r['donor_oriented_patch_delta']
    result = []
    for key,cells in sorted(grouped.items(), key=lambda x:str(x[0])):
        slots = ('xz','zx') if key[1]=='irrelevant_counterbalanced' else (None,)
        required = {(v,q,s) for v in ('x','z') for q in ('x','z') for s in slots}
        if set(cells) != required: raise ValueError('incomplete mechanistic query/binding contrast')
        def value(v,q): return sum(cells[(v,q,s)] for s in slots)/len(slots)
        result.append({'history_id':key[0],'condition':key[1],'binding':key[2],'component':key[3],
                       'site':key[4],'layer':key[5],'head':key[6],
                       'patch_relevance':.5*(value('x','x')-value('x','z')+value('z','z')-value('z','x'))})
    return result


def depth_summary(records, count):
    from src.analysis.supersession_behavior import summarize_histories
    hist = defaultdict(lambda:defaultdict(list))
    for r in records:
        if r['condition'] != 'superseded' or r['binding'] != 'initial' or r['component'] != 'block_output':continue
        if r['site'] not in ('historical_value_span','final_preanswer'):continue
        depth=normalized_depth(r['layer'],count)
        window='early' if depth <= .25 else 'late' if depth >= .75 else None
        if window:hist[r['history_id']][(window,r['site'])].append(r['patch_relevance'])
    output=[]
    for hid,cells in sorted(hist.items()):
        required={(w,s) for w in ('early','late') for s in ('historical_value_span','final_preanswer')}
        if set(cells)!=required:raise ValueError('incomplete preregistered depth windows')
        mean=lambda w,s:sum(cells[(w,s)])/len(cells[(w,s)])
        output.append({'history_id':hid,
            'early_historical_minus_readout':mean('early','historical_value_span')-mean('early','final_preanswer'),
            'late_readout_minus_historical':mean('late','final_preanswer')-mean('late','historical_value_span')})
    return {'history_rows':output,'summaries':{k:summarize_histories([r[k] for r in output])
            for k in ('early_historical_minus_readout','late_readout_minus_historical')},
            'interpretation':'Two fixed depth/position contrasts; no literal information movement'}


def build_tasks(ids, pairs, tokenizer, count, *, heads=False, model=None):
    """Fixed task grid, inspectable cheaply without pretrained weights in core mode."""
    tasks = []
    core = CONTRACT['mechanism_core']
    late = sorted({round(x*(count-1)) for x in core['secondary_depths']})
    for pid,members in progress(sorted(pairs.items()), desc='Planning intervention tasks', unit='pair'):
        row = members[0]
        if not heads:
            if row['condition'] not in core['conditions'] or not row['edited_field'].startswith('initial_'):
                continue
            sites = site_map(row,tokenizer)
            for layer in range(count):
                for site in core['all_layer_sites']:
                    if site not in sites: raise ValueError('missing required core semantic site')
                    tasks.append((pid,layer,'block_output',site,None))
            for layer in late:
                for component in ('attention_output','mlp_output'):
                    tasks.append((pid,layer,component,'final_preanswer',None))
                if row['history_id'] in ids[:core['secondary_histories']]:
                    for site in core['secondary_sites']:
                        if site not in sites: raise ValueError('missing required secondary semantic site')
                        tasks.append((pid,layer,'block_output',site,None))
        else:
            discovery = row['history_id'] in ids[:CONTRACT['head_discovery_histories']]
            if discovery and not (row['condition']=='superseded' and row['edited_field'].startswith('initial')):
                continue
            for layer in late:
                qheads,_,_ = head_dimensions(model,layer)
                for head in range(qheads):tasks.append((pid,layer,'query_head','final_preanswer',head))
    return tasks


def run(a, config, candidate):
    from src.cross_model.runtime import load_pinned_model, hook_smoke
    from src.data.progress import prepare_jsonl_progress, append_jsonl_record
    from src.data.io import read_jsonl, sha256_file
    from src.analysis.supersession_behavior import summarize_histories
    logger.info("Validating confirmatory dataset and scores before mechanism run")
    rows, info = verify_confirmation(a.dataset, config, a.config, a.candidates)
    score_info(a.scores, a.dataset, config, a.config, a.candidates, 'confirmatory')
    # Hooks run only after gate and completed confirmatory behavioral scoring.
    ids, pairs = selected_pairs(rows, include_current=a.heads)
    if not a.heads:
        pairs = {pid:m for pid,m in pairs.items() if m[0]['condition'] in CONTRACT['mechanism_core']['conditions']}
    flat = [r for m in pairs.values() for r in m.values()]
    from scripts.cross_model import tokenizer
    tok = tokenizer(config, a.local_files_only); check_tokenizer(tok, candidate)
    audit = audit_pairs(flat, tok)
    if not audit['all_mechanism_aligned']:
        write_new(a.output+'.stopped.json', sealed({'stage':'mechanism', 'reason':
            'unequal token spans/sequence lengths in the fixed battery; no pairs dropped',
            'edit_audit':audit, 'history_ids':ids, 'provenance':manifest(config,a.config,a.candidates,a.dataset)}))
        print('mechanistic branch stopped: fixed span alignment failed'); return
    logger.info("Fixed mechanism plan: histories=%d pairs=%d; loading model for real hook smoke",
                len(ids), len(pairs))
    model, tok = load_pinned_model(config); blocks = get_decoder_blocks(model); n = len(blocks)
    prov = manifest(config,a.config,a.candidates,a.dataset)
    smoke = hook_smoke(model,tok,semantic_positions(flat[0],tok)[0])
    prov.update(real_hook_smoke=smoke, score_sha256=sha256_file(a.scores), tokenizer_sha256=candidate['tokenizer_sha256'],
                chat_template_sha256=candidate['chat_template_sha256'], edit_audit=audit,
                dataset_seed=sorted({r['seed'] for r in rows}), history_ids=ids,
                resolved_device_map={k:str(v) for k,v in getattr(model,'hf_device_map',{}).items()},
                mode='heads' if a.heads else 'core')
    tasks = build_tasks(ids, pairs, tok, n, heads=a.heads, model=model)
    logger.info("Mechanism task grid contains %d tasks across %d decoder blocks", len(tasks), n)
    # Every head is recorded, but selection/profile analysis is strictly split by history.
    task_rows = [{'example_id':str(t), 'task':list(t)} for t in tasks]
    fingerprint = {k:v for k,v in prov.items() if k!='timestamp_utc'}
    completed = prepare_jsonl_progress(a.output,a.dataset,a.candidates,fingerprint,task_rows,resume=a.resume)
    expected = {r['example_id']:r['task'] for r in task_rows}
    if a.resume and Path(a.output).exists():
        for r in read_jsonl(a.output):
            if r.get('task') != expected.get(r['example_id']): raise ValueError('mechanism resume task mismatch')
    likelihood_cache = {}
    for task in progress(tasks, desc='Patching mechanism tasks', unit='task'):
        tid = str(task)
        if tid in completed: continue
        pid,layer,component,site,head = task
        record = patch_pair(model,tok,pairs[pid],candidate,layer,component,site,head,likelihood_cache)
        append_jsonl_record(a.output, {**record,'example_id':tid,'task':list(task)})
    records = read_jsonl(a.output)
    if len(records) != len(task_rows) or {r['example_id'] for r in records} != set(expected):
        raise ValueError('incomplete mechanistic task grid')
    rel = relevance(records)
    summaries = {}
    selected = None
    if a.heads:
        discovery = [r for r in rel if r['history_id'] in ids[:6]]
        profiles = defaultdict(list)
        for r in discovery: profiles[(r['layer'],r['head'])].append(r['patch_relevance'])
        # Fixed top two absolute stale relevance; negative is never labeled suppression.
        selected = sorted(profiles,key=lambda k:(-abs(sum(profiles[k])/len(profiles[k])), k))[:2]
        rel = [r for r in rel if r['history_id'] in ids[6:] and (r['layer'],r['head']) in selected]
    groups = defaultdict(list)
    for r in rel:
        key = '|'.join(str(r[k]) for k in ('condition','binding','component','site','layer','head'))
        groups[key].append(r['patch_relevance'])
    summaries = {k:summarize_histories(v) for k,v in groups.items()}
    analysis = {'stage':'mechanism','selected_heads':selected,
        'history_relevance':rel,'summaries':summaries,'provenance':prov,
        'depth_shift': None if a.heads else depth_summary(rel,n),
        'coordinates':{str(i):normalized_depth(i,n) for i in range(n)},
        'interpretation':'Donor-oriented patches; qualitative localization, no movement or circuit homology claim'}
    analysis_path = Path(a.output+'.analysis.json')
    if analysis_path.exists():
        if not a.resume: raise FileExistsError(analysis_path)
        existing = read_sealed(analysis_path)
        check_manifest(existing['provenance'],config,a.config,a.candidates,a.dataset)
        if {k:v for k,v in existing.items() if k!='provenance'} != {k:v for k,v in analysis.items() if k!='provenance'}:
            raise ValueError('existing mechanistic analysis differs from recomputed grid')
        prov = existing['provenance']
    else:
        write_new(analysis_path,sealed(analysis))
    write_new(a.output+'.provenance.json',sealed({'stage':'mechanism','scores_sha256':sha256_file(a.output),'provenance':prov}))


def execution_plan(rows, tokenizer, candidate, count):
    """Core workload estimate from fixed inputs only; never loads weights or logits."""
    ids,pairs=selected_pairs(rows,include_current=False)
    pairs={pid:m for pid,m in pairs.items() if m[0]['condition'] in CONTRACT['mechanism_core']['conditions']}
    tasks=build_tasks(ids,pairs,tokenizer,count)
    component_counts={c:sum(t[2]==c for t in tasks) for c in ('block_output','attention_output','mlp_output')}
    patch_forwards=0
    for pid,*_ in progress(tasks, desc='Estimating mechanism forward calls', unit='task'):
        b=pairs[pid][0]
        classes=[candidate['events'][b[v]] for v in ('source_value','replacement_value')]
        scoring_calls=1+sum(len(e['ids'])>1 for members in classes for e in members)
        patch_forwards+=2+2*scoring_calls  # Two donor captures and two recipient scores.
    prompts={semantic_positions(r,tokenizer)[0] for m in pairs.values() for r in m.values()}
    full_scoring_calls=1+sum(len(e['ids'])>1 for members in candidate['events'].values() for e in members)
    return {'history_ids':ids,'pair_count':len(pairs),'task_count':len(tasks),
            'tasks_by_component':component_counts,'scored_classes_per_patch':2,
            'donor_and_patched_forward_calls':patch_forwards,
            'unpatched_forward_calls_upper_bound':len(prompts)*full_scoring_calls,
            'total_forward_calls_upper_bound':patch_forwards+len(prompts)*full_scoring_calls,
            'notes':['No wall-time or GPU-memory guarantee; batch one, no KV cache.',
                     'Unpatched class masses cached per prompt; upper bound allows every class once.',
                     'Core has two all-layer sites and late secondary sites on four fixed histories.']}
