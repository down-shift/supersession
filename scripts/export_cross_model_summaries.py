"""Export metadata-stratified summaries from sealed saved artifacts; no inference."""
import csv
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.cross_model import robustness_protocol as design
from src.cross_model import robustness_v2 as v2
from src.cross_model.protocol import read_sealed, sealed, write_new
from src.cross_model.robustness_analysis import summary as history_summary
from src.data.io import sha256_file, read_jsonl

MODELS = ('qwen3_8b', 'gemma3_4b', 'phi4_mini')
V1_ROOT = Path('outputs/cross_model_v1_reanalysis_corrected_20261003')
V2_ROOT = Path('outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003')
V2_FREEZE = V2_ROOT / 'protocol_freeze.json'


def v1_attribute_frame_summaries(model, report, dataset_rows):
    """Group existing corrected per-history estimates by frozen dataset metadata."""
    metadata = {}
    for row in dataset_rows:
        value = (row['attribute'], row['prompt_variant'], row['prompt_family'])
        previous = metadata.setdefault(row['history_id'], value)
        if previous != value:
            raise ValueError(f"{row['history_id']}: target attribute/frame changes within a history")
    histories = {r['history_id']: r for r in report['history_rows']}
    if set(histories) != set(metadata):
        raise ValueError('saved reanalysis histories do not match dataset metadata')
    primary = 'R_superseded_minus_R_irrelevant_counterbalanced'
    live = 'R_live_minus_R_superseded'
    if primary not in report['statistics']['results'] or live not in report['statistics']['results']:
        raise ValueError('corrected saved report lacks the established v1 contrasts')
    groups = defaultdict(list)
    frame_groups = defaultdict(list)
    for hid, row in histories.items():
        attribute, frame, family = metadata[hid]
        groups[(attribute, frame, family)].append(row)
        frame_groups[(frame, family)].append(row)
    rows = []

    def append(level, attribute, frame, family, members):
        primary_stats = history_summary([x[primary] for x in members])
        live_stats = history_summary([x[live] for x in members])
        rows.append({'model': model, 'summary_level': level, 'target_attribute': attribute,
                     'lexical_frame': frame, 'prompt_family': family,
                     'n_histories': len(members), 'primary_contrast': primary,
                     'primary': primary_stats, 'live_minus_superseded': live_stats})

    for (attribute, frame, family), members in sorted(groups.items()):
        append('attribute × frame', attribute, frame, family, members)
    for (frame, family), members in sorted(frame_groups.items()):
        append('frame marginal', 'all attributes', frame, family, members)
    overall = next(r for r in rows if r['summary_level'] == 'frame marginal')
    if overall['primary'] != report['statistics']['results'][primary]:
        raise ValueError('stratified export changed the original full-sample primary estimate')
    if overall['live_minus_superseded'] != report['statistics']['results'][live]:
        raise ValueError('stratified export changed the original full-sample secondary contrast')
    return rows


def interval(value):
    lo, hi = value['ci95_cluster_bootstrap']
    return f"{value['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def span_gap(a_start, a_end, b_start, b_end):
    """Number of unoccupied tokens between two half-open spans; zero on overlap/adjacency."""
    if min(a_start, a_end, b_start, b_end) < 0 or a_end < a_start or b_end < b_start:
        raise ValueError('invalid token span')
    return max(0, b_start - a_end, a_start - b_end)


def span_to_position_distance(start, end, position):
    """Minimum token-index distance from tokens in [start,end) to one position."""
    if start < 0 or end <= start or position < 0:
        raise ValueError('invalid token span or position')
    if position < start:
        return start - position
    if position >= end:
        return position - (end - 1)
    return 0


def summarize_v2_token_geometry(model, candidate, rows, freeze_sha256):
    """Describe final-v2 validation token distances from existing tokenizer maps."""
    if candidate.get('contract') != design.CONTRACT or candidate.get('stage') != 'validation':
        raise ValueError(f'{model}: candidate map is not final-v2 tokenizer validation')
    if candidate.get('freeze_sha256') != freeze_sha256:
        raise ValueError(f'{model}: tokenizer geometry binds to a different protocol freeze')
    if candidate.get('edit_audit', {}).get('status') != 'passed':
        raise ValueError(f'{model}: tokenizer edit audit did not pass')
    if (not candidate.get('canonical_raw_R_defined')
            or set(candidate.get('canonical_one_token_ids', {})) != set(design.VALUES)):
        raise ValueError(f'{model}: complete canonical candidate audit is missing')
    if candidate.get('edit_audit', {}).get('all_mechanism_aligned') is not True:
        raise ValueError(f'{model}: paired-edit span alignment did not pass')
    lineage = design.verify_v1_lineage(candidate['provenance']['config'])
    if candidate.get('original_v1_lineage') != lineage:
        raise ValueError(f'{model}: original-v1 score/candidate lineage check is stale')
    geometry = candidate.get('token_span_geometry')
    if not geometry or set(geometry.get('example_to_prompt_sha256', {})) != {r['example_id'] for r in rows}:
        raise ValueError(f'{model}: token geometry does not cover validation members')
    groups = defaultdict(list)
    links = geometry['example_to_prompt_sha256']
    by_prompt = geometry['by_prompt_sha256']
    for row in rows:
        if row['pair_direction'] != 1:
            continue
        phash = links[row['example_id']]
        entry = by_prompt.get(phash)
        if not entry or entry.get('prompt_sha256') != phash:
            raise ValueError(f'{model}: prompt geometry hash mismatch')
        spans = entry['spans']
        field = row['edited_field']
        if field not in spans or 'queried_entity' not in spans:
            raise ValueError(f'{model}: edited-value or queried-entity span is missing')
        value, query = spans[field], spans['queried_entity']
        if value['text'] != row['replacement_value'] or query['text'] != row['query_entity']:
            raise ValueError(f'{model}: token span text does not match the validation member')
        start, end = value['token_start'], value['token_end_exclusive']
        qstart, qend = query['token_start'], query['token_end_exclusive']
        if not (0 <= start < end <= entry['prompt_token_length']
                and 0 <= qstart < qend <= entry['prompt_token_length']):
            raise ValueError(f'{model}: token span falls outside the rendered prompt')
        final_preanswer = entry['prompt_token_length'] - 1
        key = (row['condition'], row['historical_entity_order'], row['current_entity_order'],
               row['edited_variable'] == row['query'])
        groups[key].append((span_gap(start, end, qstart, qend),
                            span_to_position_distance(start, end, final_preanswer)))
    result = []
    for (condition, historical_order, current_order, edit_matches_query), pairs in sorted(groups.items()):
        query_gaps, preanswer_distances = zip(*pairs)
        result.append({'model': model, 'condition': condition,
                       'historical_entity_order': historical_order,
                       'current_entity_order': current_order,
                       'edited_entity_is_queried_entity': edit_matches_query,
                       'n_edited_members': len(pairs),
                       'query_entity_gap_tokens': {'mean': statistics.mean(query_gaps),
                           'median': statistics.median(query_gaps), 'min': min(query_gaps), 'max': max(query_gaps)},
                       'final_preanswer_distance_tokens': {'mean': statistics.mean(preanswer_distances),
                           'median': statistics.median(preanswer_distances), 'min': min(preanswer_distances),
                           'max': max(preanswer_distances)}})
    if len(result) != len(v2.CONDITIONS) * 2 * 2 * 2:
        raise ValueError(f'{model}: incomplete condition/order/query-match geometry summaries')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('outputs/cross_model_reporting_20261003'))
    args = parser.parse_args()
    amendment_path = Path('docs/analysis_amendment_20261003.md')
    amendment_record_path = V2_ROOT / 'analysis_amendment_record.json'
    amendment_record = read_sealed(amendment_record_path)
    if amendment_record.get('amendment_sha256') != sha256_file(amendment_path):
        raise ValueError('dated analysis amendment differs from its provenance record')

    v1_rows = []
    original_report_hashes = {}
    for model in MODELS:
        report_path = V1_ROOT / f'{model}.json'
        report = read_sealed(report_path)
        dataset = Path(report['dataset_path'])
        scores = Path(report['scores_path'])
        if (sha256_file(dataset) != report['dataset_sha256']
                or sha256_file(scores) != report['scores_sha256']):
            raise ValueError(f'{model}: saved-score reanalysis source hashes no longer match')
        original_report_hashes[model] = sha256_file(report_path)
        v1_rows.extend(v1_attribute_frame_summaries(model, report, read_jsonl(dataset)))

    freeze = design.verify_freeze(V2_FREEZE)
    freeze_hash = sha256_file(V2_FREEZE)
    validation_histories, validation_rows = v2.generate('validation')
    v2_rows, model_status = [], {}
    for model in MODELS:
        path = V2_ROOT / model / 'candidates.json'
        if not path.exists():
            model_status[model] = {
                'status': 'pending_workspace_artifact',
                'reason': ('no final-freeze tokenizer geometry artifact is present in this workspace; '
                           'the user reports pinned files are available on a remote PC, which was not accessed; '
                           'geometry is therefore unverified')}
            continue
        candidate = read_sealed(path)
        if candidate.get('provenance', {}).get('model_id') != design.MODEL_SETTINGS[model][0]:
            raise ValueError(f'{model}: model identity differs from pinned settings')
        if candidate.get('freeze_sha256') != freeze_hash:
            raise ValueError(f'{model}: candidate map freeze hash mismatch')
        if candidate.get('provenance', {}).get('code_sha256') != design.code_hash():
            raise ValueError(f'{model}: candidate map protocol code hash is stale')
        if candidate.get('surface_geometry_audit') != __import__('src.cross_model.tokens', fromlist=['surface_geometry_audit']).surface_geometry_audit(candidate['events']):
            raise ValueError(f'{model}: candidate surface geometry audit is invalid')
        v2_rows.extend(summarize_v2_token_geometry(model, candidate, validation_rows, freeze_hash))
        model_status[model] = {'status': 'passed', 'candidate_path': str(path),
                               'candidate_sha256': sha256_file(path),
                               'tokenizer_sha256': candidate['tokenizer_sha256'],
                               'chat_template_sha256': candidate['chat_template_sha256'],
                               'lineage': candidate['original_v1_lineage']}

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    v1_json = out_dir / 'cross_model_v1_attribute_frame_summaries.json'
    v1_csv = out_dir / 'cross_model_v1_attribute_frame_summaries.csv'
    v1_md = out_dir / 'cross_model_v1_attribute_frame_summaries.md'
    v2_json = out_dir / 'relational_v2_token_distances.json'
    v2_csv = out_dir / 'relational_v2_token_distances.csv'
    v2_md = out_dir / 'relational_v2_token_distances.md'
    paths = (v1_json, v1_csv, v1_md, v2_json, v2_csv, v2_md)
    if any(path.exists() for path in paths):
        raise FileExistsError('one or more reporting export paths already exist; choose a fresh directory')

    v1_artifact = {'source_reports': original_report_hashes,
                   'method': 'group existing corrected per-history v1 estimates by dataset attribute/prompt_variant; reuse frozen history-bootstrap summary; full-sample estimate checked exact against source report',
                   'primary_estimand_unchanged': True, 'rows': v1_rows}
    write_new(v1_json, sealed(v1_artifact))
    with v1_csv.open('x', newline='', encoding='utf-8') as handle:
        fields = ['model', 'summary_level', 'target_attribute', 'lexical_frame', 'prompt_family',
                  'n_histories', 'contrast', 'mean_nats', 'ci95_low_nats', 'ci95_high_nats', 'fraction_positive']
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in v1_rows:
            for key, label in (('primary', row['primary_contrast']),
                               ('live_minus_superseded', 'R_live − R_superseded')):
                stats = row[key]
                writer.writerow({'model': row['model'], 'summary_level': row['summary_level'],
                    'target_attribute': row['target_attribute'], 'lexical_frame': row['lexical_frame'],
                    'prompt_family': row['prompt_family'], 'n_histories': row['n_histories'],
                    'contrast': label, 'mean_nats': f"{stats['mean']:.8f}",
                    'ci95_low_nats': f"{stats['ci95_cluster_bootstrap'][0]:.8f}",
                    'ci95_high_nats': f"{stats['ci95_cluster_bootstrap'][1]:.8f}",
                    'fraction_positive': f"{stats['fraction_positive']:.8f}"})
    lines = ['# v1 target-attribute and lexical-frame summaries', '',
             'Descriptive history-bootstrap stratifications of the existing corrected saved-score analyses. '
             'The full-sample history-bootstrap estimates remain primary and are checked unchanged against the source reports.', '',
             '| Model | Slice | Attribute | Lexical frame | n | Superseded − counterbalanced unassigned, nats [95% CI] | Live − superseded, nats [95% CI] |',
             '|---|---|---|---|---:|---:|---:|']
    for row in v1_rows:
        lines.append(f"| {row['model']} | {row['summary_level']} | {row['target_attribute']} | {row['lexical_frame']} | {row['n_histories']} | {interval(row['primary'])} | {interval(row['live_minus_superseded'])} |")
    with v1_md.open('x', encoding='utf-8') as handle: handle.write('\n'.join(lines)+'\n')

    v2_artifact = {'freeze_path': str(V2_FREEZE), 'freeze_sha256': freeze_hash,
                   'validation_histories': len(validation_histories), 'model_status': model_status,
                   'distance_definitions': {'query_entity_gap_tokens': 'count of tokens between half-open edited-value and queried-entity spans; overlap or adjacency is zero',
                       'final_preanswer_distance_tokens': 'minimum absolute token-index distance from the edited-value span tokens to prompt_token_length - 1'},
                   'scope': 'descriptive tokenizer-validation sample only; not scored behavioral data',
                   'rows': v2_rows}
    write_new(v2_json, sealed(v2_artifact))
    if v2_rows:
        with v2_csv.open('x', newline='', encoding='utf-8') as handle:
            fields = ['model', 'condition', 'historical_entity_order', 'current_entity_order',
                      'edited_entity_is_queried_entity', 'n_edited_members',
                      'query_gap_mean_tokens', 'query_gap_median_tokens', 'query_gap_min_tokens', 'query_gap_max_tokens',
                      'preanswer_distance_mean_tokens', 'preanswer_distance_median_tokens',
                      'preanswer_distance_min_tokens', 'preanswer_distance_max_tokens']
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
            for row in v2_rows:
                q,p=row['query_entity_gap_tokens'],row['final_preanswer_distance_tokens']
                writer.writerow({'model':row['model'],'condition':row['condition'],
                    'historical_entity_order':row['historical_entity_order'],'current_entity_order':row['current_entity_order'],
                    'edited_entity_is_queried_entity':row['edited_entity_is_queried_entity'],'n_edited_members':row['n_edited_members'],
                    'query_gap_mean_tokens':q['mean'],'query_gap_median_tokens':q['median'],'query_gap_min_tokens':q['min'],'query_gap_max_tokens':q['max'],
                    'preanswer_distance_mean_tokens':p['mean'],'preanswer_distance_median_tokens':p['median'],
                    'preanswer_distance_min_tokens':p['min'],'preanswer_distance_max_tokens':p['max']})
    else:
        v2_csv.write_text('model,condition,historical_entity_order,current_entity_order,status\n')
    geom_lines = ['# Final-v2 tokenizer-validation token distances', '',
                  'Descriptive only; these distances come from four tokenizer-validation histories per available model, not development or confirmation data.', '',
                  '| Model | Condition | Historical order | Current order | Edited entity is queried | n | Median gap to queried entity (tokens) | Median distance to final pre-answer (tokens) |',
                  '|---|---|---:|---:|---|---:|---:|---:|']
    for row in v2_rows:
        geom_lines.append(f"| {row['model']} | {row['condition']} | {row['historical_entity_order']} | {row['current_entity_order']} | {row['edited_entity_is_queried_entity']} | {row['n_edited_members']} | {row['query_entity_gap_tokens']['median']} | {row['final_preanswer_distance_tokens']['median']} |")
    for model,status in model_status.items():
        if status['status'] != 'passed': geom_lines.append(f"| {model} | — | — | — | — | — | pending | {status['reason']} |")
    with v2_md.open('x',encoding='utf-8') as handle: handle.write('\n'.join(geom_lines)+'\n')
    print(json.dumps({'v1_rows':len(v1_rows),'v1_summaries':str(v1_md),'v2_geometry_rows':len(v2_rows),'v2_summaries':str(v2_md),'models':model_status},indent=2))


if __name__ == '__main__':
    main()
