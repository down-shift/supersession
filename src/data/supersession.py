"""Fresh controlled datasets for binding, stale retrieval, and version selection.

These generators are independent of the frozen four-query datasets.
"""
from __future__ import annotations
import random

def make_control_history(history_id, values, condition, seed=0, variables=('x','z')):
    """Matched live, superseded, or irrelevant-occurrence history with explicit roles."""
    if condition not in ('live','superseded','irrelevant'): raise ValueError('unknown control condition')
    if len(values)<4: raise ValueError('at least four values required')
    rng=random.Random(seed); source,other,current,decoy=rng.sample(list(values),4); x,z=variables
    if condition=='live': rows=[f'Assignment: {x} = {source}',f'Assignment: {z} = {other}']; answer=source; status='live_current'
    elif condition=='superseded': rows=[f'Initial assignment: {x} = {source}',f'Initial assignment: {z} = {other}',f'Update: {x} = {current}',f'Update: {z} = {decoy}']; answer=current; status='superseded_initial'
    else: rows=[f'Assignment: {x} = {current}',f'Assignment: {z} = {other}',f'Unassigned candidate: {source}',f'Unassigned candidate: {decoy}']; answer=current; status='irrelevant_occurrence'
    return {'history_id':history_id,'condition':condition,'variables':[x,z],'query':'x','answer':answer,'source_value':source,
      'current_x':answer,'old_x':source if condition=='superseded' else None,'semantic_status':status,'context_lines':rows}

def render_control_example(history, query='x'):
    variable=history['variables'][0 if query=='x' else 1]
    question=f'What is {variable}?' if history['condition']=='live' else f'After all updates, what is {variable}?'
    return '\n'.join(history['context_lines']+[question,'Answer:'])

def make_status_history(history_id, values, seed=0, variables=('x','z')):
    """Return YES/NO records sharing the same values, variables, and layout."""
    if len(values)<4 or len(set(values))!=len(values): raise ValueError('provide at least four distinct values')
    rng=random.Random(seed); initial_x,initial_z,proposal_x,proposal_z=rng.sample(list(values),4)
    rows=[]
    for status in ('YES','NO'):
        current_x=proposal_x if status=='YES' else initial_x
        current_z=proposal_z if status=='YES' else initial_z
        rows.append({'history_id':f'{history_id}:{status.lower()}','variables':list(variables),'initial_x':initial_x,'initial_z':initial_z,
          'proposed_x':proposal_x,'proposed_z':proposal_z,'update_accepted':status=='YES','status':status,
          'current_x':current_x,'current_z':current_z,
          'x_versions':[initial_x,proposal_x] if status=='YES' else [initial_x],
          'z_versions':[initial_z,proposal_z] if status=='YES' else [initial_z],
          'semantic_status':{'initial_x':'superseded_initial' if status=='YES' else 'accepted_current','proposed_x':'accepted_current' if status=='YES' else 'rejected_update',
            'initial_z':'superseded_initial' if status=='YES' else 'accepted_current','proposed_z':'accepted_current' if status=='YES' else 'rejected_update'}})
    return rows

def render_status_example(history, query='x'):
    x,z=history['variables']; accepted='YES' if history['update_accepted'] else 'NO'
    return (f'Initial assignment: {x} = {history["initial_x"]}\nInitial assignment: {z} = {history["initial_z"]}\n'
      f'Proposed update: {x} = {history["proposed_x"]}\nUpdate accepted: {accepted}\n'
      f'Proposed update: {z} = {history["proposed_z"]}\nUpdate accepted: {accepted}\n'
      f'After all accepted updates, what is {history["variables"][0 if query=="x" else 1]}?\nAnswer:')

def render_version_chain(history, query='x'):
    rows=[]
    for i in range(history['depth']+1):
        for variable in ('x','z'):
            prefix='Initial assignment' if i==0 else 'Update'
            rows.append(f'{prefix}: {history["variables"][0 if variable=="x" else 1]} = {history[f"{variable}_versions"][i]}')
    variable=history['variables'][0 if query=='x' else 1]
    return '\n'.join(rows+[f'After all updates, what is {variable}?','Answer:'])

def status_counterfactual_pair(history, field):
    """Create baseline/edit members changing exactly one named value field."""
    if field not in ('initial_x','initial_z','proposed_x','proposed_z'): raise ValueError('field must identify an initial or proposed value')
    values=[history[k] for k in ('initial_x','initial_z','proposed_x','proposed_z')]
    replacement=next((v for v in ('maple','navy','ochre','pearl','quartz','ruby','silver') if v not in values),None)
    if replacement is None: raise ValueError('no unused replacement candidate available')
    edit={**history,field:replacement,'semantic_status':dict(history['semantic_status'])}
    for role in ('x','z'):
        init=f'initial_{role}'; prop=f'proposed_{role}'
        if field==init:
            if history['update_accepted']: edit[f'{role}_versions']=[replacement,history[prop]]
            else: edit[f'{role}_versions']=[replacement]
        if field==prop:
            if history['update_accepted']: edit[f'{role}_versions']=[history[init],replacement]
        edit[f'current_{role}']=edit[f'{role}_versions'][-1]
        edit['semantic_status'][init]='superseded_initial' if history['update_accepted'] else 'accepted_current'
        edit['semantic_status'][prop]='accepted_current' if history['update_accepted'] else 'rejected_update'
    return ({'pair_direction':0,**history},{'pair_direction':1,**edit,'replacement_value':replacement,'edited_field':field})

def make_version_chain(history_id, values, depth, seed=0):
    if depth not in (0,1,2,4,8): raise ValueError('depth must be one of 0, 1, 2, 4, 8')
    if len(values)<2*(depth+1): raise ValueError('need distinct values for both version chains')
    chosen=random.Random(seed).sample(list(values),2*(depth+1)); x=chosen[:depth+1]; z=chosen[depth+1:]
    return {'history_id':history_id,'depth':depth,'x_versions':x,'z_versions':z,'current_x':x[-1],'current_z':z[-1],
      'variables':['x','z'],'version_metadata':{v:{'variable':name,'version_index':i,'distance_from_current':depth-i,'is_current':i==depth,'status':'current' if i==depth else 'obsolete'}
      for name,chain in (('x',x),('z',z)) for i,v in enumerate(chain)}}

def version_edit_pair(history, variable, version_index, replacement):
    key=f'{variable}_versions'
    if variable not in ('x','z') or not 0<=version_index<len(history[key]): raise ValueError('invalid variable/version index')
    chain=list(history[key]); old=chain[version_index]
    if replacement in set(history['x_versions']+history['z_versions']): raise ValueError('replacement collides with an existing version')
    edited={**history,key:chain[:version_index]+[replacement]+chain[version_index+1:],
      'version_metadata':{k:dict(v) for k,v in history['version_metadata'].items()}}
    meta=edited['version_metadata'].pop(old); edited['version_metadata'][replacement]={**meta,'counterfactual_replacement':True}
    edited[f'current_{variable}']=edited[key][-1]
    return {'baseline':history,'edited':edited,'variable':variable,'version_index':version_index,'source_value':old,'replacement_value':replacement}
