#!/usr/bin/env python3
"""Tokenizer-only complete-sequence audit; does not load or run a model."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from src.utils import load_config,save_json
from src.data.downstream_transfer import generate,render
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/downstream_transfer_v1.yaml');p.add_argument('--output',required=True);p.add_argument('--values-json',default='configs/downstream_transfer_values.json');p.add_argument('--codes-json',default='configs/downstream_transfer_codes.json');a=p.parse_args()
c=load_config(a.config);m=c['model'];tok=AutoTokenizer.from_pretrained(m.get('tokenizer_id') or m['id'],revision=m.get('tokenizer_revision') or m['revision'])
r=generate('development',24,json.loads(Path(a.values_json).read_text()),json.loads(Path(a.codes_json).read_text()))[0];prompt=render(r,tok,m.get('chat_template',True));audit={}
prefix=tok(prompt,add_special_tokens=False)['input_ids']
for code in r['code_vocabulary']:
 full=tok(prompt+' '+code,add_special_tokens=False)['input_ids']
 if full[:len(prefix)]!=prefix or len(full)<=len(prefix): raise ValueError(f'code {code} is not a stable continuation')
 audit[code]={'token_ids':full[len(prefix):],'n_tokens':len(full)-len(prefix)}
doc={'protocol':'downstream_transfer_v1','audit':'tokenizer_only_no_model_inference','tokenizer_id':m.get('tokenizer_id') or m['id'],'tokenizer_revision':getattr(tok,'_commit_hash',None) or m.get('tokenizer_revision') or m['revision'],'chat_template':m.get('chat_template',True),'continuation_prefix_stable':True,'codes':audit}
save_json(doc,a.output);print(f"audited {len(audit)} code candidates; token lengths={sorted({v['n_tokens'] for v in audit.values()})}")
