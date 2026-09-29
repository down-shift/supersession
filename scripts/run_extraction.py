#!/usr/bin/env python3
import argparse
from src.utils import load_config,provenance,save_json
from src.data.io import read_jsonl
from src.models.loader import load_model,decoder_blocks
from src.experiments.extract_activations import extract
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/primary.yaml"); p.add_argument("--dataset",default="outputs/primary/dataset.jsonl"); p.add_argument("--token-ids",default="outputs/primary/token_ids.json"); p.add_argument("--output",default="outputs/primary/activations"); a=p.parse_args()
c=load_config(a.config); m,t=load_model(c); meta=extract(m,t,read_jsonl(a.dataset),a.output,chat=c["model"].get("chat_template",True)); run=provenance(c,a.dataset); run["candidate_token_ids"]=__import__("json").load(open(a.token_ids))["token_ids"]; save_json({"examples":meta,"layer_order":["embedding_pre_block_0"]+[f"block_{i}_output" for i in range(len(decoder_blocks(m)))],"provenance":run},a.output+"/metadata.json"); print(f"saved activations for {len(meta)} examples")
