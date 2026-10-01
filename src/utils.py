import yaml, json, hashlib, subprocess, platform, importlib.metadata as md
from pathlib import Path
from datetime import datetime, timezone

def load_config(path):
    with open(path,encoding="utf8") as f: return yaml.safe_load(f)
def provenance(config,dataset_path=None):
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: commit=None
    pkgs={}
    for x in ("torch","transformers","accelerate","bitsandbytes","tqdm","numpy","scikit-learn","pandas","matplotlib","PyYAML"):
        try: pkgs[x]=md.version(x)
        except md.PackageNotFoundError: pkgs[x]=None
    digest=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest() if dataset_path and Path(dataset_path).exists() else None
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),"git_commit":commit,"config":config,"dataset_sha256":digest,"model_id":config.get("model",{}).get("id"),"model_revision":config.get("resolved_model_revision",config.get("model",{}).get("revision")),"tokenizer_id":config.get("model",{}).get("tokenizer_id") or config.get("model",{}).get("id"),"tokenizer_revision":config.get("resolved_tokenizer_revision",config.get("model",{}).get("tokenizer_revision")),"seed":config.get("seed"),"packages":pkgs,"python":platform.python_version(),"device":config.get("model",{}).get("device"),"dtype":config.get("model",{}).get("dtype"),"quantization":config.get("resolved_quantization",config.get("model",{}).get("quantization","none")),"device_map":config.get("model",{}).get("device_map"),"chat_template":config.get("model",{}).get("chat_template"),"chat_template_settings":{"enable_thinking":False,"answer_prefix":"Answer:","candidate_continuation_prefix":" ","answer_only_instruction":True},"transformers_compatibility_shims":config.get("transformers_compatibility_shims",[])}
def save_json(data,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2,default=str)+"\n",encoding="utf8")
