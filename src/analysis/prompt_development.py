"""Deterministic prompt-development summaries and error taxonomy."""
from collections import Counter, defaultdict

QUERY_ROLES={"current_x":"current_x","initial_x":"old_x","current_z":"current_z","initial_z":"old_z"}

def temporal_gap(record):
    var=record["query_id"][-1]
    old=record["order"].index("O_"+var); current=record["order"].index("C_"+var)
    return current-old-1

def classify_prediction(record, candidate_values):
    pred=record.get("generated_first_token","").strip()
    expected=record.get("answer")
    if pred==expected: return None
    q=record["query_id"]
    mapping={"current_x":record["current_x"],"initial_x":record["old_x"],"current_z":record["current_z"],"initial_z":record["old_z"]}
    query_var=q[-1]; query_time="initial" if q.startswith("initial") else "current"
    other_var="z" if query_var=="x" else "x"
    if pred==mapping[f"{'current' if query_time=='initial' else 'initial'}_{query_var}"]: return "same_variable_other_time"
    if pred==mapping[f"{query_time}_{other_var}"]: return "other_variable_same_time"
    if pred==mapping[f"{'current' if query_time=='initial' else 'initial'}_{other_var}"]: return "other_variable_other_time"
    if pred in candidate_values: return "other_history_candidate"
    return "off_candidate"

def summarize_errors(records, candidate_values):
    out=defaultdict(Counter); totals=Counter()
    for r in records:
        cls=classify_prediction(r,candidate_values)
        for dim,key in (("prompt_variant",r.get("prompt_variant")),("query_id",r["query_id"]),("order",tuple(r["order"])),("temporal_gap",temporal_gap(r)),("positions",(r["order"].index("O_"+r["query_id"][-1]),r["order"].index("C_"+r["query_id"][-1])))):
            totals[(dim,key)]+=1
            if cls: out[(dim,key)][cls]+=1
            else: out[(dim,key)]
    return {f"{dim}:{key}":{"wrong_count":sum(counts.values()),"n":totals[(dim,key)],"error_rate":sum(counts.values())/totals[(dim,key)],"classes":dict(counts)} for (dim,key),counts in out.items()}

def summarize_variant(records, threshold):
    byq=defaultdict(list); byorder=defaultdict(list); bygap=defaultdict(list)
    for r in records:
        acc=int(r["full_vocab_next_token_accuracy"]); byq[r["query_id"]].append(acc); byorder[" ".join(r["order"])].append(acc); bygap[temporal_gap(r)].append(acc)
    query={q:sum(v)/len(v) for q,v in sorted(byq.items())}
    return {"query_accuracy":query,"minimum_accuracy":min(query.values()),"mean_accuracy":sum(query.values())/len(query),"accuracy_by_order":{k:sum(v)/len(v) for k,v in sorted(byorder.items())},"accuracy_by_temporal_gap":{str(k):sum(v)/len(v) for k,v in sorted(bygap.items())},"meets_development_criterion":len(query)==4 and min(query.values())>=threshold}

def select_prompt(summaries):
    for variant in ("first_latest","initial_update"):
        if summaries.get(variant,{}).get("meets_development_criterion"):
            return {"selected_variant":variant,"confirmatory_permitted":True,"selection_rule":"first_latest preferred; otherwise initial_update; timestamped diagnostic only"}
    return {"selected_variant":None,"confirmatory_permitted":False,"selection_rule":"first_latest preferred; otherwise initial_update; timestamped diagnostic only"}

def gate_allows_confirmatory(query_results, threshold=.99):
    required={"current_x","initial_x","current_z","initial_z"}
    return set(query_results)==required and all(float(query_results[q]["full_vocab_next_token_accuracy"])>=threshold for q in required)
