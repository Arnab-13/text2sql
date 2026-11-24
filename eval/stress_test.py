"""
Schema-scaling stress test. Each BIRD dev question is asked against a schema padded with
K irrelevant tables borrowed from the OTHER dev databases (K = 0, 10, 30, 60 ...).
  system a: base + full schema        system b: fine-tuned B + full schema
  system c: base + retrieval          system d: fine-tuned D + retrieval
Full-schema prompts (a, b) grow with K; retrieval prompts (c, d) should stay about the same size.
The gold query still runs on the real database; a prediction that uses a distractor table is simply invalid.

  python stress_test.py --system a --dev_json J --dev_db_root R --out /kaggle/working
"""
import os, sys, json, time, random, argparse, sqlite3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))
from common.prompts import build_prompt, get_schema_ddl, extract_sql
from harness import score_dataset

HUB = {"b": "text2sql-lora-b", "d": "text2sql-lora-d"}


def table_ddls(db_path):
    """{table_name: CREATE TABLE sql} for one SQLite file."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    conn.close()
    return dict(rows)


def distractor_pool(db, all_dbs, ddls):
    """Ordered, de-duplicated list of (src_db, table) borrowed from other DBs. Seeded per DB, so the
    pool for K=10 is a prefix of the pool for K=30 (nested), and every system sees the same tables."""
    own = set(ddls[db])
    pool, seen = [], set(own)
    for d in all_dbs:
        if d == db:
            continue
        for t in ddls[d]:
            if t not in seen:
                seen.add(t); pool.append((d, t))
    random.Random(f"pool-{db}").shuffle(pool)
    return pool


def full_schema(db, k, pool, ddls, db_root):
    if k == 0:                                     # identical to the normal A/B prompt
        return get_schema_ddl(f"{db_root}/{db}/{db}.sqlite")
    stmts = list(ddls[db].values()) + [ddls[d][t] for d, t in pool[:k]]
    random.Random(f"order-{db}-{k}").shuffle(stmts)
    return ";\n\n".join(stmts) + ";"


def merged_kb(db, k, pool, kbs):
    """The real DB's KB plus the K distractor tables (columns + FKs), for retrieval."""
    kb = {"tables": list(kbs[db]["tables"]), "columns": list(kbs[db]["columns"]),
          "foreign_keys": list(kbs[db]["foreign_keys"])}
    chosen = {}
    for d, t in pool[:k]:
        chosen.setdefault(d, set()).add(t)
    for d, ts in chosen.items():
        kb["tables"] += sorted(ts)
        kb["columns"] += [c for c in kbs[d]["columns"] if c["table"] in ts]
        kb["foreign_keys"] += [f for f in kbs[d]["foreign_keys"] if f["from_table"] in ts and f["to_table"] in ts]
    return kb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["a", "b", "c", "d"], required=True)
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--dev_db_root", required=True)
    ap.add_argument("--levels", default="0,10,30,60")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--out", default="/kaggle/working")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    retrieval = args.system in ("c", "d")

    dev = sorted(json.load(open(args.dev_json)), key=lambda e: e["question_id"])
    dev = dev[::max(1, len(dev) // args.n)][:args.n]
    all_dbs = sorted(d for d in os.listdir(args.dev_db_root)
                     if os.path.isdir(f"{args.dev_db_root}/{d}") and not d.startswith("__"))
    ddls = {d: table_ddls(f"{args.dev_db_root}/{d}/{d}.sqlite") for d in all_dbs}
    pools = {d: distractor_pool(d, all_dbs, ddls) for d in all_dbs}
    print(f"{len(dev)} questions | {len(all_dbs)} DBs | distractor pool sizes: "
          f"{min(len(p) for p in pools.values())}-{max(len(p) for p in pools.values())} tables", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map={"": 0}, token=token)
    if args.system in HUB:
        from peft import PeftModel
        from huggingface_hub import HfApi, snapshot_download
        repo = f"{HfApi(token=token).whoami()['name']}/{HUB[args.system]}"
        d = snapshot_download(repo, allow_patterns="epoch-1/*", token=token)
        model = PeftModel.from_pretrained(model, os.path.join(d, "epoch-1"))
    model.eval()

    if retrieval:
        sys.path.insert(0, os.path.join(ROOT, "retriever"))
        from schema_kb import build_schema_kb, find_value_matches
        from doc_builder import build_documents
        from retriever import SchemaRetriever
        from context_builder import build_retrieved_context
        kbs = {d: build_schema_kb(args.dev_db_root, d) for d in all_dbs}
        vmatch = {}
        for i, ex in enumerate(dev):               # value search only touches the real DB; compute once, reuse per level
            vmatch[ex["question_id"]] = find_value_matches(
                f"{args.dev_db_root}/{ex['db_id']}/{ex['db_id']}.sqlite", ex["question"])
        print("value matches done", flush=True)
        rets = {}

    for k in levels:
        res_path = f"{args.out}/stress_{args.system}_K{k}_results.json"
        if os.path.exists(res_path):
            print(f"K={k}: already done, skipping", flush=True); continue
        t0 = time.time()
        prompts = {}
        for ex in dev:
            db, evid = ex["db_id"], ex.get("evidence", "")
            if retrieval:
                if (db, k) not in rets:
                    kb = merged_kb(db, k, pools[db], kbs)
                    rets[(db, k)] = (kb, SchemaRetriever(build_documents(db, kb)["column_docs"]))
                kb, ret = rets[(db, k)]
                got = ret.retrieve_columns(ex["question"] + " " + evid, top_k=15, value_matches=vmatch[ex["question_id"]])
                schema = build_retrieved_context(kb, got, top_m_tables=6)
            else:
                schema = full_schema(db, k, pools[db], ddls, args.dev_db_root)
            prompts[ex["question_id"]] = build_prompt(schema, ex["question"], evid)
        texts = {q: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for q, p in prompts.items()}
        ntok = {q: len(tok(t, add_special_tokens=False)["input_ids"]) for q, t in texts.items()}
        print(f"K={k}: prompts built | avg prompt tokens {sum(ntok.values())/len(ntok):.0f} | max {max(ntok.values())}", flush=True)

        order = sorted(texts, key=lambda q: ntok[q]); preds = {}
        for b in range(0, len(order), args.bs):
            qs = order[b:b + args.bs]
            enc = tok([texts[q] for q in qs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     eos_token_id=[151645, 151643], pad_token_id=tok.pad_token_id)
            for q, o in zip(qs, out):
                preds[q] = extract_sql(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True))
            del enc, out; torch.cuda.empty_cache()
            if (b // args.bs) % 25 == 0:
                print(f"  K={k} {min(b + args.bs, len(order))}/{len(order)}  {time.time()-t0:.0f}s", flush=True)

        examples = [{"question_id": e["question_id"], "id": e["question_id"], "difficulty": e.get("difficulty"),
                     "db_id": e["db_id"], "gold_sql": e["SQL"], "pred_sql": preds[e["question_id"]],
                     "prompt_tokens": ntok[e["question_id"]]} for e in dev]
        res = score_dataset(examples, args.dev_db_root, timeout=30, n_workers=4)
        res["K"] = k; res["system"] = args.system; res["avg_prompt_tokens"] = sum(ntok.values()) / len(ntok)
        json.dump(res, open(res_path, "w"))
        print(f"[system {args.system} | K={k}] EX {res['execution_accuracy']:.3f} | valid {res['valid_sql_rate']:.3f} | "
              f"tokens {res['avg_prompt_tokens']:.0f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
