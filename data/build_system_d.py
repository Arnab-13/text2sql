"""
Build the System D training data: the SAME questions, order and evidence-dropout as System B
(read from splits.json), but with the schema-aware RETRIEVED context in place of the full schema.
Retrieval is identical to System C at inference: top-15 columns from hybrid + value matching,
FK-bridge expansion, top-6 tables. Gold SQL is never used to build the context, so training
contexts carry the same retrieval mistakes the model will meet at test time.

CPU only. One worker per database; each finished database is saved as a shard, so a crash or a
time-out never costs more than the databases in flight (re-run the same command to resume).

  probe:  python build_system_d.py ... --limit_per_db 5 --max_dbs 12 --out /kaggle/working/probe
  full:   python build_system_d.py ... --out /kaggle/working/system_d_data
"""
import os, sys, json, time, argparse
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def work(job):
    """Runs in a worker process: build the KB + retriever for one DB, then every question of that DB."""
    db, items, a = job
    shard = f"{a['out']}/shards/{db}.json"
    if os.path.exists(shard):
        return db, len(items), 0.0, 0.0, 0
    sys.path.insert(0, os.path.join(ROOT, "retriever"))
    try:
        import torch; torch.set_num_threads(1)      # one thread per worker, so 4 workers = 4 cores
    except ImportError:
        pass
    from schema_kb import build_schema_kb, find_value_matches
    from doc_builder import build_documents
    from retriever import SchemaRetriever
    from context_builder import build_retrieved_context
    t0 = time.time()
    kb = build_schema_kb(a["db_root"], db)
    ret = SchemaRetriever(build_documents(db, kb)["column_docs"])
    kb_s = time.time() - t0
    db_path = f"{a['db_root']}/{db}/{db}.sqlite"
    out, failed, t1 = {}, 0, time.time()
    for qid, question, ev in items:
        try:
            vm = find_value_matches(db_path, question)
            got = ret.retrieve_columns(question + " " + ev, top_k=15, value_matches=vm)
            out[qid] = build_retrieved_context(kb, got, top_m_tables=6)
        except Exception as e:
            out[qid] = None; failed += 1
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    json.dump(out, open(shard, "w"))
    return db, len(items), kb_s, (time.time() - t1) / max(1, len(items)), failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--db_root", required=True, help="folder holding {db_id}/{db_id}.sqlite for the TRAIN databases")
    ap.add_argument("--splits", required=True, help="splits.json from the System B data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit_per_db", type=int, default=0, help="probe mode: only this many questions per DB")
    ap.add_argument("--max_dbs", type=int, default=0, help="probe mode: only this many DBs")
    args = ap.parse_args()

    train = json.load(open(args.train_json))
    splits = json.load(open(args.splits))
    dropped = set(splits["evidence_dropped"])
    by_db = defaultdict(list)
    for qid in splits["train"] + splits["val"]:
        ex = train[qid]
        by_db[ex["db_id"]].append((qid, ex["question"], "" if qid in dropped else ex.get("evidence", "")))
    jobs = sorted(by_db.items(), key=lambda kv: -len(kv[1]))          # biggest first = better load balance
    probe = bool(args.limit_per_db or args.max_dbs)
    if args.max_dbs:
        jobs = jobs[::max(1, len(jobs) // args.max_dbs)][:args.max_dbs]
    if args.limit_per_db:
        jobs = [(d, it[:args.limit_per_db]) for d, it in jobs]
    a = {"db_root": args.db_root, "out": args.out}
    print(f"{len(jobs)} databases, {sum(len(i) for _, i in jobs)} questions, {args.workers} workers", flush=True)

    import multiprocessing as mp
    done_q, total_q, t0, failed_total = 0, sum(len(i) for _, i in jobs), time.time(), 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for db, n, kb_s, q_s, failed in pool.imap_unordered(work, [(d, it, a) for d, it in jobs]):
            done_q += n; failed_total += failed
            print(f"[{done_q}/{total_q}] {db}: {n} q | KB {kb_s:.0f}s | {q_s:.2f}s/question | failed {failed} "
                  f"| elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    if probe:
        print("probe finished -- nothing merged. Multiply s/question by the question counts to estimate the full run.")
        return

    # ---- merge shards into train.jsonl / val.jsonl, in the same order as System B ----
    from common.prompts import build_prompt
    ctxs = {}
    for fn in os.listdir(f"{args.out}/shards"):
        ctxs.update({int(k): v for k, v in json.load(open(f"{args.out}/shards/{fn}")).items()})
    stats = {}
    for split in ("train", "val"):
        rows, failed = [], 0
        for qid in splits[split]:
            ctx = ctxs.get(qid)
            if ctx is None:
                failed += 1; continue
            ex = train[qid]
            ev = "" if qid in dropped else ex.get("evidence", "")
            rows.append({"id": qid, "db_id": ex["db_id"], "prompt": build_prompt(ctx, ex["question"], ev),
                         "completion": ex["SQL"]})
        with open(f"{args.out}/{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stats[split] = {"rows": len(rows), "failed_dropped": failed}
    json.dump(splits, open(f"{args.out}/splits.json", "w"))
    json.dump(stats, open(f"{args.out}/meta.json", "w"), indent=2)
    print("merged:", stats)
    try:
        import numpy as np
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        L = np.array([len(tok(json.loads(l)["prompt"] + json.loads(l)["completion"])["input_ids"])
                      for l in open(f"{args.out}/train.jsonl")])
        print(f"tokens per training example: mean {L.mean():.0f} | p90 {int(np.percentile(L, 90))} | max {L.max()}")
    except Exception as e:
        print("token stats skipped:", e)


if __name__ == "__main__":
    main()
