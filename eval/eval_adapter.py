"""
Score a model (base, or base + LoRA adapter) on the held-out validation questions.
Same prompts as training (read from val.jsonl), same extract_sql, same harness as Systems A/C.

  base only:      python eval_adapter.py --data D --n 300 --tag base
  with adapter:   python eval_adapter.py --data D --n 300 --adapter epoch-1 --hub_name text2sql-lora-b --tag b_epoch1
"""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.prompts import extract_sql
from harness import score_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--adapter", default=None, help="folder inside the hub repo, e.g. epoch-1 or latest")
ap.add_argument("--hub_name", default="text2sql-lora-b")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/kaggle/working")
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--max_new", type=int, default=400)
args = ap.parse_args()

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    token = None

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token); tok.padding_side = "left"
# fp16, unquantized: the same loading as System A, so the only change is the adapter
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map={"": 0}, token=token)
if args.adapter:
    from peft import PeftModel
    from huggingface_hub import HfApi, snapshot_download
    repo = f"{HfApi(token=token).whoami()['name']}/{args.hub_name}"
    d = snapshot_download(repo, allow_patterns=f"{args.adapter}/*", token=token)
    model = PeftModel.from_pretrained(model, os.path.join(d, args.adapter))
model.eval()

rows = [json.loads(l) for l in open(f"{args.data}/val.jsonl")]
rows = rows[::max(1, len(rows) // args.n)][:args.n]
texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                 add_generation_prompt=True) for r in rows]
order = sorted(range(len(rows)), key=lambda i: len(texts[i]))      # similar lengths together = less padding
preds, t0 = {}, time.time()
for b in range(0, len(order), args.bs):
    ids = order[b:b + args.bs]
    enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                             eos_token_id=[151645, 151643], pad_token_id=tok.pad_token_id)
    for i, o in zip(ids, out):
        preds[i] = extract_sql(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True))
    print(f"{min(b + args.bs, len(order))}/{len(order)}  {time.time()-t0:.0f}s", flush=True)

examples = [{"id": r["id"], "db_id": r["db_id"], "gold_sql": r["completion"], "pred_sql": preds[i]}
            for i, r in enumerate(rows)]
res = score_dataset(examples, f"{args.data}/val_databases", timeout=30, n_workers=4)
print(f"\n[{args.tag}] EX {res['execution_accuracy']:.3f} | valid {res['valid_sql_rate']:.3f} | "
      f"scored {res['n_scored']} (skipped {res['n_skipped_bad_gold']} bad gold)")
json.dump({k: v for k, v in res.items()}, open(f"{args.out}/val_{args.tag}.json", "w"))
