"""
Resumable QLoRA fine-tuning of Qwen2.5-Coder-3B-Instruct for text-to-SQL.
Used for System B (full-schema prompts) and, later, System D (retrieved-context prompts):
the only difference between the two runs is the --data folder.

Design choices (each one was forced by something we measured):
  * Plain training loop, not HF Trainer: the Trainer path ran out of GPU memory on a T4.
  * Loss only on the SQL answer, and the LM head runs only on answer positions
    (logits_to_keep) -- the prompt is ~95% of the tokens and needs no logits.
  * Checkpoints (adapter + optimizer + step) are pushed to a private HF Hub repo, so a
    dead Kaggle session never costs more than --save_every steps.
  * Fixed shuffle seed per epoch + global LR schedule, so a resumed run is equivalent
    to an uninterrupted one.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse, json, math, random, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file
from huggingface_hub import HfApi, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder containing train.jsonl and val.jsonl")
ap.add_argument("--out", default="/kaggle/working/run")
ap.add_argument("--hub_name", required=True, help="private HF repo name for checkpoints")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max_steps", type=int, default=0, help="stop after this GLOBAL step (0 = run all)")
ap.add_argument("--save_every", type=int, default=100)
ap.add_argument("--eval_every", type=int, default=100)
ap.add_argument("--val_n", type=int, default=96, help="validation examples for the quick val loss")
ap.add_argument("--max_hours", type=float, default=10.5, help="save + stop gracefully after this long")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()

# ---------- Hugging Face ----------
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    token = None            # fall back to a cached login
api = HfApi(token=token)
repo_id = f"{api.whoami()['name']}/{args.hub_name}"
api.create_repo(repo_id, private=True, exist_ok=True)
print("checkpoints ->", repo_id, flush=True)

# ---------- data ----------
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)

def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def encode(row):
    # identical chat formatting to inference; labels = -100 on the prompt
    text = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                   tokenize=False, add_generation_prompt=True)
    p = tok(text, add_special_tokens=False)["input_ids"]
    c = tok(row["completion"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {"input_ids": p + c, "attention_mask": [1] * (len(p) + len(c)), "labels": [-100] * len(p) + c}

train = [encode(r) for r in load(f"{args.data}/train.jsonl")]
val_rows = load(f"{args.data}/val.jsonl")
val = [encode(r) for r in val_rows[::max(1, len(val_rows) // args.val_n)][:args.val_n]]
print(f"train {len(train)} | val-loss subset {len(val)} | load time {time.time()-T0:.0f}s", flush=True)

# ---------- model ----------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                             device_map={"": 0}, dtype=torch.float16, token=token)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in model.parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
model.print_trainable_parameters()

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")

steps_per_epoch = len(train) // args.accum
total_steps = steps_per_epoch * args.epochs
warmup = max(1, int(0.03 * total_steps))
stop_at = min(total_steps, args.max_steps) if args.max_steps else total_steps

def lr_scale(s):
    if s < warmup:
        return (s + 1) / warmup
    prog = (s - warmup) / max(1, total_steps - warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

def epoch_order(ep):
    idx = list(range(len(train)))
    random.Random(args.seed + ep).shuffle(idx)
    return idx

# ---------- checkpoint helpers ----------
def save(tag, step, full):
    d = f"{args.out}/{tag}"
    os.makedirs(d, exist_ok=True)
    try:
        model.save_pretrained(d)
        if full:
            torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict()}, f"{d}/train_state.pt")
            json.dump({"step": step}, open(f"{d}/state.json", "w"))
        api.upload_folder(folder_path=d, repo_id=repo_id, path_in_repo=tag,
                          commit_message=f"{tag} @ step {step}")
        print(f"   saved '{tag}' at step {step} -> hub", flush=True)
    except Exception as ex:                       # never let a failed upload kill a long run
        print(f"   !! saving/uploading '{tag}' failed (training continues): {ex}", flush=True)

start = 0
if args.resume:
    d = os.path.join(snapshot_download(repo_id, allow_patterns="latest/*", token=token), "latest")
    set_peft_model_state_dict(model, load_file(f"{d}/adapter_model.safetensors"))
    st = torch.load(f"{d}/train_state.pt", map_location="cuda:0")
    opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
    start = json.load(open(f"{d}/state.json"))["step"]
    print(f"RESUMED from step {start}", flush=True)

@torch.no_grad()
def val_loss():
    model.eval()
    tot, n = 0.0, 0
    for e in val:
        ids = torch.tensor([e["input_ids"]], device="cuda:0")
        att = torch.tensor([e["attention_mask"]], device="cuda:0")
        lab = torch.tensor([e["labels"]], device="cuda:0")
        s = int((lab[0] != -100).nonzero()[0])
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=att, logits_to_keep=lab.shape[1] - s + 1)
        lg = out.logits[:, :-1, :].float()
        tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), lab[:, s:].reshape(-1), reduction="sum").item()
        n += lab.shape[1] - s
    model.train()
    return tot / n

# ---------- training ----------
print(f"steps/epoch {steps_per_epoch} | total {total_steps} | warmup {warmup} | running {start} -> {stop_at}", flush=True)
model.train()
torch.cuda.reset_peak_memory_stats(0)
log = open(f"{args.out}/train_log.jsonl", "a")
order_ep, order = -1, None
done = start

for g in range(start, stop_at):
    t_step = time.time()
    ep, k = divmod(g, steps_per_epoch)
    if ep != order_ep:
        order, order_ep = epoch_order(ep), ep
    batch = [train[i] for i in order[k * args.accum:(k + 1) * args.accum]]
    for grp in opt.param_groups:
        grp["lr"] = args.lr * lr_scale(g)

    n_ans = sum(sum(1 for x in e["labels"] if x != -100) for e in batch)
    step_loss, step_tokens, longest = 0.0, 0, 0
    for e in batch:
        ids = torch.tensor([e["input_ids"]], device="cuda:0")
        att = torch.tensor([e["attention_mask"]], device="cuda:0")
        lab = torch.tensor([e["labels"]], device="cuda:0")
        s = int((lab[0] != -100).nonzero()[0])
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=att, logits_to_keep=lab.shape[1] - s + 1)
        logits = out.logits[:, :-1, :].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, s:].reshape(-1),
                               reduction="sum") / n_ans
        scaler.scale(loss).backward()
        step_loss += loss.item(); step_tokens += ids.shape[1]; longest = max(longest, ids.shape[1])
        del out, logits, loss, ids, att, lab

    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    done = g + 1

    peak = torch.cuda.max_memory_allocated(0) / 1e9
    torch.cuda.reset_peak_memory_stats(0)
    rec = {"step": done, "epoch": round(done / steps_per_epoch, 3), "loss": round(step_loss, 4),
           "gnorm": round(gnorm, 3), "scale": scaler.get_scale(), "lr": args.lr * lr_scale(g),
           "tokens": step_tokens, "peak_gb": round(peak, 1), "sec": round(time.time() - t_step, 1)}
    if done % args.eval_every == 0 or done == stop_at:
        rec["val_loss"] = round(val_loss(), 4)
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(" | ".join(f"{k_} {v}" for k_, v in rec.items()), flush=True)

    if done % args.save_every == 0:
        save("latest", done, full=True)
    if done % steps_per_epoch == 0:
        save(f"epoch-{done // steps_per_epoch}", done, full=False)
    if (time.time() - T0) / 3600 > args.max_hours:
        print(f"time limit ({args.max_hours} h) reached -- saving and stopping; re-run with --resume", flush=True)
        break

save("latest", done, full=True)
print(f"finished at step {done}/{total_steps}; elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
