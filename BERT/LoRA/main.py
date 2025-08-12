#!/usr/bin/env python
# lora_rte32_lossstop_bar.py
"""
LoRA on BERT-base for GLUE-RTE (32-shot)

• Rank-r adapters on every Linear weight in all 12 encoder blocks
• Pooler dense + classifier kept full-rank & trainable
• **All 1-D tensors (biases, LayerNorm weights) are also trainable**
• Adam with cosine LR; stop when mini-batch loss < --loss_stop
• Live tqdm progress bar
"""

import random, argparse, warnings, numpy as np, torch
from datasets         import load_dataset
from transformers     import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader
from peft             import LoraConfig, get_peft_model, TaskType
from tqdm             import tqdm

# ─── silence cosmetic warnings ─────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)
warnings.filterwarnings("ignore", message="A parameter name that contains `beta`")
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma`")

# ─── helpers ───────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def preprocess(ex, tok):
    t = tok(ex["sentence1"], ex["sentence2"], truncation=True, padding="max_length", max_length=128)
    t["labels"] = ex["label"]
    return t

def accuracy(model, loader, dev):
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            ok  += (model(**b).logits.argmax(-1) == b["labels"]).sum().item()
            tot += b["labels"].size(0)
    return 100 * ok / tot
# ───────────────────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    dev = torch.device(args.device)

    # ─── data (32-shot) ────────────────────────────────────────────
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    ds  = load_dataset("glue", "rte")
    train = (ds["train"].shuffle(seed=42).select(range(32)).map(lambda e: preprocess(e, tok), batched=True))
    valid = ds["validation"].map(lambda e: preprocess(e, tok), batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    tr_loader = DataLoader(train.with_format("torch", columns=cols), batch_size=8, shuffle=True)
    va_loader = DataLoader(valid.with_format("torch", columns=cols), batch_size=32)

    # ─── model + LoRA ──────────────────────────────────────────────
    base = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(dev)
    cfg  = LoraConfig(
        task_type      = TaskType.SEQ_CLS,
        r              = args.rank,
        lora_alpha     = 1,
        lora_dropout   = 0.0,
        bias           = "none",   # we manually handle biases below
        target_modules = ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"],
        modules_to_save = ["classifier", "bert.pooler.dense"],
    )
    model = get_peft_model(base, cfg).to(dev)

    # ▸ unfreeze every 1-D parameter (biases & LayerNorm γ/β)
    for p in model.parameters():
        if p.dim() == 1:
            p.requires_grad_(True)

    # ─── optimiser & scheduler ─────────────────────────────────────
    opt   = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10_000, eta_min=1e-6)

    # ─── training loop with loss-based early stop ──────────────────
    model.train()
    step = 0; last_loss = float("inf")
    pbar = tqdm(unit="step", desc="LoRA fine-tune")

    while last_loss > args.loss_stop:
        for batch in tr_loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss  = model(**batch).loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            last_loss = loss.item(); step += 1
            pbar.update(1); pbar.set_postfix(loss=f"{last_loss:.4f}")

            if last_loss <= args.loss_stop:
                break
    pbar.close()
    print(f"Stopped at step {step} with loss {last_loss:.4f} ≤ {args.loss_stop}")

    # ─── evaluation ────────────────────────────────────────────────
    acc = accuracy(model, va_loader, dev)
    print(f"LoRA (encoder adapters + 1-D dense) – BERT-base RTE-32shot: {acc:.1f}%")

# ─── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--rank",      type=int, default=8,   help="LoRA rank r")
    ap.add_argument("--loss_stop", type=float, default=1e-3, help="stop when mini-batch loss < this value")
    main(ap.parse_args())
