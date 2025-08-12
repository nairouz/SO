#!/usr/bin/env python
# fair_so_rte32_lossstop.py
"""
Sparse Optimizer (SO) on BERT-base for GLUE-RTE (32-shot)

• Same seed / data / LR / loss-based stop as the LoRA script
• Encoder matrices updated sparsely   (κ , T)
• All 1-D parameters + pooler + classifier updated densely
• Live progress bar with tqdm
"""

import random, argparse, torch, numpy as np
from datasets            import load_dataset
from transformers        import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data    import DataLoader
from SparseOptimizer     import SO
from tqdm                import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta`")
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma`")

# ---------- helpers -------------------------------------------------
def set_seed(seed=42):
    random.seed(seed);  np.random.seed(seed)
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
# --------------------------------------------------------------------

def main(args):
    set_seed(42)
    dev = torch.device(args.device)

    # ---------------- data ------------------------------------------
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    ds  = load_dataset("glue", "rte")
    train = (ds["train"].shuffle(seed=42).select(range(32)).map(lambda e: preprocess(e, tok), batched=True))
    valid = ds["validation"].map(lambda e: preprocess(e, tok), batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    tr_loader = DataLoader(train.with_format("torch", columns=cols), batch_size=8, shuffle=True)
    va_loader = DataLoader(valid.with_format("torch", columns=cols), batch_size=32)

    # ---------------- model ----------------------------------------
    model = (BertForSequenceClassification.from_pretrained("bert-base-uncased").to(dev))

    # -------- parameter groups: sparse vs. dense -------------------
    sparse_params, dense_params = [], []
    for n, p in model.named_parameters():
        # keep pooler / classifier / ALL bias & LayerNorm (1-D) dense
        if p.dim() == 1 or n.startswith(("classifier.", "bert.pooler.")):
            dense_params.append(p)
        else:
            sparse_params.append(p)
    param_groups = [
        {"params": sparse_params, "dense": False},  # encoder matrices
        {"params": dense_params,  "dense": True}    # rest
    ]

    # ---------------- optimiser & scheduler ------------------------
    opt = SO(param_groups, lr=2e-5, betas=(0.9, 0.999), density_ratio=args.kappa, T=args.T_sparse)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10_000, eta_min=1e-6) 

    # ---------------- training loop --------------------------------
    model.train()
    step      = 0
    last_loss = float("inf")
    pbar      = tqdm(unit="step", desc="SO fine-tune")

    while last_loss > args.loss_stop:
        for batch in tr_loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss  = model(**batch).loss
            opt.zero_grad();  loss.backward()
            opt.step();       sched.step()

            last_loss = loss.item()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{last_loss:.4f}")

            if last_loss <= args.loss_stop:
                break
    pbar.close()
    print(f"Stopped at step {step} with loss {last_loss:.4f} ≤ {args.loss_stop}")

    # ---------------- evaluation -----------------------------------
    acc = accuracy(model, va_loader, dev)
    print(f"SO – BERT-base RTE-32shot: {acc:.1f}%")

# ---------------- CLI ----------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--kappa",     type=float, default=0.0001, help="sparsity ratio κ (e.g. 0.001 = 0.1 %)")
    ap.add_argument("--T_sparse",  type=int,   default=5, help="support refresh interval T")
    ap.add_argument("--loss_stop", type=float, default=1e-3, help="stop when mini-batch loss < this value")
    main(ap.parse_args())
