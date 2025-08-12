# Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation [ICCV 2024]

The official implementation of [*Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation*](https://arxiv.org/abs/2504.12436).


## 👥 Authors
- [Nairouz Mrabah](https://scholar.google.com/citations?user=pJm5B2YAAAAJ&hl=en)  
- [Nicolas Richet](https://scholar.google.com/citations?view_op=list_works&hl=fr&hl=fr&user=REJ_xkEAAAAJ)  
- [Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=en)  
- [Éric Granger](https://scholar.google.ca/citations?user=TmfbdagAAAAJ&hl=en)

## 🎨 Approach



## 📁 Project structure

```
SO/
├── clip/                # CLIP code (local copy used by this repo)
├── datasets/            # dataset loaders (imagenet, sun397, fgvc, eurosat, ...)
├── optimizers/          # SO optimizer implementation 
├── main.py              # entry point (training / eval harness)
├── train.py             # training utilities
├── run_utils.py         # argument parsing, logging, helpers
├── utils.py             # misc utilities
└── requirements.txt     # dependencies for this repo
```

## 🗄️ Datasets

We follow the same dataset processing and organization as several previous VLM few-shot adaptation methods. Please place all datasets under a single root (e.g., $DATA) and follow the [DATASETS.md](DATASETS.md) to install the datasets. All instructions for download and folder layout are provided. Then pass --root_path $DATA at run time. 

Benchmarks used (11 total): ImageNet‑1k, SUN397, FGVC‑Aircraft, EuroSAT, Stanford Cars, Food‑101, Oxford‑IIIT Pets, Flowers‑102, Caltech‑101, DTD, UCF101. 

The dataset argument values in this repo match the loaders in datasets/ (e.g., imagenet, sun397, fgvc, eurosat, stanford_cars, food101, oxford_pets, oxford_flowers, caltech101, dtd, ucf101).

## 🛠️ Installation
Requires Python ≥3.10 and PyTorch. Create a fresh env, install PyTorch that matches your CUDA, then install the repo requirements.

```
# fresh environment
conda create -y -n so python=3.11
conda activate so

# install PyTorch first (choose the right CUDA build from pytorch.org)
# example for CUDA 12.4 wheels:
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 -f https://download.pytorch.org/whl/cu124

# now install project deps
pip install -r requirements.txt
```

## ⚙️ How to use SO ?

SO is a simple optimizer that can be used to train any model efficiently. The optimizer lives in ``SparseOptimizer.py`` as class ``SO``.

```
import torch
from SparseOptimizer import SO

model = ...  # any nn.Module
model.train()

optimizer = SO(
    model.parameters(),     # like Adam
    lr=3e-4,                # like Adam
    betas=(0.9, 0.999),     # like Adam
    eps=1e-8,               # like Adam
    weight_decay=0.0,       # like Adam
    density_ratio=5e-4,     # density_ratio = 1 - κ and κ represents how sparse the gradients/moments are
    T=10                    # refresh interval for the sparse support
)

for images, labels in dataloader:
    logits = model(images)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 ```

Arguments you will likely tune:
* density_ratio (1 - κ): fraction of entries kept (e.g., 5e-5 to 1e-3 for very sparse fine-tuning).
* T: refresh interval (e.g., 10).
* lr, betas, weight_decay: like Adam, defaults work well.
   
## ⚙️ How to run the code for VLM few-shot adaptation ?

## 📚 Citation

If you find this project useful, please cite it as follows:

```bibtex
@inproceedings{mrabah2025so,
  title     = {Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation},
  author    = {Mrabah, Nairouz and Richet, Nicolas and Ben Ayed, Ismail and Granger, Eric},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

