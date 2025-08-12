# Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation [ICCV 2024]

The official implementation of [*Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation*](https://arxiv.org/abs/2504.12436).


## 👥 Authors

- [Nairouz Mrabah](https://scholar.google.com/citations?user=pJm5B2YAAAAJ&hl=en)  
- [Nicolas Richet](https://scholar.google.com/citations?view_op=list_works&hl=fr&hl=fr&user=REJ_xkEAAAAJ)  
- [Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=en)  
- [Eric Granger](https://scholar.google.ca/citations?user=TmfbdagAAAAJ&hl=en)

## 🧠 Approach

This repo introduces Sparse Optimization (SO), a new efficient optimizer that preserves the model expressivity and mitigates overfitting. SO relies on two paradigms:

* Local sparsity and global density: dynamically update a very tiny subset of weights while the deployed model remains dense. High sparsity is enforced in both the gradient and moment updates while allowing the sparsity support to evolve dynamically throughout training.

* Local randomness and global importance: sparsify gradients via random selection, and prune the optimizer momentums by importance. The gradient captures local and iteration-specific information. The first moment aggregates the gradients over the whole path and, thus, reflects long-term parameter importance. Random pruning of the gradient prevents the model from relying too much on short-term and local high-magnitude updates. The importance-based selection of the first moment ensures that only connections with long-term significance are updated.

These choices mitigate overfitting, stabilize adaptation in low-data regimes, and reduce memory consumption. Across 11 diverse datasets, SO delivers state-of-the-art few-shot performance with lower memory overhead.

<img width="460" height="796" alt="image" src="https://github.com/user-attachments/assets/2e7ca631-40d4-4dc6-9a72-f23520e4da43" />

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

## ⚙️ How to run the code for BERT few-shot adaptation ?

Extra deps (NLP)
 ```
pip install "transformers>=4.44" "datasets>=2.20" "peft>=0.11" sentencepiece accelerate scikit-learn
 ```

The scripts auto-download GLUE-RTE via 🤗 Datasets and sample a 32-shot train split with a fixed seed.

Single run — SO (our optimizer)
 ```
# from repo root (adjust path if your BERT scripts are elsewhere)
cd BERT
python main.py \
  --device cuda \
  --kappa 5e-5 \        # fraction kept (density ratio); e.g., 5e-5 = 0.005%
  --T_sparse 10 \       # refresh interval for the sparse support
  --loss_stop 1e-3      # stop when mini-batch loss <= this value
 ```
Prints final validation accuracy: SO – BERT-base RTE-32shot: XX.X%. 

Single run — LoRA (baseline)
 ```
cd BERT
python main.py \
  --device cuda \
  --rank 8 \            # LoRA rank r (try 2, 4, 8, 16)
  --loss_stop 1e-3      # stop when mini-batch loss <= this value
 ```
Prints final validation accuracy: LoRA – BERT-base RTE-32shot: XX.X%. 

SO sweep over (kappa, T) → so_grid_results.csv
 ```
cd BERT
python script.py
 ```
Edit the list of KAPPAS and TS inside the script if you want different grids. Results are written to so_grid_results.csv. 

LoRA sweep over rank r → lora_grid_results.csv
 ```
cd BERT
python script_LoRA.py
 ```
Edit the list of RANKS inside the script to try other ranks. Results are written to lora_grid_results.csv. 


Both runners use BERT-base with a balanced 32-shot training subset of GLUE-RTE and stop when the mini-batch loss falls below --loss_stop (default 1e-3). 
In the SO runner, κ is the size of the upated fraction. For example, κ = 5e-5 updates 0.005% of entries per refresh. The mask is refreshed every T_sparse steps. 

## ✨ SO vs LoRA for BERT few-shot Adaptation 

<table>
  <caption><strong>Fine-tuning BERT (32-shot; stop when ℒ ≤ 10<sup>−3</sup>).</strong></caption>
  <thead>
    <tr>
      <th rowspan="2" align="left">Task</th>
      <th colspan="4" align="center">LoRA</th>
      <th colspan="3" align="center">SO</th>
    </tr>
    <tr>
      <th align="center">r=2</th>
      <th align="center">r=4</th>
      <th align="center">r=8</th>
      <th align="center">r=16</th>
      <th align="center">κ=0.001%</th>
      <th align="center">κ=0.01%</th>
      <th align="center">κ=0.1%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">RTE</td>
      <td align="center">54.5</td>
      <td align="center">54.2</td>
      <td align="center">54.2</td>
      <td align="center">54.5</td>
      <td align="center">57.0</td>
      <td align="center">57.0</td>
      <td align="center">55.6</td>
    </tr>
  </tbody>
</table>

## ⚙️ How to run the code for CLIP few-shot adaptation ?

## ✨ SO vs LoRA for CLIP few-shot Adaptation  

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

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

