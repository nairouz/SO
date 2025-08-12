# Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation [ICCV 2024]

The official implementation of [*Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation*](https://arxiv.org/abs/2504.12436).


## 👥 Authors
- [Nairouz Mrabah](https://scholar.google.com/citations?user=pJm5B2YAAAAJ&hl=en)  
- [Nicolas Richet](https://scholar.google.com/citations?view_op=list_works&hl=fr&hl=fr&user=REJ_xkEAAAAJ)  
- [Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=en)  
- [Éric Granger](https://scholar.google.ca/citations?user=TmfbdagAAAAJ&hl=en)


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
The requirements.txt is intentionally small and only contains what the code actually uses (PyTorch/vision, numpy/scipy, Pillow, tqdm, ftfy, regex, gdown).
```

## Citation

If you find this project useful, please cite it as follows:

```bibtex
@inproceedings{mrabah2025so,
  title     = {Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation},
  author    = {Mrabah, Nairouz and Richet, Nicolas and Ben Ayed, Ismail and Granger, Eric},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

