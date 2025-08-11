import random
import argparse  
import numpy as np 
import torch
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=1, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Location of trainable parameters
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=2000, type=int)
    parser.add_argument('--early_stop_loss', default=0.01, type=float, help="Stop training early when epoch loss falls below this threshold.")
    parser.add_argument('--batch_size', default=32, type=int)
    # Optimizer arguments
    parser.add_argument('--s', default=0.99995, type=float, help='the sparsity ratio')
    parser.add_argument('--t', default=5, type=int, help='number of iterations before updating the sparsity support')
    # Saving arguments
    parser.add_argument('--save_path', default=None, help='path to save the model after training, not saved if None')
    parser.add_argument('--filename', default='CLIP_weights', help='file name to save the model weights (.pt extension will be added)')
    # Evaluation arguments
    parser.add_argument('--eval_only', default=False, action='store_true', help='evaluate the model after the loading the weights')
    args = parser.parse_args()

    return args