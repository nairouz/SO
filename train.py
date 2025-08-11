import csv
import torch
from tqdm import tqdm 
from utils import *
import torch.nn.functional as F
from optimizers.SparseOptimizer import SO 

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': list(range(12))
}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': list(range(12))
    }
}

def mark_as_trainable(model, args):
    for _, param in model.named_parameters():
        param.requires_grad = False
    trainable_params = [] 
    if args.encoder in ['text', 'both']:
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = model.transformer  
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, param in block.named_parameters():
                    if "attn" in name and "weight" in name:  
                        param.requires_grad = True  
                        trainable_params.append((f"text_encoder.resblocks[{i}].{name}", tuple(param.shape)))
                    elif "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name: 
                        param.requires_grad = True  
                        trainable_params.append((f"text_encoder.resblocks[{i}].{name}", tuple(param.shape)))
    if args.encoder in ['vision', 'both']:
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = model.visual.transformer  
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, param in block.named_parameters():
                    if "attn" in name and "weight" in name: 
                        param.requires_grad = True  
                        trainable_params.append((f"vision_encoder.resblocks[{i}].{name}", tuple(param.shape)))
                    elif "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name: 
                        param.requires_grad = True  
                        trainable_params.append((f"vision_encoder.resblocks[{i}].{name}", tuple(param.shape)))
    print("\n✅ Trainable Parameters in CLIP-SO:")
    for param_name, shape in trainable_params:
        print(f"{param_name}: {shape}")

def log_training_results_to_csv(args, iteration, train_accuracy, test_accuracy, loss):
    backbone = args.backbone.replace("/", "-")
    csv_file = f"./results/results_dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_seed_{args.seed}.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Iteration", "Dataset", "Backbone", "Random Seed", "Shots", "Train Accuracy", "Test Accuracy", "Loss"])
        writer.writerow([iteration, args.dataset, args.backbone, args.seed, args.shots, train_accuracy, test_accuracy, loss])
        
def log_results_to_csv(args, final_accuracy):
    backbone = args.backbone.replace("/", "-")
    csv_file = f"./results/final_results_dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_seed_{args.seed}.txt"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Dataset", "Shots", "Final Test Accuracy"])
        writer.writerow([args.dataset, args.shots, final_accuracy])
        
def evaluate_CLIP(clip_model, loader, dataset, device):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        texts = clip.tokenize(texts).to(device)
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    acc, tot_samples = 0.0, 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    
    return acc / tot_samples

def run_clip(args, clip_model, logit_scale, dataset, train_loader, test_loader):
    VALIDATION = False
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model, args.device)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader, args.device)
    test_features, test_labels = test_features.to(args.device), test_labels.to(args.device)
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))    
    test_features, test_labels = test_features.cpu(), test_labels.cpu()
    clip_model = clip_model.to(args.device)

    if args.eval_only:
        load_path = os.path.join(args.save_path, f"{args.filename}.pt")
        load_clip_weights(clip_model, load_path)
        acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_as_trainable(clip_model, args)
    total_iters = args.n_iters * args.shots

    density_ratio = 1 - args.s 
    params = [
        {'params': [param for param in clip_model.parameters() if param.requires_grad], 'dense': False},  
    ]
    optimizer = SO(params=params, lr=args.lr, betas=(0.9, 0.999), density_ratio=density_ratio, T=args.t)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    count_iters = 0
    with tqdm(total=total_iters, desc="Training Progress", unit="iter") as pbar:
        while count_iters < total_iters:
            clip_model.train()
            train_accuracy, tot_samples, loss_epoch = 0, 0, 0.0
            text_features = textual_features.t()

            for images, target in train_loader:
                images, target = images.to(args.device), target.to(args.device)
                texts = clip.tokenize([dataset.template[0].format(c.replace('_', ' ')) for c in dataset.classnames]).to(args.device)
                class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                cosine_similarity = logit_scale * image_features @ text_features.t()
                loss = F.cross_entropy(cosine_similarity, target)
                train_accuracy += cls_acc(cosine_similarity, target) * target.shape[0]
                loss_epoch += loss.item() * target.shape[0]
                tot_samples += target.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                count_iters += 1
                pbar.update(1) 
                if count_iters == total_iters:
                    break
            train_accuracy /= tot_samples
            loss_epoch /= tot_samples
            if loss_epoch < args.early_stop_loss:
                print(f"✅ Training stopped early at iteration {count_iters} because epoch loss reached {loss_epoch:.6f}.")
                break
            if VALIDATION:
                clip_model.eval()
                acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
                print(f"Iteration {count_iters}, Train Acc: {train_accuracy:.2f}, Test Acc: {acc_test:.2f}, Loss: {loss_epoch:.4f}")
                log_training_results_to_csv(args, count_iters, train_accuracy, acc_test, loss_epoch)
    clip_model.eval()
    acc_test = evaluate_CLIP(clip_model, test_loader, dataset, args.device)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    log_results_to_csv(args, acc_test)
    if args.save_path is not None:
        save_path = os.path.join(args.save_path, f"{args.filename}.pt")
        save_clip_weights(clip_model, save_path)

    return