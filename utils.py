from tqdm import tqdm 
import torch  
import clip  
import os

def save_clip_weights(clip_model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(clip_model.state_dict(), save_path)
    print(f"CLIP weights saved to {save_path}")

def load_clip_weights(clip_model, load_path, strict=True):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"File {load_path} does not exist.")
    clip_model.load_state_dict(torch.load(load_path), strict=strict)
    print(f"CLIP weights loaded from {load_path}")
    return clip_model

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred))  
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0] 
    return acc

def clip_classifier(classnames, template, clip_model, device):
    with torch.no_grad(): 
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template] 
            texts = clip.tokenize(texts).to(device) 
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) 
            class_embedding = class_embeddings.mean(dim=0) 
            class_embedding /= class_embedding.norm()  
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).to(device)  
    return clip_weights

def pre_load_features(clip_model, loader, device):
    features, labels = [], [] 
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)): 
            images, target = images.to(device), target.to(device)
            image_features = clip_model.encode_image(images) 
            image_features /= image_features.norm(dim=-1, keepdim=True)  
            features.append(image_features.cpu())  
            labels.append(target.cpu())  
        features = torch.cat(features) 
        labels = torch.cat(labels)
    return features, labels