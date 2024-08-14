import argparse
import json
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('input', type=str,help='Path to the input image')
    parser.add_argument('checkpoint', type=str,help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=2, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = checkpoint['classifier']
    elif checkpoint['arch'] == 'resnet34':
        model = models.vgg19(pretrained=True)
        model.fc = checkpoint['classifier']
    else:
        raise ValueError(f"Unsupported architecture: {checkpoint['arch']}")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    img = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    return img_tensor

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    img = process_image(image_path)
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        probs = torch.exp(output)
        top_probs, top_indices = probs.topk(topk)
    
    # Ensure they are 1D arrays, even for topk = 1
    top_probs = top_probs.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()

    # Convert to 1D lists if not already
    if topk == 1:
        top_probs = [top_probs]
        top_indices = [top_indices]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes


def main():
    args = get_input_args()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(args.checkpoint)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    top_probs, top_classes = predict(args.input, model, args.top_k, device)
    
    top_flowers = [cat_to_name[cls] for cls in top_classes]
    
    print("\nPredictions:")
    for i in range(len(top_flowers)):
        print(f"{top_flowers[i]}: {top_probs[i]:.3f}")

if __name__ == "__main__":
    main()