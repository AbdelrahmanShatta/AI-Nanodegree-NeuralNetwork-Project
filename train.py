import argparse
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset")
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet34'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    return dataloaders, image_datasets

def build_model(arch, hidden_units, output_dim):
    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_dim),
        nn.LogSoftmax(dim=1)
    )
    model.fc = classifier
    
    return model

def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    model.to(device)
    steps = 0
    print_every = 40
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)
                        validation_loss += batch_loss.item()
                        
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}, Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                
                running_loss = 0
                model.train()

def save_checkpoint(model, image_datasets, save_dir, arch, hidden_units, output_dim, learning_rate,):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'output_dim': output_dim,
        'learning_rate': learning_rate,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'classifier': model.fc
    }
    torch.save(checkpoint, f'{save_dir}/checkpoint_{arch}.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif checkpoint['arch'] == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {checkpoint['arch']}")
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_units'], checkpoint['output_dim']),
        nn.LogSoftmax(dim=1)
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint['arch'], checkpoint['learning_rate']

def main():
    args = get_input_args()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    dataloaders, image_datasets = load_data(args.data_dir)
    
    output_dim = len(cat_to_name)
    model = build_model(args.arch, args.hidden_units, output_dim)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    
    save_checkpoint(model, image_datasets, args.save_dir, args.arch, args.hidden_units, output_dim, args.learning_rate)
    
    print(f"Model trained and saved to {args.save_dir}/checkpoint_{args.arch}.pth")

if __name__ == "__main__":
    main()