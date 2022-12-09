#Functions for model training, classes and testing
import torch
from torch import nn
from typing import Any, cast, Dict, List, Optional, Union
from utils.rep3d import *
import numpy as np
import pickle
import sys



#Check GPU access
def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = 'cuda'
    return device

#Datasets
#Create training and testing loaders
class VoxelDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 voxel_paths, 
                 labels):
        self.voxel_paths = voxel_paths
        self.labels = labels
        assert len(self.voxel_paths) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        voxel_path = self.voxel_paths[idx]
        label = self.labels[idx]
        voxel = torch.load(voxel_path)[0]
        return voxel, label

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 sequences, 
                 labels,
                 tokenizer,
                 max_len = 2000):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        assert len(self.sequences) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        token = self.tokenizer(sequence, 
                                return_tensors="pt", 
                               max_length = self.max_len - 1, 
                               padding = "max_length", 
                               truncation=True)
        return token, label

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 voxel_paths, 
                 encoding_paths,
                 labels):
        self.voxel_paths = voxel_paths
        self.labels = labels
        self.encoding_paths = encoding_paths
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        voxel_path = self.voxel_paths[idx]
        voxel = torch.load(voxel_path)[0]
        label = self.labels[idx]
        encoding_path = self.encoding_paths[idx]
        encoding = torch.load(encoding_path)[0]
        
        return voxel, encoding, label
        
#3D Implementation of Alexnet
class alex_3d(nn.Module):
    def __init__(self, in_channels: int = 5, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


#3D Implementation of VGG16
class VGG16_3D(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            num_classes: int = 1000, 
            init_weights: bool = True, 
            dropout: float = 0.5
        ) -> None:
        
        super().__init__()
        self.in_channels = in_channels
        #self.config = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        self.config = [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128, "M"]
        self.features = self.make_layers(self.config, self.in_channels)
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
    def make_layers(self, cfg, in_channels, batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
#3D Implementation of VGG16
class FusionModel(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            num_classes: int = 1000, 
            init_weights: bool = True, 
            dropout: float = 0.5
        ) -> None:
        
        super().__init__()
        self.in_channels = in_channels
        #self.config = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        self.config = [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128, "M"]
        self.features = self.make_layers(self.config, self.in_channels)
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(44544, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, voxel, encoding: torch.Tensor) -> torch.Tensor:
        voxel = self.features(voxel)
        voxel = self.avgpool(voxel)
        voxel = torch.flatten(voxel, 1)
        x = torch.cat((encoding, voxel), dim = 1)
        x = self.classifier(x)
        return x
    
    
    def make_layers(self, cfg, in_channels, batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    
    
#Train torch model
def train_model(model, 
                epochs, 
                save_path, 
                trainloader,
                testloader,
                optimizer,
                loss_fn,
                device,
                esm=True):
    
    os.makedirs(save_path, exist_ok = True)

    #Train iteration:
    for epoch in range(epochs):
        
        #Save for epoch dict
        state_dict_path = os.path.join(save_path, f'{epoch}_state_dict.pt')
        epoch_results_path  = os.path.join(save_path, f'{epoch}_results.pt')
        
        #Load if epoch already trained
        if os.path.isfile(state_dict_path):
            model.load_state_dict(torch.load(state_dict_path))
            continue
            
        #Save training results:
        training_losses = []
        training_predictions = []
        training_labels = []

        print(f'Epoch: {epoch}')

        #Train model on train set
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            
            #Predict voxel
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            if esm:
                for k, v in inputs.items():
                    inputs[k] = torch.permute(v, (1, 0, 2))
                    inputs[k] = inputs[k][0]
                outputs = model(**inputs).logits
            else:
                outputs = model(inputs)
            
            del inputs

            # Calculate softmax and cross entropy loss
            loss = loss_fn(outputs, labels)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            
            
            #Print batch results
            print('\t', f'Batch {i}', f'Average loss: {float(loss.data.cpu())}')
                  
            #Add to data
            training_losses.append(float(loss.data.cpu()))
            training_labels.extend(labels.data.cpu().tolist())
            _, predictions = torch.max(outputs, 1)
            training_predictions.extend(predictions.tolist())

            sys.stdout.flush()
            
        #Evaluate epoch on test data
        
        #Save test results:
        test_losses = []
        test_predictions = []
        test_labels = []
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if esm:
                        for k, v in inputs.items():
                            inputs[k] = torch.permute(v, (1, 0, 2))
                            inputs[k] = inputs[k][0]
                        outputs = model(**inputs).logits
                    else:
                        outputs = model(inputs)

                    #Calculate loss
                    loss = loss_fn(outputs, labels)

                    #Add to lists
                    test_losses.append(float(loss.data.cpu()))
                    test_labels.extend(labels.data.cpu().tolist())
                    _, predictions = torch.max(outputs, 1)
                    test_predictions.extend(predictions.tolist())
                
            
            
        #Save model weights
        torch.save(model.state_dict(), state_dict_path)
        
        #Save epoch results
        epoch_results = {
            'training_losses':training_losses,
            'training_labels':training_labels,
            'training_predictions':training_predictions,
            'test_losses':test_losses,
            'test_labels':test_labels,
            'test_predictions':test_predictions,
        }
        torch.save(epoch_results, epoch_results_path)
        
    return model

#Train torch model
def train_fusion_model(model, 
                epochs, 
                save_path, 
                trainloader,
                testloader,
                optimizer,
                loss_fn,
                device,
                esm=True):
    
    os.makedirs(save_path, exist_ok = True)

    #Train iteration:
    for epoch in range(epochs):
        
        #Save for epoch dict
        state_dict_path = os.path.join(save_path, f'{epoch}_state_dict.pt')
        epoch_results_path  = os.path.join(save_path, f'{epoch}_results.pt')
        
        #Load if epoch already trained
        if os.path.isfile(state_dict_path):
            model.load_state_dict(torch.load(state_dict_path))
            continue
            
        #Save training results:
        training_losses = []
        training_predictions = []
        training_labels = []

        print(f'Epoch: {epoch}')

        #Train model on train set
        model.train()
        for i, (voxels, encodings, labels) in enumerate(trainloader):
            
            #Predict voxel
            voxels = voxels.to(device)
            encodings = encodings.to(device)
            labels = labels.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(voxels, encodings)
            
            del encodings
            del voxels

            # Calculate softmax and cross entropy loss
            loss = loss_fn(outputs, labels)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            
            
            #Print batch results
            print('\t', f'Batch {i}', f'Average loss: {float(loss.data.cpu())}')
                  
            #Add to data
            training_losses.append(float(loss.data.cpu()))
            training_labels.extend(labels.data.cpu().tolist())
            _, predictions = torch.max(outputs, 1)
            training_predictions.extend(predictions.tolist())
            
        #Evaluate epoch on test data
        
        #Save test results:
        test_losses = []
        test_predictions = []
        test_labels = []
        
        model.eval()
        with torch.no_grad():
            for voxels, encodings, labels in testloader:
                    voxels = voxels.to(device)
                    encodings = encodings.to(device)
                    labels = labels.to(device)
            
                    outputs = model(voxels, encodings)

                    #Calculate loss
                    loss = loss_fn(outputs, labels)

                    #Add to lists
                    test_losses.append(float(loss.data.cpu()))
                    test_labels.extend(labels.data.cpu().tolist())
                    _, predictions = torch.max(outputs, 1)
                    test_predictions.extend(predictions.tolist())
                
            
        #Save model weights
        torch.save(model.state_dict(), state_dict_path)
        
        #Save epoch results
        epoch_results = {
            'training_losses':training_losses,
            'training_labels':training_labels,
            'training_predictions':training_predictions,
            'test_losses':test_losses,
            'test_labels':test_labels,
            'test_predictions':test_predictions,
        }
        torch.save(epoch_results, epoch_results_path)
        
    return model

def eval_model(loader, net, device):
    #Get train set predictions
    true_labels = []
    predictions = []
    losses = []
    for i, (voxels, labels) in enumerate(loader):
        
        #Predict voxel
        voxels = voxels.to(device)
        labels = labels.to(device)
        outputs = net(voxels)
        true_labels.append(int(labels.data.cpu()))

        _, predicted = torch.max(outputs.data, 1)
        predictions.append(int(predicted.data.cpu()))
        loss = error(outputs, labels)
        losses.append(float(loss.data.cpu()))
        
           
    return true_labels, predictions, losses