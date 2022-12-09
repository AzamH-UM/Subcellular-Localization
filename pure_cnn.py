import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import Bio.PDB
from glob import glob
from tqdm import tqdm
import cv2 
from torchvision import models 
from torchvision.models import ResNet34_Weights

with open("./Data/Data.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)
classes = [int(i[1]) for i in data if i[2] == "train"]

CLASS_WEIGHTS = compute_class_weight("balanced", classes=np.unique(classes), y=classes)
CLASS_WEIGHTS = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float32)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

class protein_sequence_ds(Dataset):
    def __init__(self, csv_path, split, pdb_path):
        self.csv_path = csv_path
        self.pdb_path = pdb_path
        with open(self.csv_path) as f:
            reader = csv.reader(f)
            data = list(reader)
        self.split = split
        if split == 'train':
            self.data = [i for i in data if i[2]=="train"]
        elif split == 'test':
            self.data = [i for i in data if i[2]=="test"]
        
        self.files = glob(self.pdb_path + "/*.pdb")

    def __getitem__(self, index):
        id = self.data[index][0]
        file = [i for i in self.files if id in i][0]

        cont_map = np.load(file.replace(".pdb", ".npy"))

        label = self.data[index][1]
        return torch.from_numpy(cont_map), int(label)
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    batch_size = 32
    device = torch.device("cuda:0")
    train_ds = protein_sequence_ds('./Data/DataSplit.csv', 'train', './Data/deeploc_af2')
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=5)

    test_ds = protein_sequence_ds('./Data/DataSplit.csv', 'test', './Data/deeploc_af2')
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    criterion.to(device)
    model.train()
    epochs = 50

    top_acc = 0
    for epoch in range(epochs):
        model.train()
        with tqdm(train_dl, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = criterion(output, target)
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size

                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        if epoch % 3 == 0:
            model.eval()
            y_pred = []
            y_true = []

            # iterate over test data
            for inputs, labels in tqdm(test_dl):
                with torch.no_grad():
                    output = model(inputs.cuda()) # Feed Network

                    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                    y_pred.extend(output) # Save Prediction
                    
                    labels = labels.data.cpu().numpy()
                    y_true.extend(labels) # Save Truth
            acc = sum(np.array(y_true) == np.array(y_pred))/len(y_true)

            if(acc > top_acc):
                top_acc = acc
                torch.save(model.state_dict(), f"./Models/model_{epoch}_acc{str(acc)}.pth")
