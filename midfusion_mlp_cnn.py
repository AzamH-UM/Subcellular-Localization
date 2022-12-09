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

with open("./Data/deeploc.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)
classes = [int(i[2]) for i in data if i[-1] == "train"]

CLASS_WEIGHTS = compute_class_weight("balanced", classes=np.unique(classes), y=classes)
CLASS_WEIGHTS = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
PROTEIN_LETTERS = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'P':14, 'Q':15, 'R':16, 'S':17, 'T':18, 'U':19, 'V':20, 'W':21, 'Y':22, 'X':23, 'Z':24}

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

def seq_to_tensor(seq):
    sq_len = len(seq)
    seq_ten = np.zeros(len(PROTEIN_LETTERS), dtype=np.float32)
    uniques = np.unique(list(seq), return_counts=True)

    for i in range(len(uniques[0])):
        seq_ten[PROTEIN_LETTERS[uniques[0][i]]] = uniques[1][i]/sq_len
    return seq_ten

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
        self.feats = np.load("data/esm_feats.npy", allow_pickle=True).item()
        
        self.files = glob(self.pdb_path + "/*.pdb")

    def __getitem__(self, index):
        id = self.data[index][0]
        file = [i for i in self.files if id in i][0]
        cont_map = np.load(file.replace(".pdb", ".npy"))
        seq = self.feats[self.data[index][0]][0]
        label = self.data[index][1]

        return torch.from_numpy(cont_map), torch.from_numpy(seq), int(label)
    
    def __len__(self):
        return len(self.data)

class seq_model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(seq_model, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, 2048),
                      nn.ReLU(),

                      nn.Linear(2048, 1024),
                      nn.BatchNorm1d(1024),
                      nn.ReLU(),

                      nn.Linear(1024, 512),
                      nn.BatchNorm1d(512),
                      nn.ReLU(),

                      nn.Linear(512, 256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(),

                      nn.Linear(256, 128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(),

                      nn.Linear(128, num_classes),
                      nn.ReLU()
                      )
    
    def forward(self, x):
        return self.model(x)

class combo_model(nn.Module):
    def __init__(self):
        super(combo_model, self).__init__()
        self.cnn = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(512, 64)
        self.cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.seq_mod = seq_model(1280, 64)

        self.comb_mod = nn.Sequential(nn.Linear(128, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32, 10),
                        nn.Softmax())
    
    def forward(self, img, seq):
        img = self.cnn(img)
        seq = self.seq_mod(seq)
        x = torch.cat((img, seq), dim=1)
        
        return self.comb_mod(x)


if __name__ == "__main__":
    batch_size = 32
    device = torch.device("cuda:0")
    train_ds = protein_sequence_ds('./data/Data.csv', 'train', './data/deeploc_af2')
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    test_ds = protein_sequence_ds('./data/Data.csv', 'test', './data/deeploc_af2')
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = combo_model()
    
    # criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.to(device)
    criterion.to(device)
    model.train()
    epochs = 50

    for epoch in range(epochs):
        model.train()
        with tqdm(train_dl, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                img, seq, target = data[0].to(device), data[1].to(device) ,data[2].to(device),
                optimizer.zero_grad()
                output = model(img, seq)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = criterion(output, target)
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size

                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                #  sleep(0.1)

        if epoch % 2 == 0:
            model.eval()
            y_pred = []
            y_true = []
            top_acc = 0
            # iterate over test data
            for data in tqdm(test_dl):
                with torch.no_grad():
                    img, seq, labels = data[0].to(device), data[1].to(device) ,data[2].to(device),
                    output = model(img, seq)

                    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                    y_pred.extend(output)
                    
                    labels = labels.data.cpu().numpy()
                    y_true.extend(labels)
            acc = 100*sum(np.array(y_true) == np.array(y_pred))/len(y_true)
            
            if(acc>top_acc):
                print(f"New Top Accuracy: {acc}")
                torch.save(model.state_dict(), f"./Models/model_combo{str(acc)}.pth")
