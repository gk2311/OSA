# %%
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import GroupShuffleSplit
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import torch.optim as optim
import time
import sys
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances, classification_report
from sklearn.metrics import confusion_matrix
import pickle as pkl
from tqdm import tqdm as tqdm
from tqdm import tqdm
from skimage import exposure
from skimage import io
from os import path
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
from PIL import Image
from tqdm import tqdm as tqdm
from tqdm.auto import tqdm
import cv2

TRAIN_FILE = "/media/Datacenter_storage/GK/SleepApnoea/Data/PreporcessIMG_A_Train_Binary.csv"
TEST_FILE = "/media/Datacenter_storage/GK/SleepApnoea/Data/PreporcessIMG_A_Test_Binary.csv"
VAL_FILE = "/media/Datacenter_storage/GK/SleepApnoea/Data/PreporcessIMG_A_Val_Binary.csv"

files = {
    "train": TRAIN_FILE,
    "test": TEST_FILE,
    "val": VAL_FILE
}
df = {key: pd.read_csv(value) for key, value in files.items()}

hyperparameters = {
    "n_iters": 5,
    "print_every": 1,
    "plot_every": 1,
    "batch_size": 32,
    "model_name": "Resnet18",
    "save_dir": "/media/Datacenter_storage/GK/SleepApnoea/Code/OSA_PD_Results",
    "metric_name": "precision",
    "lr": 0.001,
    "maximize_metric": True,
    "patience": 7,
    "early_stop": False,
    "prev_val_loss": 1e10,
    "num_workers": 8
}

class OSADataset(Dataset):
    def __init__(self, df, transform, mode):
        self.df = df
        self.mode = mode
        self.transform = transform
        self.targets = df["Binary_Label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_name = self.df.at[index, "PNG_FIXED"]
        y = self.df.at[index, "Binary_Label"]
        one_hot = np.zeros(len(self.df["Binary_Label"].unique()))
        try:
            x = io.imread(image_name)
            x = ((x - x.mean()) / x.std()).astype(np.uint8)
            if self.transform:
                x = self.transform(x)
                one_hot[y]=1.
                one_hot = np.array(one_hot).astype(np.float32) 
            return x, one_hot
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            return None

class Model(nn.Module):
    def __init__(self, pretrained):
        super(Model, self).__init__()
        self.model = models.resnet18(pretrained=False)
        print(device)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
            nn.Softmax()
        )
        print('model loaded')
    
    def len_layers(self):
    count = 0
    for param in self.model.parameters():
        count += 1
    return count

def num_freeze_layers(self, block):
    count = 0
    flg = 0
    for layer_name, param in reversed(list(self.model.named_parameters())):
        if flg == 1:
            if block in layer_name:
                count += 1
        else:
            count += 1
            if block in layer_name:
                flg = 1
    return count
def freeze_layers(self, block):
    lay = self.num_freeze_layers(block)
    layers = self.len_layers()
    for param in self.model.parameters():
        if layers == lay:
            break
        param.requires_grad = False
        layers -= 1

    def forward(self, x):
        x = self.model(x)
        return x

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            #transforms.RandomRotation(degrees=90),
                            transforms.RandomHorizontalFlip(),
                            transforms.Normalize(mean=[0.5081083, 0.5081083, 0.5081083],std=[0.281246, 0.281246, 0.281246]),
                           ])

datagen = {key: OSADataset(df=value.copy(), transform=trans, mode=key) for key, value in df.items()}
batch_size = hyperparameters["batch_size"]
num_workers = hyperparameters["num_workers"]
# 
dataloaders = {
   key: DataLoader(dataset=value, shuffle=(key == "train"), batch_size=batch_size, num_workers=num_workers)
   for key, value in datagen.items()
}

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def show_images(images, title):
    images = np.transpose(images.numpy(), (0, 2, 3, 1))
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    for i in range(25):
        ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])

def evaluate(model, loader, criterion, device, y_true, y_pred, y_prob):
    model.eval()
    current_loss = 0
    counter = 0
    start = time.time()
    for v, data in tqdm(enumerate(val_loader)):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        output = model(data[0].view(-1,3, 256, 256))
        loss = criterion(output,data[1])
        probs = torch.max(output, dim=1).values.cpu().detach().numpy()
        preds = torch.argmax(output, dim=1).cpu().detach().numpy().astype(int)

        y_pred += list(preds)
        y_prob +=list(probs)
        y_true += list(torch.argmax(data[1],axis=1).cpu().detach().numpy())
        current_loss += loss.item()
        counter += 1
        sys.stdout.flush()

    avg_loss = current_loss / counter
    classification_rep = classification_report(y_true, y_pred)
    print("Classification Report")
    print(classification_rep)
    confusion_mtx = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix")
    print(confusion_mtx)
    return avg_loss, classification_rep, confusion_mtx

def train(do_train, itr, n_iters, model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
print_every, patience):
    train_losses = []
    val_losses = []
    early_stop = 0
    best_val_recall = float('-inf')
    if do_train:
        while (itr != n_iters) and not (early_stop):
            model.train()
            y_pred = []
            y_true = []
            y_prob = []
            current_loss = 0
            counter = 0
            for j, data in tqdm(enumerate(train_loader)):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)

                output = model(data[0].view(-1,3, 256, 256))

                loss = criterion(output, data[1])
                loss.backward()
                optimizer.step()
                scheduler.step()
                current_loss += loss.item()
                counter += 1

                probs = torch.max(output, dim=1).values.cpu().detach().numpy()
                preds = torch.argmax(output, dim=1).cpu().detach().numpy().astype(int)
                y_pred += list(preds)
                y_prob += list(probs)
                y_true += list(torch.argmax(data[1],axis=1).cpu().detach().numpy())
                sys.stdout.flush()
            print('ITERATION:', itr, '\nTRAIN LOSS:', current_loss/counter)
            train_losses.append(current_loss/counter)
            classification_rep = classification_report(y_true, y_pred)
            print("Classification Report")
            print(classification_rep)
            confusion_mtx = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:")
            print(confusion_mtx)
            sys.stdout.flush()

        if itr % hyperparameters["print_every"] == 0:
            current_loss = 0
            model.eval()
            y_pred = []
            y_true = []
            y_prob= []
            start = time.time()
            for v, data in enumerate(val_loader):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                output = model(data[0].view(-1,3, 256, 256))
                loss = criterion(output,data[1])
                probs = torch.max(output, dim=1).values.cpu().detach().numpy()
                for prob in probs:
                    sys.stdout.write(f"{prob:0.4f}\n")

if __name__ == '__main__':
    
    model_name = hyperparameters["model_name"]
    
    model = Model(model_name)
    if cuda:
        model = nn.DataParallel(model)
        model = model.to(device)
        model.train()
        
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=hyperparameters["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.11)
    
    train_loader = DataLoader(OSADataset(df=df["train"], transform=trans, mode="train"), batch_size=32, shuffle = True)
    val_loader = DataLoader(OSADataset(df=df["val"], transform=trans, mode='val'), batch_size=32, shuffle=False)
    test_loader = DataLoader(OSADataset(df=df["test"], transform=trans, mode='test'), batch_size=32, shuffle=False)

    # Visualize a sample from dataloader
    images, _ = next(iter(train_loader))
    show_images(images, "Train Images")
    
    images, _ = next(iter(val_loader))
    show_images(images, "Validation Images")

    for epoch in range(5):
        train_loss = train(do_train=True, itr=0, n_iters=hyperparameters["n_iters"], model = model, train_loader = train_loader, val_loader = val_loader, criterion = criterion, optimizer = optimizer , scheduler = scheduler, device = device, print_every = hyperparameters["print_every"], patience = hyperparameters["patience"])
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f'Epoch: {epoch+1}/100, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    test_loss, test_acc = test(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')