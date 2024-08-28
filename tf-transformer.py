# https://velog.io/@kyyle/%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84-FT-Transformer

!pip install rtdl
!pip install optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import rtdl
from typing import Any, Dict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

def seed_everything(seed = 21):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_split_data():
    # df = pd.read_csv('캐글데이터')
    X = {}
    y = {}
    X['train'], X['test'], y['train'], y['test'] = train_test_split(df.iloc[:, :-1], df.income, test_size = 0.10, random_state=21)
    X['train'], X['val'], y['train'], y['val'] = train_test_split(X['train'], y['train'], test_size = 0.10, random_state=21)

    return X, y


def preprocessing(X, y):
    cat_index = X['train'].select_dtypes(['object']).columns
    num_index = X['train'].select_dtypes(['int64']).columns

    # categorical cardinalities for CategoricalFeatureTokenizer
    cat_cardinalities = []

    # StandardScaler
    ss = StandardScaler()
    X['train'][num_index] = ss.fit_transform(X['train'][num_index])
    X['val'][num_index] = ss.transform(X['val'][num_index])
    X['test'][num_index] = ss.transform(X['test'][num_index])
    # float64 -> float32 (recommended)
    X['train'][num_index] = X['train'][num_index].apply(lambda x: x.astype('float32'))
    X['val'][num_index] = X['val'][num_index].apply(lambda x: x.astype('float32'))
    X['test'][num_index] = X['test'][num_index].apply(lambda x: x.astype('float32'))

    # LabelEncoder
    for col in cat_index:
      le = LabelEncoder()

      X['train'][col] = le.fit_transform(X['train'][col])

      # X_val, X_test에만 존재하는 label이 있을 경우
      for label in np.unique(X['val'][col]):
        if label not in le.classes_:
          le.classes_ = np.append(le.classes_, label)

      for label in np.unique(X['test'][col]):
        if label not in le.classes_:
          le.classes_ = np.append(le.classes_, label)

      X['val'][col] = le.transform(X['val'][col])
      X['test'][col] = le.transform(X['test'][col])

      # cardinalities
      max_cat = np.max([np.max(X['train'][col]),
                        np.max(X['val'][col]),
                        np.max(X['test'][col])]) + 1
      cat_cardinalities.append(max_cat)

    # y = 1 if > 50K
    y['train'] = np.where(y['train']=='>50K', 1, 0).reshape(-1, 1)
    y['val'] = np.where(y['val']=='>50K', 1, 0).reshape(-1, 1)
    y['test'] = np.where(y['test']=='>50K', 1, 0).reshape(-1, 1)

    return X, y, cat_cardinalities

def setting_rtdl(data, label):
    '''
    DataFrame, np.array -> torch.Tensor
    ResNet: model(X_num, X_cat) / split X -> X_num, X_cat
    '''
    cat_index = data['train'].select_dtypes(['int64']).columns
    num_index = data['train'].select_dtypes(['float32']).columns

    X = {'train': {},
         'val': {},
         'test': {}}
    y = {'train': {},
         'val': {},
         'test': {}}

    X['train']['num'] = torch.tensor(data['train'][num_index].values, device=device)
    X['train']['cat'] = torch.tensor(data['train'][cat_index].values, device=device)

    X['val']['num'] = torch.tensor(data['val'][num_index].values, device=device)
    X['val']['cat'] = torch.tensor(data['val'][cat_index].values, device=device)

    X['test']['num'] = torch.tensor(data['test'][num_index].values, device=device)
    X['test']['cat'] = torch.tensor(data['test'][cat_index].values, device=device)

    # dtype=float for BCELoss
    y['train'] = torch.tensor(label['train'], dtype=torch.float, device=device)
    y['val'] = torch.tensor(label['val'], dtype=torch.float, device=device)
    y['test'] = torch.tensor(label['test'], dtype=torch.float, device=device)

    return X, y

class TensorData(Dataset):
    def __init__(self, num, cat, label):
        self.num = num
        self.cat = cat
        self.label = label
        self.len = self.label.shape[0]

    def __getitem__(self, index):
        return self.num[index],self.cat[index], self.label[index]

    def __len__(self):
        return self.len


def model_train(model, data_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0
    corr = 0

    # for rtdl
    for x_num, x_cat, label in tqdm(data_loader):
        optimizer.zero_grad()

        x_num, x_cat, label = x_num.to(device), x_cat.to(device), label.to(device)
        output = model(x_num, x_cat)
        output = torch.sigmoid(output)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        pred = output >= torch.FloatTensor([0.5]).to(device)
        corr += pred.eq(label).sum().item()
        running_loss += loss.item() * x_num.size(0)

    if scheduler:
        scheduler.step()

    # Average accuracy & loss
    accuracy = corr / len(data_loader.dataset)
    loss = running_loss / len(data_loader.dataset)
    history['train_loss'].append(loss)
    history['train_accuracy'].append(accuracy)

    return loss, accuracy


def model_evaluate(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        corr = 0

        for x_num, x_cat, label in data_loader:
            x_num, x_cat, label = x_num.to(device), x_cat.to(device), label.to(device)
            output = model(x_num, x_cat)
            output = torch.sigmoid(output)

            pred = output >= torch.FloatTensor([0.5]).to(device)
            corr += pred.eq(label).sum().item()
            running_loss += criterion(output, label).item() * x_num.size(0)

        accuracy = corr / len(data_loader.dataset)
        loss = running_loss / len(data_loader.dataset)
        history['val_loss'].append(loss)
        history['val_accuracy'].append(accuracy)

        return loss, accuracy


def model_tune(model, train_loader, val_loader, criterion, optimizer, device):
    model.train()

    # train_loader
    for x_num, x_cat, label in train_loader:
        optimizer.zero_grad()
        x_num, x_cat, label = x_num.to(device), x_cat.to(device), label.to(device)
        output = model(x_num, x_cat)
        output = torch.sigmoid(output)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    # val_loader
    model.eval()
    with torch.no_grad():
        running_loss = 0
        corr = 0

        for x_num, x_cat, label in val_loader:
            x_num, x_cat, label = x_num.to(device), x_cat.to(device), label.to(device)
            output = model(x_num, x_cat)
            output = torch.sigmoid(output)
            pred = output >= torch.FloatTensor([0.5]).to(device)
            corr += pred.eq(label).sum().item()
            running_loss += criterion(output, label).item() * x_num.size(0)

        val_accuracy = corr / len(val_loader.dataset)
        val_loss = running_loss / len(val_loader.dataset)

        return val_loss, val_accuracy


def plot_loss(history):
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()

def plot_acc(history):
    plt.plot(history['train_accuracy'], label='train', marker='o')
    plt.plot(history['val_accuracy'], label='val',  marker='o')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
  
def ready_data():
    # data setting
    seed_everything()
    X, y = read_split_data()
    X, y, cardinalities = preprocessing(X, y)
    X, y = setting_rtdl(X, y)

    # dataset, dataloader
    train_data = TensorData(X['train']['num'], X['train']['cat'], y['train'])
    val_data = TensorData(X['val']['num'], X['val']['cat'], y['val'])
    test_data = TensorData(X['test']['num'], X['test']['cat'], y['test'])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    return X, y, cardinalities, train_loader, val_loader, test_loader

ft_t = rtdl.FTTransformer.make_baseline(n_num_features=X['train']['num'].shape[1],
                                        cat_cardinalities=cardinalities,
                                        d_token=64,
                                        n_blocks=3,
                                        attention_dropout=0.2,
                                        ffn_d_hidden=128,
                                        ffn_dropout=0.1,
                                        residual_dropout=0.1,
                                        d_out=1).to(device)

# test evaluate
ft_t.load_state_dict(torch.load('FT-Transformer_Best.pth'))
test_loss, test_acc = model_evaluate(ft_t, test_loader, criterion, device)
print('Test loss:', test_loss, '\nTest acc:', test_acc)
