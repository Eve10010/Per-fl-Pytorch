import sys
import numpy as np
import torch
from args import args_parser
import pickle
import random
sys.path.append('../')
from torch.utils.data import Dataset, DataLoader

args = args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_name):
    """
    :param file_name: csv file name
    :return: normalized dataframe
    """
    with open('data/' + file_name, 'rb') as file:
        data = pickle.load(file)
    return data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq(file_name, B):
    """
    :param file_name: pickle file name
    :param B: batch size
    :return: DataLoader data
    """
    data = load_data(file_name)
#    print(data[0])
#    data = np.array([token for st in data for token in st])
#    data = np.array([token for st in data for token in st])
#    print("data[0]", data[0])
    length = len(data)
    seq = []
    for i in range(length):
        train_data, train_label = data[i]
        train_data = torch.Tensor(train_data).view(3, 32, 32).type(torch.float32)
        train_label = torch.Tensor([train_label]).view(-1).type(torch.float32)
        train_label = train_label.squeeze()
        seq.append((train_data, train_label))
#        print("seq", seq)
    random.shuffle(seq)
    Dtr = seq[0:int(len(seq) * 0.8)]
    Dte = seq[int(len(seq) * 0.8):len(seq)]

    train_len = int(len(Dtr) / B) * B
    test_len = int(len(Dte) / B) * B
    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    # print("Dtr[0]", Dtr[0])
    # print("len_Dtr", len(Dtr))

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=True, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=B, shuffle=True, num_workers=0)

    return Dtr, Dte


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))
