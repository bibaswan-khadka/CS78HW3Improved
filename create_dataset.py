from torch.utils.data import TensorDataset
from torch import load
import torch
import torchvision

def create_dataset(data_path):
    """
    Reads the data and prepares the training and validation sets. No preprocessing is required.

    Arguments
    ---------
    data_path: (string),  the path to the file containing the data

    Return
    ------
    train_ds: (TensorDataset), the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """
    data = load(data_path)
    data_tr = data['images_tr']
    sets_tr = data['sets_tr']
    anno_tr = data['anno_tr']

    #per_pixel_mean = torch.mean(data_tr, 0)
    #data_tr = data_tr - per_pixel_mean

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(1)
    ])
    transformspers = torchvision.transforms.Compose([
      torchvision.transforms.ColorJitter(),
    ])
    # pers = transformspers(data_tr[sets_tr == 1])
    combinedtr = torch.cat((data_tr[sets_tr ==1],transforms(data_tr[sets_tr ==1])),0)
    #combinedtrpers = torch.cat((combinedtr,pers),0)
    combinedlabel = torch.cat((anno_tr[sets_tr==1],anno_tr[sets_tr==1].flip(2)),0)
    #combinedlabelpers = torch.cat((combinedlabel,anno_tr[sets_tr==1]),0)
    train_ds = TensorDataset(combinedtr,combinedlabel)
    #train_ds = TensorDataset(data_tr[sets_tr ==1],anno_tr[sets_tr==1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], anno_tr[sets_tr == 2])

    return train_ds, val_ds
