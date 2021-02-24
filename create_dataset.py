from torch.utils.data import TensorDataset
from torch import load
import torch

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
    train_ds = TensorDataset(data_tr[sets_tr == 1], anno_tr[sets_tr == 1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], anno_tr[sets_tr == 2])

    return train_ds, val_ds
