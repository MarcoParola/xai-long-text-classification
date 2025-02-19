import torch
import torchvision
import numpy as np
import os
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import datasets



def load_dataset(dataset, data_dir, val_split=0.15, test_split=0.15):

    train, val, test = None, None, None

    torch.manual_seed(42)
    np.random.seed(42)
    
    if dataset == 'movies':
        train = # TODO
        val = # TODO
        test = # TODO

    elif dataset == 'refugees':
        train = # TODO
        val = # TODO
        test = # TODO
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return train, val, test


if __name__ == "__main__":

    # TODO fai un main di test per testare il funzionamento di ogni dataset prima di usarlo
