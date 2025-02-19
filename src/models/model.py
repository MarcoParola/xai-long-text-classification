import torch
import torchvision
import numpy as np
import os


def load_model(model_name, num_classes):
    model = None
    if model_name == 'bert':
        # TODO implementare il caricamento del modello bert
    if model_name == 'bert-lstm':
        # TODO implementare il caricamento del modello bert-lstm
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model