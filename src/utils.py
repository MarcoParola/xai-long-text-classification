import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Subset
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import numpy as np
import omegaconf

def get_save_model_callback(save_path):
    save_model_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=save_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
    )
    return save_model_callback

def get_early_stopping(patience=10):
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
    )
    return early_stopping_callback
  
