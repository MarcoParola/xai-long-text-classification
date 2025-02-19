import hydra
import torch
import os
import wandb
from tqdm import tqdm
from src.datasets.dataset import load_dataset
from src.models.model import load_model
from src.log import get_loggers
from omegaconf import OmegaConf

@hydra.main(config_path='config', config_name='config')
def main(cfg):   
    wandb_logger = get_loggers(cfg) # loggers

    # TODO: IMPLEMENTARE LA FUNZIONE LOAD DATASET CHE CARICA IL DATASET CHE VOGLIAMO FINETUNARE
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir)

    # CARICA IL MODELLO GRAZIE ALLA FUNZIONE LOAD MODEL
    model = load_model(cfg.model, cfg.dataset.num_classes)
    device = torch.device(cfg.device)
    model.to(device)

    quantitative_metrics = # TODO

    for idx, (x, y) in enumerate(tqdm(test)):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        # COMPUTE QUANTITATIVE METRICS

    # results logging
    wandb_logger.log_metrics({
        # TODO
    })

     
if __name__ == '__main__':
    main()

