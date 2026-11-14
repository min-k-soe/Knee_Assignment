from args import get_args
import numpy as np
import pandas as pd
import os
import torch
from datasets import KneeDataset
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model

if __name__ == '__main__':
    # 1: we need arguments
    args = get_args()

    # 2: Loading the CSV files
    train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    # 3: Define your datasets
    train_dataset = KneeDataset(train_set)
    val_dataset = KneeDataset(val_set)

    # 4: Create data dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    # 5: Initialize the model
    model = UNetLext(
        input_channels=1,
        output_channels=1,
        pretrained_model=False,
        path_weights='',
        path_pretrained='',
        resolution=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 6: Train the model
    train_model(model, train_loader, val_loader, device)