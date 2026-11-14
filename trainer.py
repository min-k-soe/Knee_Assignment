from args import get_args
import os
import torch
import torch.nn as nn
from utils import dice_loss_from_logits, dice_score_from_logits


def train_model(model, train_loader, val_loader, device):
    args = get_args()

    model = model.to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_dice = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, 'best_model.pth')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for data_batch in train_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice = validate_model(model, val_loader, bce, device)

        print(f"Epoch [{epoch + 1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model (Val Dice: {best_val_dice:.4f}) to {best_model_path}")


def validate_model(model, val_loader, bce, device):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()
            running_dice += dice_score_from_logits(outputs, masks)

    val_loss = running_loss / len(val_loader)
    val_dice = running_dice / len(val_loader)

    return val_loss, val_dice