import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from args import get_args
from datasets import KneeDataset
from model import UNetLext
from utils import dice_score_from_logits


def evaluate():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test set
    test_set = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))
    has_masks = 'masks' in test_set.columns and test_set['masks'].notna().any()

    # Load best trained model
    model = UNetLext(
        input_channels=1,
        output_channels=1,
        pretrained_model=False,
        path_weights='',
        path_pretrained='',
        resolution=False
    ).to(device)

    best_model_path = os.path.join(args.out_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    if has_masks:
        test_set = test_set.dropna(subset=['xray', 'masks']).reset_index(drop=True)
        test_dataset = KneeDataset(test_set)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

        total_dice = 0.0
        sample_images, sample_masks, sample_preds = None, None, None

        with torch.no_grad():
            for i, data_batch in enumerate(test_loader):
                images = data_batch['xray'].to(device)
                masks = data_batch['mask'].to(device)
                logits = model(images)

                total_dice += dice_score_from_logits(logits, masks)

                if i == 0:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    sample_images = images.cpu()
                    sample_masks = masks.cpu()
                    sample_preds = preds.cpu()

        mean_test_dice = total_dice / len(test_loader)
        print(f"Test Set Mean Dice Score: {mean_test_dice:.4f}")

        num_samples = min(3, sample_images.size(0))
        fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3.5 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for idx in range(num_samples):
            axes[idx][0].imshow(sample_images[idx, 0], cmap='gray')
            axes[idx][0].set_title('Input X-Ray')
            axes[idx][0].axis('off')

            axes[idx][1].imshow(sample_masks[idx, 0], cmap='gray')
            axes[idx][1].set_title('Ground Truth Mask')
            axes[idx][1].axis('off')

            axes[idx][2].imshow(sample_preds[idx, 0], cmap='gray')
            axes[idx][2].set_title('Predicted Mask')
            axes[idx][2].axis('off')
    else:
        # Test set has no ground-truth masks (unlabeled test split)
        # Fill dummy path temporarily only if needed, or read xrays directly for inference
        from datasets import read_xray
        sample_images, sample_preds = [], []
        num_samples = min(3, len(test_set))

        with torch.no_grad():
            for idx in range(num_samples):
                xray_np = read_xray(test_set['xray'].iloc[idx])  # (1, H, W)
                img_tensor = torch.from_numpy(xray_np).unsqueeze(0).to(device)  # (1, 1, H, W)
                logits = model(img_tensor)
                pred = (torch.sigmoid(logits) > 0.5).float().cpu()[0]
                sample_images.append(xray_np)
                sample_preds.append(pred)

        print(f"Generated predicted masks for {len(test_set)} test images (test.csv has no ground-truth masks).")

        fig, axes = plt.subplots(num_samples, 2, figsize=(7, 3.5 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for idx in range(num_samples):
            axes[idx][0].imshow(sample_images[idx][0], cmap='gray')
            axes[idx][0].set_title('Input X-Ray (Test)')
            axes[idx][0].axis('off')

            axes[idx][1].imshow(sample_preds[idx][0], cmap='gray')
            axes[idx][1].set_title('Predicted Mask')
            axes[idx][1].axis('off')

    plt.tight_layout()
    save_fig_path = os.path.join(args.out_dir, 'test_predictions.png')
    plt.savefig(save_fig_path)
    print(f"Saved example predictions to {save_fig_path}")
    plt.show()


if __name__ == '__main__':
    evaluate()