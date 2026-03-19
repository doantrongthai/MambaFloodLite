import torch
import numpy as np

def calculate_iou(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if all_preds.ndim == 4:
        all_preds = all_preds.squeeze(1)
    if all_labels.ndim == 4:
        all_labels = all_labels.squeeze(1)

    all_preds = (all_preds > threshold).astype(np.uint8)
    all_labels = all_labels.astype(np.uint8)

    all_preds_flat = all_preds.reshape(-1)
    all_labels_flat = all_labels.reshape(-1)

    intersection = np.logical_and(all_preds_flat, all_labels_flat).sum()
    union = np.logical_or(all_preds_flat, all_labels_flat).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def calculate_dice_score(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if all_preds.ndim == 4:
        all_preds = all_preds.squeeze(1)
    if all_labels.ndim == 4:
        all_labels = all_labels.squeeze(1)

    all_preds = (all_preds > threshold).astype(np.uint8)
    all_labels = all_labels.astype(np.uint8)

    all_preds_flat = all_preds.reshape(-1)
    all_labels_flat = all_labels.reshape(-1)

    intersection = np.logical_and(all_preds_flat, all_labels_flat).sum()
    dice = (2.0 * intersection) / (all_preds_flat.sum() + all_labels_flat.sum() + 1e-8)

    return float(dice)