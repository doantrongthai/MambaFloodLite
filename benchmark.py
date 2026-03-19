import os
import sys
import torch
import random
import numpy as np
import argparse
import gdown
import zipfile
import json
import warnings

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.use_deterministic_algorithms(True, warn_only=False)
    warnings.filterwarnings('ignore', category=UserWarning)
    print(f"Seed set to {seed}")


DATASET = {
    'id': '1tg3N5DW27LWgJ9cTvNeIUz5xoBqSrmEs',
    'dir': 'floodkaggle'
}


def download_dataset():
    folder = DATASET['dir']
    if os.path.exists(folder):
        print("Dataset exists. Skipping.")
        return

    url = f'https://drive.google.com/uc?id={DATASET["id"]}'
    output = 'floodkaggle.zip'

    print("Downloading FloodKaggle...")
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(DATASET['dir'])

    os.remove(output)
    print("Dataset ready.")


def run_multiseed_experiments(args, seeds):
    print(f"\n{'#'*70}")
    print(f"MULTI-SEED EXPERIMENT")
    print(f"Seeds: {seeds}")
    print(f"{'#'*70}\n")

    results = []
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}\n")

        set_seed(seed)
        from utils.trainer import train_segmentation

        result = train_segmentation(
            model_name=args.model,
            loss_name='bce',
            size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dataset=DATASET['dir'],
            output_path=os.path.join(args.output_path, f'seed_{seed}'),
            seed=seed,
            num_classes=1
        )
        results.append({
            'seed': seed,
            'test_loss': result['test_loss'],
            'iou': result['iou'],
            'dice': result['dice'],
            'best_val_loss': result['best_val_loss'],
            'total_params': result['total_params']
        })

    losses     = [r['test_loss']    for r in results]
    ious       = [r['iou']          for r in results]
    dices      = [r['dice']         for r in results]
    val_losses = [r['best_val_loss'] for r in results]

    print(f"\n{'='*70}")
    print("STATISTICS FOR PAPER")
    print(f"{'='*70}")
    print(f"Test Loss:    {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"IoU:          {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
    print(f"Dice Score:   {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"Val Loss:     {np.mean(val_losses):.4f} +/- {np.std(val_losses):.4f}")
    print(f"Parameters:   {results[0]['total_params']:,}")
    print(f"{'='*70}\n")

    result_file = os.path.join(args.output_path, f'{args.model}_floodkaggle_multiseed.json')
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'model': args.model,
                'dataset': 'floodkaggle',
                'loss': 'bce',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'size': args.size
            },
            'seeds': seeds,
            'results': results,
            'statistics': {
                'test_loss':  {'mean': float(np.mean(losses)),     'std': float(np.std(losses))},
                'iou':        {'mean': float(np.mean(ious)),        'std': float(np.std(ious))},
                'dice':       {'mean': float(np.mean(dices)),       'std': float(np.std(dices))},
                'val_loss':   {'mean': float(np.mean(val_losses)),  'std': float(np.std(val_losses))}
            },
            'total_params': results[0]['total_params']
        }, f, indent=2)

    print(f"Results saved to: {result_file}")
    print(f"\nLaTeX format:")
    print(f"Test Loss: ${np.mean(losses):.4f} \\pm {np.std(losses):.4f}$")
    print(f"IoU:       ${np.mean(ious):.4f} \\pm {np.std(ious):.4f}$")
    print(f"Dice:      ${np.mean(dices):.4f} \\pm {np.std(dices):.4f}$\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Flood Detection Training')

    parser.add_argument('--model',       type=str,   required=True)
    parser.add_argument('--size',        type=int,   default=256)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--output_path', type=str,   default='outputs')
    parser.add_argument('--download',    action='store_true')
    parser.add_argument('--multiseed',   action='store_true')
    parser.add_argument('--seeds',       type=int, nargs='+', default=[42, 123, 456, 789, 2024])

    args = parser.parse_args()

    set_seed(args.seed)

    if args.download:
        download_dataset()

    if args.multiseed:
        run_multiseed_experiments(args, seeds=args.seeds)
        return

    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset:       floodkaggle")
    print(f"Model:         {args.model}")
    print(f"Size:          {args.size}")
    print(f"Loss:          bce")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Seed:          {args.seed}")
    print(f"Output Path:   {args.output_path}")
    print("="*70)

    from utils.trainer import train_segmentation
    train_segmentation(
        model_name=args.model,
        loss_name='bce',
        size=args.size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset=DATASET['dir'],
        output_path=args.output_path,
        seed=args.seed,
        num_classes=1
    )

if __name__ == '__main__':
    main()