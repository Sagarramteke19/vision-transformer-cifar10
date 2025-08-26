import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from src.datasets import get_cifar10 # type: ignore
from src.models.vit_timm import create_model as create_vit # type: ignore
from src.models.cnn_baseline import create_model as create_cnn # type: ignore

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='vit_tiny_patch16_224', help='timm model name or resnet18 for baseline')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--use-cuda', action='store_true')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--ckpt', type=str, default='')
    args = ap.parse_args()

    device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')
    train_loader, test_loader = get_cifar10(batch_size=args.batch_size, img_size=args.img_size)

    if args.model == 'resnet18':
        model = create_cnn(num_classes=10)
    else:
        model = create_vit(model_name=args.model, num_classes=10, pretrained=False)

    model.to(device)
    ckpt_path = os.path.join('results', f'best_{args.model.split("_")[0]}.pt')

    if args.eval and args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        acc = evaluate(model, test_loader, device)
        print(f'[EVAL] Accuracy: {acc:.4f}')
        return

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        acc = evaluate(model, test_loader, device)
        print(f'[VAL] epoch={epoch} acc={acc:.4f}')
        if acc > best:
            best = acc
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f'[CKPT] saved: {ckpt_path}')

    print(f'[DONE] best_acc={best:.4f}')

if __name__ == '__main__':
    main()
