"""
GazeMLP の学習スクリプト。
合成データ生成・学習・評価・モデル保存を行う。

使い方:
    python train.py                     # 合成データで動作確認
    python train.py --data X.npy y.npy  # 実データで学習
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Dict, Any

from model import GazeMLP, hybrid_loss


# ──── 評価指標 ────────────────────────────────────────────────────────────────

def compute_mgae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean Gaze Angular Error (degrees).
    MGAE = mean arccos(dot(g, g_hat) / (||g|| ||g_hat||)) * 180/pi
    """
    eps = 1e-8
    p = pred   / (np.linalg.norm(pred,   axis=-1, keepdims=True) + eps)
    t = target / (np.linalg.norm(target, axis=-1, keepdims=True) + eps)
    cos_sim = np.sum(p * t, axis=-1).clip(-1 + eps, 1 - eps)
    return float(np.mean(np.arccos(cos_sim) * 180.0 / np.pi))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error（正規化画面座標上）。"""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ──── 合成データ生成 ──────────────────────────────────────────────────────────

def make_synthetic_data(
    n: int = 2000,
    noise_sigma: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    視線推定のダミーデータを生成。
    実際の視線データが得られるまでパイプライン検証に使用。

    X: (n, 7)  [Lx, Ly, Rx, Ry, Pitch, Yaw, dist]
    y: (n, 2)  normalized screen coords [0, 1]
    """
    rng = np.random.RandomState(seed)
    y = rng.uniform(0.05, 0.95, (n, 2)).astype(np.float32)

    X = np.zeros((n, 7), dtype=np.float32)
    X[:, 0] = 0.30 + 0.12 * y[:, 0] + rng.normal(0, noise_sigma, n)   # Lx
    X[:, 1] = 0.35 + 0.10 * y[:, 1] + rng.normal(0, noise_sigma, n)   # Ly
    X[:, 2] = 0.55 + 0.12 * y[:, 0] + rng.normal(0, noise_sigma, n)   # Rx
    X[:, 3] = 0.35 + 0.10 * y[:, 1] + rng.normal(0, noise_sigma, n)   # Ry
    X[:, 4] = 0.25 * (y[:, 1] - 0.5) + rng.normal(0, 0.015, n)        # Pitch
    X[:, 5] = 0.70 * (y[:, 0] - 0.5) + rng.normal(0, 0.015, n)        # Yaw
    X[:, 6] = 0.12 + rng.normal(0, 0.004, n)                           # dist

    return X, y


# ──── トレーナー ──────────────────────────────────────────────────────────────

class GazeTrainer:
    """GazeMLP の学習管理クラス。"""

    def __init__(
        self,
        gamma:      float = 1.0,
        hidden_dim: int   = 16,
        device:     str   = 'auto',
    ):
        self.gamma = gamma

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = GazeMLP(input_dim=7, hidden_dim=hidden_dim, output_dim=2)
        self.model.to(self.device)

        self.history: Dict[str, list] = {
            'train_loss': [], 'val_loss': [], 'mgae': [], 'rmse': []
        }

    def make_loaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.2,
        batch_size: int  = 32,
    ) -> Tuple[DataLoader, DataLoader]:
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(Xt, yt)
        n_val   = max(1, int(len(ds) * val_ratio))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True),
            DataLoader(val_ds,   batch_size=128,        shuffle=False),
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int   = 300,
        lr:           float = 1e-3,
        patience:     int   = 30,
    ) -> Dict[str, list]:
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_val = float('inf')
        patience_cnt = 0
        best_state   = None

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Device: {self.device}  |  Params: {param_count}")
        print(f"{'Epoch':>6} | {'Train':>8} | {'Val':>8} | {'MGAE[d]':>8} | {'RMSE':>8}")
        print("-" * 50)

        for ep in range(1, epochs + 1):
            # ── Train ──
            self.model.train()
            t_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(Xb)
                loss = hybrid_loss(pred, yb, gamma=self.gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                t_losses.append(loss.item())

            # ── Validate ──
            self.model.eval()
            v_losses, preds_all, tgts_all = [], [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    pred = self.model(Xb)
                    v_losses.append(hybrid_loss(pred, yb, self.gamma).item())
                    preds_all.append(pred.cpu().numpy())
                    tgts_all.append(yb.cpu().numpy())

            t_loss = float(np.mean(t_losses))
            v_loss = float(np.mean(v_losses))
            preds  = np.concatenate(preds_all)
            tgts   = np.concatenate(tgts_all)
            mgae   = compute_mgae(preds, tgts)
            rmse   = compute_rmse(preds, tgts)

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['mgae'].append(mgae)
            self.history['rmse'].append(rmse)

            scheduler.step(v_loss)

            if ep % 20 == 0 or ep == 1:
                print(f"{ep:6d} | {t_loss:8.5f} | {v_loss:8.5f} | {mgae:7.2f} | {rmse:8.5f}")

            # ── Early stopping ──
            if v_loss < best_val - 1e-6:
                best_val = v_loss
                patience_cnt = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  -> Early stop at epoch {ep} (best val={best_val:.6f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            Xt   = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(Xt).cpu().numpy()
        return {
            'mgae': compute_mgae(pred, y),
            'rmse': compute_rmse(pred, y),
        }

    def save(self, path: str = 'gaze_model.pth') -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim':  self.model.input_dim,
                'hidden_dim': 16,
                'output_dim': self.model.output_dim,
            },
            'gamma':   self.gamma,
            'history': self.history,
        }, path)
        print(f"  Saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.gamma   = ckpt.get('gamma', self.gamma)
        self.history = ckpt.get('history', self.history)
        print(f"  Loaded ← {path}")


# ──── エントリポイント ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GazeMLP 学習スクリプト")
    parser.add_argument('--data',    nargs=2, metavar=('X_NPY', 'Y_NPY'),
                        help='実データ .npy ファイルパス (X y)')
    parser.add_argument('--epochs',  type=int,   default=300)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--gamma',   type=float, default=1.0,
                        help='Angular Loss 重み γ（Q1の回答で更新予定）')
    parser.add_argument('--hidden',  type=int,   default=16)
    parser.add_argument('--output',  type=str,   default='gaze_model.pth')
    parser.add_argument('--samples', type=int,   default=2000,
                        help='合成データのサンプル数（--data 未指定時）')
    args = parser.parse_args()

    if args.data:
        print(f"Loading data: {args.data[0]}, {args.data[1]}")
        X = np.load(args.data[0]).astype(np.float32)
        y = np.load(args.data[1]).astype(np.float32)
    else:
        print(f"Generating synthetic data: {args.samples} samples")
        X, y = make_synthetic_data(n=args.samples)

    print(f"  X: {X.shape}  y: {y.shape}")

    trainer = GazeTrainer(gamma=args.gamma, hidden_dim=args.hidden)
    train_loader, val_loader = trainer.make_loaders(X, y, val_ratio=0.2)

    print("\n-- Training start --")
    trainer.train(train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    print("\n-- Final evaluation --")
    metrics = trainer.evaluate(X, y)
    print(f"  MGAE : {metrics['mgae']:.3f} deg  (target: <= 2.5 deg)")
    print(f"  RMSE : {metrics['rmse']:.5f}  (target: <= 0.015 normalized)")

    trainer.save(args.output)

    if metrics['mgae'] <= 2.5:
        print("[PASS] MGAE target achieved")
    else:
        print("[FAIL] MGAE not achieved -- tune gamma / lr / epochs")


if __name__ == '__main__':
    main()
