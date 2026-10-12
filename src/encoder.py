"""
Lesion-grounded 2.5D image encoder (one per fold) and the visual-token cache for the language model.

The encoder is the 2.5D network of the companion CALM-Stroke study (ImageNet ResNet-18 per axial slice + gated
attention), trained here to predict the structured findings of make_reports.py (territories, side, volume class,
multiplicity, chronic infarct, diffusion restriction). Its direct predictions are the "direct classifier" baseline.

Outputs per fold k:
  results/vlm/direct_fold{k}.csv        direct structured predictions on held-out fold k
  <work>/vlm/feats_fold{k}.npy          (N_cache, S + 1, 512) float16 visual tokens for every scan (S slices + pooled)
  <work>/models/vlmenc_fold{k}.pt
Usage: python src/encoder.py [--folds 0 1 2 3 4] [--quick]
"""
import argparse, time
try:
    from PIL import Image  # noqa: F401  (on Windows, load Pillow before torch)
except Exception:
    pass
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from common import RES, WORK
from augment import augment

TERR = ["ACA", "MCA", "DEEP", "PCA", "THAL", "BS", "CEREB"]
SIDE = {"L": 0, "R": 1, "B": 2}


class DS(Dataset):
    def __init__(self, cache, rows, lab):
        self.cache, self.rows, self.lab = cache, np.asarray(rows), lab; self.X = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.X is None:
            self.X = np.load(self.cache / "X.npy", mmap_mode="r")
        r = int(self.rows[i]); L = self.lab
        return dict(x=torch.from_numpy(np.asarray(self.X[r], dtype=np.float32)), row=torch.tensor(r),
                    terr=torch.from_numpy(L["terr"][r]), side=torch.tensor(L["side"][r]), vol=torch.tensor(L["vol"][r]),
                    mult=torch.tensor(L["mult"][r]), chron=torch.tensor(L["chron"][r]), restr=torch.tensor(L["restr"][r]))


class GroundEncoder(nn.Module):
    def __init__(self, z_idx):
        super().__init__()
        import torchvision
        try:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print("ImageNet weights unavailable:", e); m = torchvision.models.resnet18()
        m.fc = nn.Identity(); self.enc = m; self.z = z_idx
        self.to3 = nn.Conv2d(2, 3, 1)
        with torch.no_grad():
            self.to3.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [.5, .5]]).view(3, 2, 1, 1)); self.to3.bias.zero_()
        self.pos = nn.Parameter(torch.zeros(len(z_idx), 512))           # slice-position embedding
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128); self.att_w = nn.Linear(128, 1)
        self.heads = nn.ModuleDict(dict(terr=nn.Linear(512, len(TERR)), side=nn.Linear(512, 3), vol=nn.Linear(512, 4),
                                        mult=nn.Linear(512, 1), chron=nn.Linear(512, 1), restr=nn.Linear(512, 3)))

    def tokens(self, x):
        x = x[:, :, :, :, self.z]; B, C, D, H, S = x.shape
        f = self.enc(self.to3(x.permute(0, 4, 1, 2, 3).reshape(B * S, C, D, H))).view(B, S, -1) + self.pos
        a = torch.softmax(self.att_w(torch.tanh(self.att_v(f)) * torch.sigmoid(self.att_u(f))), 1)
        h = (a * f).sum(1)
        return f, h

    def forward(self, x):
        f, h = self.tokens(x)
        return {k: m(h) for k, m in self.heads.items()}, f, h


def loss_fn(o, b):
    return (F.binary_cross_entropy_with_logits(o["terr"], b["terr"]) + F.cross_entropy(o["side"], b["side"])
            + F.cross_entropy(o["vol"], b["vol"]) + F.binary_cross_entropy_with_logits(o["mult"][:, 0], b["mult"])
            + F.binary_cross_entropy_with_logits(o["chron"][:, 0], b["chron"]) + F.cross_entropy(o["restr"], b["restr"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--epoch-size", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.epoch_size, a.patience, a.folds = 1, 16, 1, a.folds[:1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    cache = WORK / "cache"; out_w = WORK / "vlm"; out_w.mkdir(parents=True, exist_ok=True)
    (RES / "vlm").mkdir(parents=True, exist_ok=True); (WORK / "models").mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(cache / "index.csv"); N = len(idx)
    R = pd.read_csv(RES / "vlm" / "reports.csv")
    R = R.merge(idx[["row", "participant_id"]], on="participant_id")
    lab = dict(terr=np.zeros((N, len(TERR)), np.float32), side=np.zeros(N, np.int64), vol=np.zeros(N, np.int64),
               mult=np.zeros(N, np.float32), chron=np.zeros(N, np.float32), restr=np.zeros(N, np.int64))
    for _, r in R.iterrows():
        k = int(r.row)
        lab["terr"][k] = [r[f"terr_{t}"] for t in TERR]; lab["side"][k] = SIDE[r.side]; lab["vol"][k] = r.vol_bin
        lab["mult"][k] = r.multiple; lab["chron"][k] = r.chronic; lab["restr"][k] = r.restr
    nz = np.load(cache / "X.npy", mmap_mode="r").shape[-1]
    z_idx = sorted(set(np.linspace(0.17 * nz, 0.85 * nz, 17).astype(int).tolist()))
    kw = dict(num_workers=a.workers, pin_memory=amp, persistent_workers=a.workers > 0)

    for k in a.folds:
        t0 = time.time(); torch.manual_seed(k); rng = np.random.default_rng(k)
        tr_all = R[R.vlm_fold != k].row.values; te = R[R.vlm_fold == k].row.values
        va = rng.choice(tr_all, int(0.1 * len(tr_all)), replace=False); tr = np.setdiff1d(tr_all, va)
        dl_tr = DataLoader(DS(cache, tr, lab), batch_size=a.batch, drop_last=True,
                           sampler=WeightedRandomSampler(torch.ones(len(tr), dtype=torch.double), a.epoch_size), **kw)
        dl_va = DataLoader(DS(cache, va, lab), batch_size=a.batch, **kw)
        model = GroundEncoder(z_idx).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
        best, bad, state = 1e9, 0, None
        print(f"\n[vlm encoder] fold {k}: train {len(tr)}, val {len(va)}, test {len(te)}", flush=True)
        for ep in range(a.epochs):
            model.train(); tl = n = 0
            for b in dl_tr:
                b = {q: v.to(dev) for q, v in b.items()}
                b["x"], _ = augment(b["x"], torch.zeros_like(b["x"][:, :1]), 1.0)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    o, _, _ = model(b["x"])
                loss = loss_fn({q: v.float() for q, v in o.items()}, b)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item(); n += 1
            model.eval(); vl = vn = 0
            with torch.no_grad():
                for b in dl_va:
                    b = {q: v.to(dev) for q, v in b.items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        o, _, _ = model(b["x"])
                    vl += loss_fn({q: v.float() for q, v in o.items()}, b).item() * len(b["x"]); vn += len(b["x"])
            vl /= max(vn, 1)
            print(f"  ep {ep:2d} train {tl/max(n,1):.3f} | val {vl:.3f}", flush=True)
            if vl < best - 1e-4:
                best, bad, state = vl, 0, {q: v.detach().cpu().clone() for q, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(state); torch.save(state, WORK / "models" / f"vlmenc_fold{k}.pt"); model.eval()
        # visual tokens for every cached scan + direct predictions on the held-out fold
        S = len(z_idx); feats = np.lib.format.open_memmap(out_w / f"feats_fold{k}.npy", mode="w+", dtype=np.float16,
                                                           shape=(N, S + 1, 512))
        dl_all = DataLoader(DS(cache, np.arange(N), lab), batch_size=a.batch, **kw)
        te_set = set(te.tolist()); rows_out = []
        with torch.no_grad():
            for b in dl_all:
                x = b["x"].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    o, f, h = model(x)
                rr = b["row"].numpy()
                feats[rr] = torch.cat([f, h[:, None]], 1).float().cpu().numpy().astype(np.float16)
                for j, r in enumerate(rr):
                    if r in te_set:
                        d = dict(row=int(r))
                        d.update({f"p_terr_{t}": float(torch.sigmoid(o["terr"][j, i].float())) for i, t in enumerate(TERR)})
                        d.update({f"p_side_{s}": float(v) for s, v in zip("LRB", torch.softmax(o["side"][j].float(), 0))})
                        d.update({f"p_vol_{i}": float(v) for i, v in enumerate(torch.softmax(o["vol"][j].float(), 0))})
                        d["p_mult"] = float(torch.sigmoid(o["mult"][j, 0].float())); d["p_chron"] = float(torch.sigmoid(o["chron"][j, 0].float()))
                        d.update({f"p_restr_{i}": float(v) for i, v in enumerate(torch.softmax(o["restr"][j].float(), 0))})
                        rows_out.append(d)
        feats.flush()
        D = pd.DataFrame(rows_out).merge(idx[["row", "participant_id"]], on="row")
        D.to_csv(RES / "vlm" / f"direct_fold{k}.csv", index=False)
        print(f"[vlm encoder] fold {k} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
