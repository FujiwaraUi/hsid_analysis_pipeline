import json
from pathlib import Path
from logging import getLogger, config as log_config

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import spectral
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = [
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Painted metal sheets",
    "Bare Soil",
    "Bitumen",
    "Self-Blocking Bricks",
    "Shadows",
]


def mk_out(outdir):
    out = Path(outdir)
    png = out / "png"
    txt = out / "txt"
    npy = out / "npy"
    csv = out / "csv"
    for p in (out, png, txt, npy, csv):
        p.mkdir(parents=True, exist_ok=True)
    return out, png, txt, npy, csv


def mk_log(out):
    cfg = Path(__file__).resolve().parent / "log_config.json"
    with cfg.open("r", encoding="utf-8") as f:
        d = json.load(f)
    if "fileHandler" in d.get("handlers", {}):
        d["handlers"]["fileHandler"]["filename"] = str(Path(out) / "run.log")
    log_config.dictConfig(d)
    return getLogger(__name__)


def load(data_dir):
    data_dir = Path(data_dir)
    x = scipy.io.loadmat(data_dir / "PaviaU.mat")["paviaU"]
    y = scipy.io.loadmat(data_dir / "PaviaU_gt.mat")["paviaU_gt"]
    return x, y


def s01(a):
    a = a.astype(np.float32, copy=False)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo)


def pca_fit(x, method="svd", center=True):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.ndim}D")

    n, b = x.shape
    if n < 2:
        raise ValueError(f"Need at least 2 samples, got {n}")

    m = x.mean(axis=0) if center else np.zeros((b,), dtype=x.dtype)
    xc = x - m if center else x

    if method == "svd":
        _, s, vt = scipy.linalg.svd(xc, full_matrices=False, lapack_driver="gesdd")
        c = vt.T
        ev = (s ** 2) / (n - 1)
    elif method == "eig":
        cov = (xc.T @ xc) / (n - 1)
        ev, vec = scipy.linalg.eigh(cov)
        idx = np.argsort(ev)[::-1]
        ev = ev[idx]
        c = vec[:, idx]
    else:
        raise ValueError(f"bad method: {method}")

    tot = np.sum(ev)
    ex = np.zeros_like(ev, dtype=np.float64) if tot <= 0 or not np.isfinite(tot) else (ev / tot) * 100.0
    return m.astype(np.float32, copy=False), c.astype(np.float32, copy=False), ex.astype(np.float64, copy=False)


def pca_proj(cube, mean, coeff, center=True, chunk=200_000):
    h, w, b = cube.shape
    k = coeff.shape[1]
    x = cube.reshape(-1, b).astype(np.float32, copy=False)
    y = np.empty((x.shape[0], k), dtype=np.float32)
    for i in range(0, x.shape[0], chunk):
        j = min(i + chunk, x.shape[0])
        blk = x[i:j]
        if center:
            blk = blk - mean
        y[i:j] = blk @ coeff
    return y.reshape(h, w, k)


def pca_run(cube, outdir, k, method="svd", center=True, fit_max=200_000, chunk=200_000, log=None):
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got {cube.ndim}D")

    _, _, b = cube.shape
    if not 1 <= k <= b:
        raise ValueError(f"k must be in [1, {b}], got {k}")

    _, png, txt, npy, _ = mk_out(outdir)
    x = cube.reshape(-1, b).astype(np.float32, copy=False)
    n = x.shape[0]

    if n > fit_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=fit_max, replace=False)
        xf = x[idx]
        if log:
            log.debug("PCA fit uses subsampling: %d/%d pixels", fit_max, n)
    else:
        xf = x
        if log:
            log.debug("PCA fit uses all pixels: %d", n)

    mean, coeff, ex = pca_fit(xf, method=method, center=center)
    coeff = coeff[:, :k]
    ex = ex[:k]
    pc = pca_proj(cube, mean, coeff, center=center, chunk=chunk)

    np.save(npy / f"pca_cube_k{k}.npy", pc)
    np.save(npy / f"pca_coeff_k{k}.npy", coeff)
    np.save(npy / "pca_mean.npy", mean)
    np.savetxt(txt / f"pca_explained_percent_k{k}.txt", ex, fmt="%.8f")
    np.savetxt(txt / f"pca_coeff_k{k}.txt", coeff, fmt="%.8e")

    if k >= 3:
        rgb = np.stack((s01(pc[:, :, 0]), s01(pc[:, :, 1]), s01(pc[:, :, 2])), axis=-1)
        plt.figure()
        plt.imshow(rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png / f"pca_rgb_pc001-003_k{k}.png", dpi=200)
        plt.close()

    for i in range(min(k, 10)):
        plt.figure()
        plt.imshow(s01(pc[:, :, i]), cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png / f"pca_pc{i + 1:03d}_k{k}.png", dpi=200)
        plt.close()

    return pc


def patch(x, y, drop0=True):
    _, _, b = x.shape
    x2 = x.reshape(-1, b)
    y2 = y.reshape(-1)
    if drop0:
        m = y2 > 0
        x2 = x2[m]
        y2 = y2[m] - 1
    return x2, y2.astype(np.int64, copy=False)


def split(x, y, r=0.5):
    return train_test_split(x, y, test_size=r, random_state=0, stratify=y)


def rep(outdir, name, yt, yp, names):
    _, _, txt, _, csv = mk_out(outdir)
    acc = metrics.accuracy_score(yt, yp)
    df = pd.DataFrame(metrics.classification_report(yt, yp, target_names=names, output_dict=True))
    df.to_csv(csv / f"classification_report_{name}.csv", index=True)
    with (txt / f"accuracy_{name}.txt").open("w", encoding="utf-8") as f:
        f.write(f"accuracy={acc:.6f}\n")
    cm = metrics.confusion_matrix(yt, yp, normalize="all")
    np.savetxt(txt / f"confusion_matrix_{name}_normalize_all.txt", cm, fmt="%.8f")
    return acc


def img(outdir, name, x, y, clf):
    _, png, _, _, _ = mk_out(outdir)
    h, w, b = x.shape
    x2 = x.reshape(-1, b)
    y2 = y.reshape(-1)
    out = np.zeros((h * w,), dtype=np.int32)
    m = y2 > 0
    out[m] = clf.predict(x2[m]).astype(np.int32) + 1
    out = out.reshape(h, w)

    spectral.imshow(classes=y, figsize=(10, 10))
    plt.axis("off")
    plt.savefig(png / f"gt_{name}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    spectral.imshow(classes=out.astype(int), figsize=(10, 10))
    plt.axis("off")
    plt.savefig(png / f"pr_{name}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def run(
    outdir="./data",
    data_dir="/Volumes/ssd/HSID/data_10.1016/PaviaU",
    pca_on=False,
    k=10,
    pca_method="svd",
    center=True,
    test_ratio=0.5,
    rf_on=True,
    svm_on=True,
    names=CLASSES,
):
    out, _, _, _, _ = mk_out(outdir)
    log = mk_log(out)
    x, y = load(data_dir)

    log.debug("X: %s %s", x.shape, type(x))
    log.debug("y: %s %s", y.shape, type(y))

    if pca_on:
        xf = pca_run(x, out, k, method=pca_method, center=center, log=log)
    else:
        xf = x
        log.debug("PCA skipped: using original cube")

    xp, yp = patch(xf, y, drop0=True)
    xtr, xte, ytr, yte = split(xp, yp, r=test_ratio)

    if rf_on:
        rf = RandomForestClassifier(random_state=0, n_jobs=-1)
        rf.fit(xtr, ytr)
        pr = rf.predict(xte)
        acc = rep(out, "rf", yte, pr, names)
        img(out, "rf", xf, y, rf)
        print(f"RF  accuracy: {acc * 100:.2f}%")

    if svm_on:
        clf = make_pipeline(StandardScaler(), svm.SVC(kernel="rbf", gamma="scale", C=1.0))
        clf.fit(xtr, ytr)
        pr = clf.predict(xte)
        acc = rep(out, "svm", yte, pr, names)
        img(out, "svm", xf, y, clf)
        print(f"SVM accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    run()
