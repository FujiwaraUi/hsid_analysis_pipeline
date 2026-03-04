import os
import json
import numpy as np
import scipy.io
import scipy.linalg
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
import spectral

from logging import getLogger, config

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _rescale_0_1(img2d):
    img2d = img2d.astype(np.float32, copy=False)
    vmin = np.nanmin(img2d)
    vmax = np.nanmax(img2d)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return np.zeros_like(img2d, dtype=np.float32)
    return (img2d - vmin) / (vmax - vmin)


def _fit_pca_basis(X_fit, method="svd", mean_centered=True):
    X_fit = np.asarray(X_fit)
    if X_fit.ndim != 2:
        raise ValueError(f"X_fit must be 2D, got {X_fit.ndim}D")

    N, B = X_fit.shape
    if N < 2:
        raise ValueError(f"Need at least 2 samples, got N={N}")

    mean = X_fit.mean(axis=0) if mean_centered else np.zeros((B,), dtype=X_fit.dtype)
    Xc = X_fit - mean if mean_centered else X_fit

    method = method.lower()
    if method == "svd":
        U, S, Vt = scipy.linalg.svd(Xc, full_matrices=False, lapack_driver="gesdd")
        coeff = Vt.T
        eigvals = (S ** 2) / (N - 1)
    elif method == "eig":
        C = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs = scipy.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        coeff = eigvecs[:, idx]
    else:
        raise ValueError(f"method must be 'svd' or 'eig', got '{method}'")

    total = np.sum(eigvals)
    if total <= 0 or not np.isfinite(total):
        explained = np.zeros_like(eigvals, dtype=np.float64)
    else:
        explained = (eigvals / total) * 100.0

    return mean.astype(np.float32, copy=False), coeff.astype(np.float32, copy=False), explained.astype(np.float64, copy=False)


def _project_cube_in_chunks(cube, mean, coeff_k, mean_centered=True, chunk_pixels=200_000):
    H, W, B = cube.shape
    K = coeff_k.shape[1]
    X = cube.reshape(-1, B).astype(np.float32, copy=False)

    Y = np.empty((X.shape[0], K), dtype=np.float32)
    for s in range(0, X.shape[0], chunk_pixels):
        e = min(s + chunk_pixels, X.shape[0])
        Xblk = X[s:e]
        if mean_centered:
            Xblk = Xblk - mean
        Y[s:e] = Xblk @ coeff_k
    return Y.reshape(H, W, K)


def mathworks_pca(
    cube,
    num_components,
    outdir,
    method="svd",
    mean_centered=True,
    fit_max_pixels=200_000,
    chunk_pixels=200_000,
    logger=None,
):
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D (H,W,B), got {cube.ndim}D")
    H, W, B = cube.shape
    if not (1 <= num_components <= B):
        raise ValueError(f"num_components must be in [1, {B}], got {num_components}")

    png_dir = os.path.join(outdir, "png")
    txt_dir = os.path.join(outdir, "txt")
    npy_dir = os.path.join(outdir, "npy")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    X_all = cube.reshape(-1, B).astype(np.float32, copy=False)
    n_pixels = X_all.shape[0]

    if n_pixels > fit_max_pixels:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_pixels, size=fit_max_pixels, replace=False)
        X_fit = X_all[idx]
        if logger:
            logger.debug(f"PCA fit uses subsampling: {fit_max_pixels}/{n_pixels} pixels")
    else:
        X_fit = X_all
        if logger:
            logger.debug(f"PCA fit uses all pixels: {n_pixels}")

    mean, coeff_full, explained_full = _fit_pca_basis(X_fit, method=method, mean_centered=mean_centered)
    coeff_k = coeff_full[:, :num_components]
    var_k = explained_full[:num_components]

    pca_cube = _project_cube_in_chunks(
        cube=cube,
        mean=mean,
        coeff_k=coeff_k,
        mean_centered=mean_centered,
        chunk_pixels=chunk_pixels,
    )

    np.save(os.path.join(npy_dir, f"pca_cube_k{num_components}.npy"), pca_cube)
    np.save(os.path.join(npy_dir, f"pca_coeff_k{num_components}.npy"), coeff_k)
    np.save(os.path.join(npy_dir, "pca_mean.npy"), mean)

    np.savetxt(os.path.join(txt_dir, f"pca_explained_percent_k{num_components}.txt"), var_k, fmt="%.8f")
    np.savetxt(os.path.join(txt_dir, f"pca_coeff_k{num_components}.txt"), coeff_k, fmt="%.8e")

    if num_components >= 3:
        rgb = np.stack(
            [_rescale_0_1(pca_cube[:, :, 0]), _rescale_0_1(pca_cube[:, :, 1]), _rescale_0_1(pca_cube[:, :, 2])],
            axis=-1,
        )
        plt.figure()
        plt.imshow(rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(png_dir, f"pca_rgb_pc001-003_k{num_components}.png"), dpi=200)
        plt.close()

    for i in range(min(num_components, 10)):
        img = _rescale_0_1(pca_cube[:, :, i])
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(png_dir, f"pca_pc{i+1:03d}_k{num_components}.png"), dpi=200)
        plt.close()

    if logger:
        logger.debug(f"PCA done: ({H},{W},{B}) -> ({H},{W},{num_components})")
        logger.debug(f"coeff: {coeff_k.shape}, var: {var_k.shape}")

    return pca_cube, coeff_k, var_k


def create_patches(X_cube, y_gt, removeZeroLabels=True):
    H, W, B = X_cube.shape
    X2 = X_cube.reshape(-1, B)
    y1 = y_gt.reshape(-1)

    if removeZeroLabels:
        m = y1 > 0
        X2 = X2[m]
        y1 = y1[m] - 1
    return X2, y1.astype(np.int64, copy=False)


def split_train_test(X, y, testRatio=0.5):
    return train_test_split(X, y, test_size=testRatio, random_state=0, stratify=y)


def save_report(outdir, prefix, y_test, y_pred, class_name):
    os.makedirs(os.path.join(outdir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "txt"), exist_ok=True)

    acc = metrics.accuracy_score(y_test, y_pred)
    rep = metrics.classification_report(y_test, y_pred, target_names=class_name, output_dict=True)
    df = pd.DataFrame(rep)

    df.to_csv(os.path.join(outdir, "csv", f"classification_report_{prefix}.csv"), index=True)
    with open(os.path.join(outdir, "txt", f"accuracy_{prefix}.txt"), "w") as f:
        f.write(f"accuracy={acc:.6f}\n")

    cm = metrics.confusion_matrix(y_test, y_pred, normalize="all")
    np.savetxt(os.path.join(outdir, "txt", f"confusion_matrix_{prefix}_normalize_all.txt"), cm, fmt="%.8f")

    return acc


def save_pred_images(outdir, prefix, X_cube, y_gt, clf):
    os.makedirs(os.path.join(outdir, "png"), exist_ok=True)

    H, W, B = X_cube.shape
    X2 = X_cube.reshape(-1, B)
    y1 = y_gt.reshape(-1)

    out = np.zeros((H * W,), dtype=np.int32)
    m = y1 > 0
    pred = clf.predict(X2[m])
    out[m] = pred.astype(np.int32) + 1
    out = out.reshape(H, W)

    spectral.imshow(classes=y_gt, figsize=(10, 10))
    plt.axis("off")
    plt.savefig(os.path.join(outdir, "png", f"gt_{prefix}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    spectral.imshow(classes=out.astype(int), figsize=(10, 10))
    plt.axis("off")
    plt.savefig(os.path.join(outdir, "png", f"pr_{prefix}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    outdir = "./data"
    os.makedirs(outdir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_config_path = os.path.join(script_dir, "log_config.json")
    if not os.path.exists(log_config_path):
        # fallback to current working directory for backwards compatibility
        log_config_path = os.path.join(os.getcwd(), "log_config.json")
    with open(log_config_path, "r") as f:
        log_conf = json.load(f)
    if "handlers" in log_conf and "fileHandler" in log_conf["handlers"]:
        log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(outdir, "run.log")
    config.dictConfig(log_conf)
    logger = getLogger(__name__)

    data_path = "/Volumes/ssd/HSID/data_10.1016/PaviaU"
    X = scipy.io.loadmat(os.path.join(data_path, "PaviaU.mat"))["paviaU"]
    y = scipy.io.loadmat(os.path.join(data_path, "PaviaU_gt.mat"))["paviaU_gt"]

    logger.debug(f"X: {X.shape} {type(X)}")
    logger.debug(f"y: {y.shape} {type(y)}")

    K = 103

    X_pca, _, _ = mathworks_pca(
        cube=X,
        num_components=K,
        outdir=outdir,
        method="svd",
        mean_centered=True,
        logger=logger,
    )

    class_name = [
        "Asphalt", "Meadows", "Gravel",
        "Trees", "Painted metal sheets", "Bare Soil",
        "Bitumen", "Self-Blocking Bricks", "Shadows",
    ]
    
    XPatches, yPatches = create_patches(X, y, removeZeroLabels=True)
    # XPatches, yPatches = create_patches(X_pca, y, removeZeroLabels=True)
    X_train, X_test, y_train, y_test = split_train_test(XPatches, yPatches, testRatio=0.5)

    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    pre_rf = rf.predict(X_test)
    acc_rf = save_report(outdir, "rf", y_test, pre_rf, class_name)
    save_pred_images(outdir, "rf", X_pca, y, rf)

    svm_clf = make_pipeline(StandardScaler(), svm.SVC(kernel="rbf", gamma="scale", C=1.0))
    svm_clf.fit(X_train, y_train)
    pre_svm = svm_clf.predict(X_test)
    acc_svm = save_report(outdir, "svm", y_test, pre_svm, class_name)
    save_pred_images(outdir, "svm", X_pca, y, svm_clf)

    print(f"RF  accuracy: {acc_rf*100:.2f}%")
    print(f"SVM accuracy: {acc_svm*100:.2f}%")
