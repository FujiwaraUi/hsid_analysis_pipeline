import os
import sys
import json
import pandas as pd

import numpy as np
import scipy
import scipy.io
import scipy.linalg
import tifffile as tif
import matplotlib.pyplot as plt
import spectral

from logging import getLogger, config

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def _rescale_0_1(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32, copy=False)
    vmin = np.nanmin(img2d)
    vmax = np.nanmax(img2d)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return np.zeros_like(img2d, dtype=np.float32)
    return (img2d - vmin) / (vmax - vmin)


def _fit_pca_basis(
    X_fit: np.ndarray,
    method: str = "svd",
    mean_centered: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X_fit: (N, B) float32/float64
    return: mean(B,), coeff(B,B), explained_ratio_percent(B,)
    """
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
        # Xc = U S Vt, principal axes = V (columns) = Vt.T
        U, S, Vt = scipy.linalg.svd(Xc, full_matrices=False, lapack_driver="gesdd")
        coeff = Vt.T  # (B,B)
        eigvals = (S ** 2) / (N - 1)  # (B,)
    elif method == "eig":
        # covariance-based eigen decomposition
        C = (Xc.T @ Xc) / (N - 1)  # (B,B)
        eigvals, eigvecs = scipy.linalg.eigh(C)  # ascending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        coeff = eigvecs[:, idx]
    else:
        raise ValueError(f"method must be 'svd' or 'eig', got '{method}'")

    total = np.sum(eigvals)
    if total <= 0 or not np.isfinite(total):
        explained = np.zeros_like(eigvals, dtype=np.float64)
    else:
        explained = (eigvals / total) * 100.0  # percent

    return mean.astype(np.float32, copy=False), coeff.astype(np.float32, copy=False), explained.astype(np.float64, copy=False)


def _project_cube_in_chunks(
    cube: np.ndarray,
    mean: np.ndarray,
    coeff_k: np.ndarray,
    mean_centered: bool = True,
    chunk_pixels: int = 200_000,
) -> np.ndarray:
    """
    cube: (H,W,B)
    mean: (B,)
    coeff_k: (B,K)
    return: (H,W,K)
    """
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
    cube: np.ndarray,
    num_components: int,
    outdir: str,
    method: str = "svd",
    mean_centered: bool = True,
    fit_max_pixels: int = 500_000,
    chunk_pixels: int = 200_000,
    logger=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MathWorks hyperpca 相当：バンド次元にPCAを実行して (H,W,K) を返す。
    結果は outdir/{png,txt,npy} に保存する。
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D (H,W,B), got {cube.ndim}D")
    H, W, B = cube.shape
    if not (1 <= num_components <= B):
        raise ValueError(f"num_components must be in [1, {B}], got {num_components}")

    os.makedirs(outdir, exist_ok=True)
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
    np.save(os.path.join(npy_dir, f"pca_mean.npy"), mean)

    np.savetxt(os.path.join(txt_dir, f"pca_explained_percent_k{num_components}.txt"), var_k, fmt="%.8f")
    np.savetxt(os.path.join(txt_dir, f"pca_coeff_k{num_components}.txt"), coeff_k, fmt="%.8e")

    # 可視化（PC1..PC3 をRGB合成、各PCをグレースケール）
    if num_components >= 3:
        rgb = np.stack(
            [
                _rescale_0_1(pca_cube[:, :, 0]),
                _rescale_0_1(pca_cube[:, :, 1]),
                _rescale_0_1(pca_cube[:, :, 2]),
            ],
            axis=-1,
        )
        plt.figure()
        plt.imshow(rgb)
        plt.axis("off")
        plt.tight_layout()
        # 変更前: f"pca_rgb_pc1-3_k{num_components}.png"
        plt.savefig(
            os.path.join(png_dir, f"pca_rgb_pc001-003_k{num_components}.png"),
            dpi=200,
        )
        plt.close()

    # --- each PC ---
    for i in range(min(num_components, 10)):
        img = _rescale_0_1(pca_cube[:, :, i])
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(png_dir, f"pca_pc{i+1:03d}_k{num_components}.png"),
            dpi=200,
        )
        plt.close()

    if logger:
        logger.debug(f"PCA done: cube(H,W,B)=({H},{W},{B}) -> pca_cube(H,W,K)=({H},{W},{num_components})")
        logger.debug(f"coeff shape: {coeff_k.shape}, var shape: {var_k.shape}")

    return pca_cube, coeff_k, var_k

def createPatches(X, y, removeZeroLabels = True):
    height = X.shape[0]
    width = X.shape[1]
    bands =  X.shape[2]

    patchesData = np.zeros((height * width, bands))
    patchesLabels = np.zeros((height * width))
    patchIndex = 0

    #  print(X.shape[0], X.shape[1], X.shape[2], height * width)

    for r in range(height):
        for c in range(width):
            patchesData[patchIndex, :] = X[r, c]
            patchesLabels[patchIndex] = y[r, c]
            patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRation=0.10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testRation, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

def make_img(X, y):
    height = X.shape[0]
    width = X.shape[1]
    outputs = np.zeros((height, width))

    os.makedirs("./data", exist_ok=True)

    for i in range(height):
        for j in range(width):
            target = y[i,j]
            if target == 0:
                continue
            else:
                # prediction := (1, 130) の配列に変換する
                prediction = clf.predict(X[i,j,:].reshape(1, -1))[0]
                outputs[i,j] = prediction+1

    # 正解画像
    #    ground_truth = spectral.imshow(classes = y,figsize =(10,10))
    # 機械学習画像
    #    predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(10,10))

    # 正解画像
    plt.figure(figsize=(10, 10))
    ground_truth = spectral.imshow(classes=y, figsize=(10, 10))
    plt.axis("off")
    plt.savefig("./data/gt.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 機械学習画像
    plt.figure(figsize=(10, 10))
    predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(10, 10))
    plt.axis("off")
    plt.savefig("./data/pr.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def print_d(i=0):
    print(f"This is point {i}")

if __name__ == "__main__":
    # set logger
    with open("./log_config.json", "r") as f:
        log_conf = json.load(f)

    config.dictConfig(log_conf)
    logger = getLogger(__name__)

    # output directory
    outdir = "./Data"
    os.makedirs(outdir, exist_ok=True)

    # data path
    data_path_1 = "/Volumes/ssd/HSID/data_10.1016/PaviaU"
    data_path_2 = "/Volumes/ssd/HSID/HISUI/HSHL1G_N329E1299_20230523072720_20240308144532.tif"

    hsid_hisui = tif.imread(data_path_2)

    # HSIデータ（PaviaU）
    hsid_paviau_x = scipy.io.loadmat(os.path.join(data_path_1, "PaviaU.mat"))["paviaU"]
    # ラベルデータ（PaviaU_gt）
    hsid_paviau_y = scipy.io.loadmat(os.path.join(data_path_1, "PaviaU_gt.mat"))["paviaU_gt"]

    logger.debug(f"PaviaU:{hsid_paviau_x.shape, type(hsid_paviau_x)}")
    logger.debug(f"PaviaU:{hsid_paviau_y.shape, type(hsid_paviau_y)}")
    logger.debug(f"HISUI X:{hsid_hisui.shape, type(hsid_hisui)}")

    # hyperpca 相当（例：K=10）
    pca_cube, coeff, var = mathworks_pca(
        cube=hsid_paviau_x,
        num_components=10,
        outdir=outdir,
        # method="svd",
        method="eig",
        mean_centered=True,
        logger=logger,
    )

    X = hsid_paviau_x
    y = hsid_paviau_y

    #Random Forest]    # 2次元画像を1次元ベクトルに変換
    XPatches, yPatches = createPatches(X, y)

    # 訓練用データとテスト用データに分割
    testRatio = 0.5
    X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)


    # 分類器はランダムフォレスト
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)


    class_name = ['Asphalt', 'Meadows', 'Gravel',
                  'Trees', 'Painted metal sheets', 'Bare Soil',
                  'Bitumen', 'Self-Blocking Bricks','Shadows']

    pre = clf.predict(X_test)


    ac_score = metrics.accuracy_score(y_test, pre)
    print('正解率:{0:.1f}%'.format(ac_score * 100))

    d = metrics.classification_report(y_test, pre, target_names=class_name, output_dict=True)
    df = pd.DataFrame(d)


    print(df)

    # CSV保存
    df.to_csv("data/classification_report.csv", index=True)

    cm = metrics.confusion_matrix(y_test, pre, normalize="all")   # 全要素の合計が1になる

    make_img(X,y)


