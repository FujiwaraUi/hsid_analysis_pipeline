import os
import scipy

import tifffile as tif

outdir = "./data"
os.makedirs(outdir, exist_ok=True)

data_path_1 = '/Volumes/ssd/HSID/data_10.1016/PaviaU'
data_path_2 = "/Volumes/ssd/HSID/HISUI/HSHL1G_N329E1299_20230523072720_20240308144532.tif"

hyper_cube = tif.imread(data_path_2)

# HSIデータ
X = scipy.io.loadmat(os.path.join(data_path_1, 'PaviaU.mat'))['paviaU']
# ラベルデータ
y =scipy.io.loadmat(os.path.join(data_path_1, 'PaviaU_gt.mat'))['paviaU_gt']
