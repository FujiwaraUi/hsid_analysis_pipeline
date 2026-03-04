import os
import sys
import scipy
import json
import tifffile as tif

from logging import getLogger, config

if __name__ == "__main__":
    # set logger
    with open('./log_config.json', 'r') as f:
        log_conf = json.load(f)
        
    config.dictConfig(log_conf)
    logger = getLogger(__name__)
    
    # output directo
    outdir = "./Data"
    os.makedirs(outdir, exist_ok=True)

    # data path
    data_path_1 = '/Volumes/ssd/HSID/data_10.1016/PaviaU'
    data_path_2 = "/Volumes/ssd/HSID/HISUI/HSHL1G_N329E1299_20230523072720_20240308144532.tif"
    
    hsid_paviau = tif.imread(data_path_2)
    
    # HSIデータ
    hsid_hisui_x = scipy.io.loadmat(os.path.join(data_path_1, 'PaviaU.mat'))['paviaU']
    # ラベルデータ
    hsid_hisui_y =scipy.io.loadmat(os.path.join(data_path_1, 'PaviaU_gt.mat'))['paviaU_gt']

    logger.debug(f"PaviaU:{hsid_paviau.shape, type(hsid_paviau)}")
    logger.debug(f"HISUI X:{hsid_hisui_x.shape, type(hsid_hisui_x)}")
    logger.debug(f"HISUI Y:{hsid_hisui_y.shape, type(hsid_hisui_y)}")

    
