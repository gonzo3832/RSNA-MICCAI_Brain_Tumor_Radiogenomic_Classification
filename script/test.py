import os
import sys 
import json
import glob
import random
import collections
import time
import re

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


data_directory = 'input/rsna-miccai-brain-tumor-radiogenomic-classification'
pytorch3dpath = "input/efficientnetpyttorch3d/EfficientNet-PyTorch-3D"

print(os.path.exists(pytorch3dpath))

mri_types = ['FLAIR','T1w','T1wCE','T2w']
SIZE = 256
NUM_IMAGES = 128 
sys.path.append(pytorch3dpath)
from efficientnet_pytorch_3d import EfficientNet3D



def load_dicom_image(path, img_size=SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    
    if rotate>0:
        rot_choices = [0,
                       cv2.ROTATE_90_CLOCKWISE, 
                       cv2.ROTATE_90_COUNTERCLOCKWISE,
                       cv2.ROTATE_180]
        data = cv2.rotate(data, rot_choices[rotate])
    data = cv2.resize(data, (img_size,img_size))
    # アウトプットは　H　* W 
    return data

def load_dicom_images_3d(scan_id, 
                         num_imgs=NUM_IMAGES, 
                         img_size=SIZE, 
                         mri_type='FLAIR',
                         split='train',
                         rotate=0):
    # mri_type フォルダ内の画像pathを昇順？にソートして取得
    files = sorted(glob.glob(f'{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm'),
                   key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    #　数字の昇順にsortされると思うが、なぜこれでsortできるかわからない...
    # 数字を返すべきところでfile pathを返しているように思う
    
    # ---------3D配列生成----------
    middle = len(files)//2
    middle_num_imgs = num_imgs//2
    p1 = max(0,middle - middle_num_imgs)
    p2 = min(len(files),middle + middle_num_imgs)
    
    # 総画像の中央から必要枚数分抜き出す
    # D * H *W * Cから　H * W * D に
    img3d = np.stack([load_dicom_image(f,rotate=rotate)for f in files[p1:p2]]).transpose(1,2,0)
    
    # 総画像が必要枚数（num_imgs)無かった場合,空の配列で補完
    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
        img3d = np.concatenate((img3d, n_zero),axis = -1)
    
    #0-1で正規化    
    img3d = img3d - np.min(img3d)
    img3d = img3d / np.max(img3d)
    
    # チャネルの次元を加えてあげてreturn(モノクロ画像なので一次元)
    return np.expand_dims(img3d,0)

    
#　症例00000に対してテスト
test_2d = load_dicom_image(glob.glob(f'{data_directory}/train/00000/FLAIR/*.dcm')[0])
print(test_2d.shape)
plt.imshow(test_2d)
test_3d = load_dicom_images_3d('00000')
print(test_3d.shape)
