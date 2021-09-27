import os
import cv2
import hydra
import pickle
import random
import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

import glob
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import re

def _load_dicom_image(path, img_size, voi_lut=True, rotation=0):
    if path.split('.')[-1] == 'dcm':
        dicom = pydicom.read_file(path)
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
    else:
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    
    if rotation>0:
        rot_choices = [0,
                       cv2.ROTATE_90_CLOCKWISE, 
                       cv2.ROTATE_90_COUNTERCLOCKWISE,
                       cv2.ROTATE_180]
        data = cv2.rotate(data, rot_choices[rotation])
    data = cv2.resize(data, (img_size,img_size))
    # アウトプットは　H　* W 
    return data
'''
import matplotlib.pyplot as plt
from IPython.display import Image
cv2.imshow(
    'test',
    _load_dicom_image(
    path='./input/train/00000/FLAIR/Image-117.png',
    img_size=256)
    )
cv2.waitkey()
cv2.destroyAllWindows()
'''


def _load_dicom_images_3d(
                         datadir,
                         scan_id, 
                         num_imgs, 
                         img_size, 
                         MRI_Type,
                         rotation):
    # MRI_Type フォルダ内の画像pathを昇順？にソートして取得
    files = sorted(glob.glob(f'{datadir}/{scan_id}/{MRI_Type}/*.*'),
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
    img3d = np.stack([_load_dicom_image(f,img_size = img_size,rotation=rotation)for f in files[p1:p2]]).transpose(1,2,0)    
    # 総画像が必要枚数（num_imgs)無かった場合,空の配列で補完
    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
        img3d = np.concatenate((img3d, n_zero),axis = -1)
    
    #0-1で正規化    
    img3d = img3d - np.min(img3d)
    img3d = img3d / np.max(img3d)
    
    # チャネルの次元を加えてあげてreturn(モノクロ画像なので一次元)
    return np.expand_dims(img3d,0)

#test_3d = _load_dicom_images_3d('/workspace/input/train',str(457).zfill(5),60,256,'FLAIR',0)

class DefaultDataset(data.Dataset):
    def __init__ (self,
                 df: pd.DataFrame,
                 datadir,
                 phase: str,
                 MRItype: str,
                 config={},
                 ):
        self.df = df
        self.datadir = datadir
        self.config = config
        self.phase = phase
        self.MRItype = MRItype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        scan_id = self.df['BraTS21ID'][idx]
        if self.phase == 'train':
            if self.config['aug']:
                rotation = np.random.randint(0,4)
            else:
                rotation = 0

            data = _load_dicom_images_3d(
                datadir = self.datadir,
                scan_id = str(int(scan_id)).zfill(5),
                num_imgs = self.config['num_imgs'],
                img_size = self.config['img_size'],
                MRI_Type = self.MRItype,
                rotation = rotation
            )
            data = torch.tensor(data).float()
            
            target = self.df['MGMT_value'][idx]
            target = torch.tensor(target).float()
            return data, target

        elif self.phase == 'valid':
            data = _load_dicom_images_3d(
                datadir = self.datadir,
                scan_id = str(int(scan_id)).zfill(5),
                num_imgs = self.config['num_imgs'],
                img_size = self.config['img_size'], 
                MRI_Type = self.MRItype,
                rotation = 0
            )
            data = torch.tensor(data).float()
            target = self.df['MGMT_value'][idx]
            target = torch.tensor(target).float()
            

            return data,target
        else:
            data = _load_dicom_images_3d(
                datadir = self.datadir,
                scan_id = str(int(scan_id)).zfill(5),
                num_imgs = self.config['num_imgs'],
                img_size = self.config['img_size'], 
                MRI_Type = self.MRItype,
                rotation = 0
            )
            data = torch.tensor(data).float()

            return data