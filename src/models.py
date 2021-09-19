import os
import torch
import torch.nn as nn
import sys
pytorch3dpath = "input/efficientnetpyttorch3d/EfficientNet-PyTorch-3D"
print(os.path.exists(pytorch3dpath))
sys.path.append(pytorch3dpath)
from efficientnet_pytorch_3d import EfficientNet3D

def get_model(config):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    model = eval(model_name)(model_params)
    # eval関数　：　文字列をpythonのコードとして実行する
    # modelのインスタンス化してることになる
    return model

####################################################################
#     Resnet50
####################################################################
class EffNet3D(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.net = EfficientNet3D.from_name(
            'efficientnet-b0',
            override_params = {'num_classes':params['num_classes']}, 
            in_channels = 1)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features = n_features,out_features=1,bias=True)
        

    def forward(self,x):
        out = self.net(x)
#        out = nn.Sigmoid()(out)
        return out


