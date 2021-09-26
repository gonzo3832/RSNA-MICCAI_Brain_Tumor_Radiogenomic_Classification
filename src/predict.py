import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import gc

def predict(model, device, test_loader):
    #model.to(device)    
    model.eval()
    pred_list = []
    for batch_idx,(data)in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(data))
        pred = pred.detach().cpu().numpy()    
        pred_list.append(pred)


    
    gc.collect()
    torch.cuda.empty_cache()
    del data

    return pred_list