import torch
# import random
#import math
import time
#from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

import torch.nn as nn
import numpy as np

def apply_VecNN_model(train_loader, validation_loader, VecNN_model, device, criterion, optimizer):

    VecNN_model.train()
    end = time.time() ##记录当前时间，用于记录前一个 batch 的结束时间，用于计算数据加载时间和批处理时间

    num_batches = len(train_loader)

    running_loss = 0.0 ##设置epoch的loss
    end = time.time() ##记录当前时间，用于记录前一个 batch 的结束时间，用于计算数据加载时间和批处理时间。
    for i, batch in enumerate(train_loader):

        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        #data_time = time.time() - end ##计算从上一个 batch 结束到当前 batch 开始所花费的时间，作为数据加载处理时间

        optimizer.zero_grad()
        pred = VecNN_model(batch_x).squeeze(-1)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        batch_time = time.time() - end ##计算运行当前batchsize的时间
        print(f"Step [{i+1}/{num_batches}] | "f"BatchLoss: {loss.item():.4f} | "f"BatchTime: {batch_time:.3f}s | ")
        end = time.time() ##更新end时间

    epoch_loss = running_loss / len(train_loader.dataset)

    ##验证集评估模型
    VecNN_model.eval()
    val_probs = []
    val_labels = []
    val_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(validation_loader):

            val_x, val_y = batch
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            pred = VecNN_model(val_x).squeeze(-1)
            loss = criterion(pred, val_y)
            val_loss += loss.item() * val_x.size(0)
            val_probs.extend(pred.cpu().numpy())
            val_labels.extend(val_y.cpu().numpy())

    val_loss = val_loss / len(validation_loader.dataset) # 平均Validation Loss
    # 转numpy
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    # 计算验证集的AUC
    val_auc = roc_auc_score(val_labels, val_probs)

    return epoch_loss, val_loss, val_auc

