import sys
# import os
import torch
# import random
import math
import time

from sklearn.utils import shuffle
# from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np

def apply_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz, unsup_loss, device, learn_method, writer=None, epoch=None):
    train_nodes = getattr(dataCenter, ds+'_train')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSage]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
    #print(params)
    optimizer = torch.optim.SGD(params, lr=0.65)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    running_loss = 0.0 ##设置epoch的loss
    total_data_time = 0.0
    total_batch_time = 0.0
    end = time.time() ##记录当前时间，用于记录前一个 batch 的结束时间，用于计算数据加载时间和批处理时间。

    for index in range(batches):
        nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)
        embs_batch = graphSage(nodes_batch)
        #print("emb std:",torch.std(embs_batch).item())
        data_time = time.time() - end ##计算从上一个 batch 结束到当前 batch 开始所花费的时间，作为数据加载处理时间

        if learn_method == 'sup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
            #print(embs_batch)

        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes))) ##反向传播更新参数,打印训练进度信息
        running_loss += loss.item() * len(train_nodes[index*b_sz:(index+1)*b_sz])
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

        batch_time = time.time() - end ##计算运行当前batchsize的时间
        end = time.time() ##生成当前时间节点
        total_data_time += data_time
        total_batch_time += batch_time

    epoch_loss = running_loss / len(train_nodes)
    if writer is not None:
        writer.add_scalar('Train/EpochLoss', epoch_loss, epoch + 1)
        writer.add_scalar('Train/DataTime', total_data_time / batches, epoch + 1)
        writer.add_scalar('Train/BatchTime', total_batch_time / batches, epoch + 1)
        
    return graphSage, epoch_loss
