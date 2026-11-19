import sys
import torch
# import random
import math
import time
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

import torch.nn as nn
import numpy as np

def get_gnn_embeddings(gnn_model, nodes):
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        # print(f'打印nodes_batch: {nodes_batch}')
        # print(nodes_batch)
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    # print(embs.size())
    assert len(embs) == len(nodes)
    return embs.detach()

def get_unseen_gnn_embeddings(gnn_model, dataCenter, ds, adj_lists): ##Graphsage推测unseen节点的gnn embs

    unseen_nodes = getattr(dataCenter, ds + '_unseen')
    unseen_adj_lists = getattr(dataCenter, ds + '_unseen_adj_lists')
    for node, neighbors in unseen_adj_lists.items():
        if node in adj_lists:
            adj_lists[node].update(neighbors)
        else:
            adj_lists[node] = set(neighbors)
    b_sz = 200
    batches = math.ceil(len(unseen_nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = unseen_nodes[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch) # GraphSAGE forward
        assert len(embs_batch) == len(nodes_batch) # 确保输出节点数量和输入一致
        embs.append(embs_batch)

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    print(embs.size())
    assert len(embs) == len(unseen_nodes) ##确保最终 embedding 数量等于 unseen node 数量

    return embs.detach()


def apply_pregraphsage_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz, unsup_loss, device, learn_method, writer=None, epoch=None):
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


def apply_VecNN_model(train_loader, validation_loader, VecNN_model, device, criterion, optimizer, graphsage_embedding=0):

    VecNN_model.train()
    end = time.time() ##记录当前时间，用于记录前一个 batch 的结束时间，用于计算数据加载时间和批处理时间

    num_batches = len(train_loader)

    running_loss = 0.0 ##设置epoch的loss
    for i, batch in enumerate(train_loader):

        if graphsage_embedding == 1:
            batch_rna, batch_graph, batch_y = batch
            batch_graph = batch_graph.to(device)
        elif graphsage_embedding == 0:
            batch_rna, batch_y = batch
            batch_graph = None
        else:
            raise ValueError("graphsage_embedding 必须为 0 或 1")

        batch_rna = batch_rna.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = VecNN_model(batch_rna, batch_graph).squeeze(-1)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_rna.size(0)
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

            if graphsage_embedding == 1:
                val_rna, val_graph, val_y = batch
                val_graph = val_graph.to(device)
            elif graphsage_embedding == 0:
                val_rna, val_y = batch
                val_graph = None
            else:
                raise ValueError("graphsage_embedding 必须为 0 或 1")

            val_rna = val_rna.to(device)
            val_y = val_y.to(device)

            pred = VecNN_model(val_rna, val_graph).squeeze(-1)
            loss = criterion(pred, val_y)
            val_loss += loss.item() * val_rna.size(0)
            val_probs.extend(pred.cpu().numpy())
            val_labels.extend(val_y.cpu().numpy())

    val_loss = val_loss / len(validation_loader.dataset) # 平均Validation Loss
    # 转numpy
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    # 计算验证集的AUC
    val_auc = roc_auc_score(val_labels, val_probs)

    return epoch_loss, val_loss, val_auc

