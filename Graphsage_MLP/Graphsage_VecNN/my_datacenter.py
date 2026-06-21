# import sys
# import os
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold

class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_Graphsage_pretrain_dataSet(self, dataSet='all'):
        feat_file = self.config['file_path'][f'{dataSet}_feat']
        interaction_file = self.config['file_path'][f'{dataSet}_interaction']

        feat_data = []
        node_map = {}  # map node to Node_ID

        with open(feat_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data.append([float(x) for x in info[1:]])
                node_map[info[0]] = i
        feat_data = np.asarray(feat_data)
        edges_lists = []
        with open(interaction_file) as fp:
            for i, line in enumerate(fp):
                node1, node2 = line.strip().split()
                edges_lists.append((node1, node2))
        
        train_edge = np.array(edges_lists)
        train_indexs = np.empty(shape=(0,))
        for idx in range(len(train_edge)):
            edges_idx = train_edge[idx]
            node1 = edges_idx[0]
            node2 = edges_idx[1]
            node1_index = node_map[str(node1)]
            node2_index = node_map[str(node2)]
            train_indexs = np.append(train_indexs, int(node1_index))
            train_indexs = np.append(train_indexs, int(node2_index))
        train_indexs = train_indexs.astype(int)
        train_indexs = np.unique(train_indexs)
        setattr(self, dataSet + '_train', train_indexs)

        adj_lists = defaultdict(set)
        with open(interaction_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                assert len(info) == 2
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)

        setattr(self, dataSet + '_feats', feat_data)
        setattr(self, dataSet + '_adj_lists', adj_lists)

        return node_map, feat_data

    def load_VecNN_training_dataSet(self, dataSet='test_random'):
        feat_file = self.config['file_path'][f'{dataSet}_feat']
        label_file = self.config['file_path'][f'{dataSet}_label']
        graphsage_model_path = self.config['file_path'][f'{dataSet}_graphsage_model']

        setattr(self, dataSet + '_graphsage_model_path', graphsage_model_path)
        feat_data = []
        node_map = {}  # map node to Node_ID
        with open(feat_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data.append([float(x) for x in info[1:]])
                node_map[info[0]] = i
        feat_data = np.asarray(feat_data)
        setattr(self, dataSet + '_feats', feat_data)
        index_to_seq = {v: k for k, v in node_map.items()}

        label_df = pd.read_csv(label_file, sep='\t', header=None) ##读取带标签数据
        
        ##生成只有RNA的索引列表
        num_rnas = 11854 ##num_rnas = len(label_df)
        RNA_index_list = np.array(list(range(num_rnas)))
        setattr(self, dataSet + '_RNA_index_list', RNA_index_list)

        five_folds = self._split_rna_kfold(RNA_index_list=RNA_index_list, index_to_seq=index_to_seq, label_df=label_df, n_splits=5, random_state=42)
        #five_folds = self._split_rna_kfold(RNA_index_list, n_splits=5, random_state=42)

        setattr(self, dataSet + '_folds', five_folds)

        return node_map, index_to_seq, label_df

    def _split_rna_kfold(self, RNA_index_list, index_to_seq, label_df, n_splits=5, random_state=42):

        seq_col = label_df.columns[0]
        label_col = label_df.columns[1]
        seq_to_label = dict(zip(label_df[seq_col], label_df[label_col]))
        # 为每个RNA索引获得标签
        RNA_labels = []
        for idx in RNA_index_list:
            seq = index_to_seq[idx]
            if seq not in seq_to_label:
                raise ValueError(f"{seq} not found in label file")
            RNA_labels.append(seq_to_label[seq])

        RNA_labels = np.array(RNA_labels)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        folds = []
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(RNA_index_list, RNA_labels)):

            train_nodes = RNA_index_list[train_idx]
            test_nodes = RNA_index_list[test_idx]

            folds.append({"fold": fold_id, "train_nodes": train_nodes, "test_nodes": test_nodes})

        return folds
