# import sys
import os
import numpy as np
import pandas as pd
import pickle

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold

class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_Graphsage_pretrain_dataSet(self, dataSet='LLPS_benchmark'):
        rna_feat_pkl_file = self.config['file_path'][f'{dataSet}_rna_feat']
        protein_feat_pkl_file = self.config['file_path'][f'{dataSet}_protein_feat']
        interaction_file = self.config['file_path'][f'{dataSet}_interaction']

        with open(rna_feat_pkl_file, "rb") as f:
            rna_feat_dataframe = pickle.load(f)
        rna_embeddings = np.stack(rna_feat_dataframe["embedding"].values)

        with open(protein_feat_pkl_file, "rb") as f:
            protein_feat_dataframe = pickle.load(f)
        protein_embeddings = np.stack(protein_feat_dataframe["embedding"].values)

        ## Normalize embeddings ##
        ## rna
        mean_rna_embeddings = np.mean(rna_embeddings, axis=0) ##对所有RNA计算平均 embedding
        centered_rna_embeddings = rna_embeddings - mean_rna_embeddings ##中心化
        centered_rna_embeddings_length = np.mean(np.sqrt(np.sum(centered_rna_embeddings * centered_rna_embeddings, axis=1))) ##计算平均向量长度，长度指的是这个点距离原点 $(0,0,0...)$ 的几何距离
        normalized_rna_embeddings = centered_rna_embeddings / np.expand_dims(centered_rna_embeddings_length, axis=-1) ##特征缩放,将所有中心化后的 RNA 向量调整到一个统一的数值量级,normalization

        #rna_sequences = rna_feat_dataframe["RNA_sequence"].values.reshape(-1, 1)
        target_dim = 1024
        current_dim = normalized_rna_embeddings.shape[1]
        padding_dim = target_dim - current_dim
        normalized_rna_embeddings_padded = np.pad(normalized_rna_embeddings,pad_width=((0, 0), (0, padding_dim)),mode="constant",constant_values=0.0).astype(np.float32)

        rna_sequences = (rna_feat_dataframe["RNA_sequence"].str.replace("U", "T").values.reshape(-1, 1))
        rna_feats = np.column_stack((rna_sequences, normalized_rna_embeddings_padded))

        ## protein
        mean_protein_embeddings = np.mean(protein_embeddings, axis = 0)
        centered_protein_embeddings = protein_embeddings - mean_protein_embeddings
        centered_protein_embeddings_length = np.mean(np.sqrt(np.sum(centered_protein_embeddings * centered_protein_embeddings, axis = 1)))
        normalized_protein_embeddings = centered_protein_embeddings / np.expand_dims(centered_protein_embeddings_length, axis = -1)

        protein_sequences = protein_feat_dataframe["Protein_sequence"].values.reshape(-1, 1)
        protein_feats = np.column_stack((protein_sequences, normalized_protein_embeddings))

        all_feats = np.vstack((rna_feats,protein_feats))
        
        workdir = self.config['file_path']['workdir']
        normalization_npz_file = os.path.join(workdir, 'normalization.npz')
        output_feats_file = os.path.join(workdir, dataSet, f'{dataSet}_all_feats.txt')

        if os.path.exists(normalization_npz_file) and os.path.exists(output_feats_file):
            print("Normalization file and feature file already exist.")
        else:
            np.savez(normalization_npz_file, mean_rna_embeddings=mean_rna_embeddings, centered_rna_embeddings_length=centered_rna_embeddings_length, mean_protein_embeddings=mean_protein_embeddings, centered_protein_embeddings_length=centered_protein_embeddings_length)
            np.savetxt(output_feats_file, all_feats, delimiter="\t", fmt="%s")
            print("Saved normalization and feature files:")
            print(f"  {normalization_npz_file}")
            print(f"  {output_feats_file}")

        feat_data = []
        node_map = {}  # map node to Node_ID
        for i, row in enumerate(all_feats):
            node_map[row[0]] = i
            feat_data.append([float(x) for x in row[1:]])

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

    def load_benchmark_dataSet(self, dataSet='LLPS_benchmark'):
        feat_file = self.config['file_path'][f'{dataSet}_all_feat'] ##seq + graph embs
        rna_feat_pkl_file = self.config['file_path'][f'{dataSet}_rna_hat_feat'] ## RNA_sequence + HAT embedding
        label_file = self.config['file_path'][f'{dataSet}_label']

        with open(rna_feat_pkl_file, "rb") as f:
            rna_feat_dataframe = pickle.load(f)
        rna_embeddings = np.stack(rna_feat_dataframe["embedding"].values)

        ## Normalize embeddings ##
        ## rna
        mean_rna_embeddings = np.mean(rna_embeddings, axis=0) ##对所有RNA计算平均 embedding
        centered_rna_embeddings = rna_embeddings - mean_rna_embeddings ##中心化
        centered_rna_embeddings_length = np.mean(np.sqrt(np.sum(centered_rna_embeddings * centered_rna_embeddings, axis=1))) ##计算平均向量长度，长度指的是这个点距离原点 $(0,0,0...)$ 的几何距离
        normalized_rna_embeddings = (centered_rna_embeddings / np.expand_dims(centered_rna_embeddings_length, axis=-1)).astype(np.float32) ##特征缩放,将所有中心化后的 RNA 向量调整到一个统一的数值量级,normalization

        rna_sequences = (rna_feat_dataframe["RNA_sequence"].str.replace("U", "T").values.reshape(-1, 1))
        rna_feats = np.column_stack((rna_sequences, normalized_rna_embeddings))

        workdir = self.config['file_path']['workdir']
        normalization_npz_file = os.path.join(workdir, 'hat_rnafm_embeddings_normalization.npz')

        if os.path.exists(normalization_npz_file):
            print("Normalization file and feature file already exist.")
        else:
            np.savez(normalization_npz_file, mean_rna_embeddings=mean_rna_embeddings, centered_rna_embeddings_length=centered_rna_embeddings_length)
            print("Saved normalization file:")
            print(f"  {normalization_npz_file}")

        feat_data = []
        node_map = {}  # map node to Node_ID
        with open(feat_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data.append([float(x) for x in info[1:]])
                node_map[info[0]] = i
        feat_data = np.asarray(feat_data) ##只有 embedding 数值
        #setattr(self, dataSet + '_feats', feat_data)
        setattr(self, dataSet + '_rna_hat_feats', rna_feats)
        index_to_seq = {v: k for k, v in node_map.items()}

        label_df = pd.read_csv(label_file, sep='\t', header=None) ##读取带标签数据
        
        ##生成只有RNA的索引列表
        ##num_rnas = 11854 
        num_rnas = len(label_df)
        print("num_rnas =", num_rnas)
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

    def load_unseen_dataSet(self, dataSet='LLPS_unseen', adj_lists=None, node_map=None, feat_data=None, graphsage_embedding=0): ##adj_lists：训练好的 GraphSAGE 图的邻接表；node_map：RNA/protein 名称 → node index 的映射；feat_data：原来训练集节点的 feature matrix
        rna_unseen_feat_pkl_file = self.config['file_path'][f'{dataSet}_rna_hat_feat']
        #protein_unseen_feat_pkl_file = self.config['file_path'][f'{dataSet}_protein_feat']
        normalization_npz = self.config['file_path']['hat_rnafm_embeddings_normalization_npz']
        unseen_label_file = self.config['file_path'][f'{dataSet}_label']
        ## 读取文件
        with open(rna_unseen_feat_pkl_file, "rb") as f:
            rna_unseen_feat_dataframe = pickle.load(f)
        unseen_rna_embeddings = np.stack(rna_unseen_feat_dataframe["embedding"].values)

        unseen_label_df = pd.read_csv(unseen_label_file, sep='\t', header=None) ##读取带标签数据

        normalization_data = np.load(normalization_npz)
        mean_rna_embeddings = normalization_data["mean_rna_embeddings"]
        centered_rna_embeddings_length = normalization_data["centered_rna_embeddings_length"]
        centered_unseen_rna_embeddings = (unseen_rna_embeddings - mean_rna_embeddings)
        normalized_unseen_rna_embeddings = (centered_unseen_rna_embeddings / centered_rna_embeddings_length)

        #unseen_feat_data = normalized_unseen_rna_embeddings[:, :640].astype(np.float32)
        unseen_rna_sequences = (rna_unseen_feat_dataframe["RNA_sequence"].str.replace("U", "T").values.reshape(-1, 1))

        if graphsage_embedding == 1:
            target_dim = 1024
            current_dim = normalized_unseen_rna_embeddings.shape[1]
            padding_dim = target_dim - current_dim
            normalized_unseen_rna_embeddings_padded = np.pad(normalized_unseen_rna_embeddings, pad_width=((0, 0), (0, padding_dim)), mode="constant", constant_values=0.0).astype(np.float32)
            unseen_rna_feats = np.column_stack((unseen_rna_sequences, normalized_unseen_rna_embeddings_padded))

            ##unseen节点加入训练图
            adj_lists_keys = adj_lists.keys() ## 获取训练图中已有节点的 index
            max_node_index = max(adj_lists_keys)  # 找到原训练图最大的 node index

            new_node_count = 0
            unseen_indexs = []
            unseen_feat_data = []

            for i, row in enumerate(unseen_rna_feats):
                node_name = row[0]
                if node_name in node_map:
                    continue
                feat_data.append([float(x) for x in row[1:]])
                unseen_feat_data.append([float(x) for x in row[1:641]])
                new_node_index = (max_node_index + 1 + new_node_count)
                node_map[node_name] = new_node_index
                unseen_indexs.append(new_node_index)
                new_node_count += 1

            unseen_interaction_file = self.config['file_path'][f'{dataSet}_interaction']
            unseen_adj_lists = defaultdict(set)
            with open(unseen_interaction_file) as fp:
                for line in fp:
                    info = line.strip().split()
                    node1_name = info[0]
                    node2_name = info[1]

                    node1 = node_map[node1_name]
                    node2 = node_map[node2_name]

                    unseen_adj_lists[node1].add(node2)
                    unseen_adj_lists[node2].add(node1)

            feat_data = np.asarray(feat_data) ##转换成 NumPy array
            unseen_feat_data = np.asarray(unseen_feat_data)
            unseen_indexs = np.asarray(unseen_indexs, dtype=int)
            setattr(self, dataSet + '_unseen', unseen_indexs) ##保存 unseen node index
            setattr(self, dataSet + '_unseen_adj_lists', unseen_adj_lists)

        elif graphsage_embedding == 0:
            unseen_feat_data = normalized_unseen_rna_embeddings.astype(np.float32) ## 不使用 GraphSAGE：不 padding、不构建图
            unseen_indexs = np.arange(len(unseen_rna_sequences), dtype=int) ## 直接使用 RNA 的行号作为 unseen index

            node_map = {seq: idx for idx, seq in enumerate(unseen_rna_sequences.flatten())} ## 保持与 get_unseen_graphsage_embeddings() 的映射格式一致
            feat_data = None ## 没有 GraphSAGE 特征矩阵

            setattr(self, dataSet + '_unseen', unseen_indexs)

        else:
            raise ValueError("graphsage_embedding 必须为 0 或 1")

        return unseen_indexs, feat_data, unseen_feat_data, unseen_label_df, node_map ## 返回新节点和组合的features

