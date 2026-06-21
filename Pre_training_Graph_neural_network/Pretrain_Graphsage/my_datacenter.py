# import sys
# import os
from collections import defaultdict
import numpy as np
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

    # def load_dataSet(self, dataSet='test_random'):
    #     feat_file = self.config['file_path'][f'{dataSet}_feat']
    #     interaction_file = self.config['file_path'][f'{dataSet}_interaction']

    #     feat_data = []
    #     node_map = {}  # map node to Node_ID

    #     feat_map = {}  # map feat to Node_ID
    #     with open(feat_file) as fp:
    #         for i, line in enumerate(fp):
    #             info = line.strip().split()
    #             feat_data.append([float(x) for x in info[1:]])
    #             node_map[info[0]] = i
    #             feat_map[info[0]] = i
    #     feat_data = np.asarray(feat_data)
    #     edges_lists = []
    #     with open(interaction_file) as fp:
    #         for i, line in enumerate(fp):
    #             node1, node2 = line.strip().split()
    #             edges_lists.append((node1, node2))

    #     folds = self._split_data_edge_kfold(edges_lists, node_map)

    #     setattr(self, dataSet + '_folds', folds)

    #     # adj_lists = defaultdict(set)
    #     # with open(interaction_file) as fp:
    #     #     for i, line in enumerate(fp):
    #     #         info = line.strip().split()
    #     #         assert len(info) == 2
    #     #         paper1 = node_map[info[0]]
    #     #         paper2 = node_map[info[1]]
    #     #         adj_lists[paper1].add(paper2)
    #     #         adj_lists[paper2].add(paper1)

    #     setattr(self, dataSet + '_feats', feat_data)
    #     #setattr(self, dataSet + '_adj_lists', adj_lists)

    #     return node_map, feat_data

    # def _split_data_edge_kfold(self, edges_lists, node_map, n_splits=5, random_state=42):
    #     edges_array = np.array(edges_lists) ##edges_lists是RNA和蛋白质的序列对，转成numpy数组方便切片
    #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) ##n_splits:折数，比如 5 折就把数据分成 5 份,shuffle:划分前先随机打乱数据顺序,random_state:打乱时使用的随机种子，确保可复现
    #     folds = []
    #     for fold_id, (train_idx, test_idx) in enumerate(kf.split(edges_array)): ##kf.split(edges_array):返回一个生成器，生成每个折的训练集和测试集的索引,fold_id 是当前折的编号（0、1、2、3、4）,rain_idx、test_idx 是索引数组，例如 [0,2,3,5,...]

    #         train_edges = edges_array[train_idx] ##取训练集的边
    #         test_edges = edges_array[test_idx] ##取测试集的边

    #         train_nodes = set() ##收集训练集涉及的所有节点
    #         test_nodes = set() ##收集测试集涉及的所有节点
    #         for u, v in train_edges: ##遍历每条训练边的两个端点，通过 node_map 把序列名转成节点ID，用 set 自动去重
    #             train_nodes.add(node_map[u])
    #             train_nodes.add(node_map[v])
    #         for u, v in test_edges: ##遍历每条测试边的两个端点，通过 node_map 把序列名转成节点ID，用 set 自动去重
    #             test_nodes.add(node_map[u])
    #             test_nodes.add(node_map[v])
            
    #         adj_lists = defaultdict(set)
    #         for u, v in train_edges:
    #             node_u = node_map[u]
    #             node_v = node_map[v]
    #             adj_lists[node_u].add(node_v)
    #             adj_lists[node_v].add(node_u)

    #         folds.append({"fold": fold_id,"train_edges": train_edges,"test_edges": test_edges,"train_nodes": np.array(list(train_nodes)),"test_nodes": np.array(list(test_nodes)),"adj_lists": adj_lists}) ##把每个折的编号、训练边、测试边、训练节点、测试节点都保存在一个字典里，并把这个字典添加到 folds 列表中

    #     return folds


    # def _split_data(self, num_nodes, test_split=3, val_split=6):
    #     rand_indices = np.random.permutation(num_nodes)

    #     test_size = num_nodes // test_split
    #     val_size = num_nodes // val_split
    #     train_size = num_nodes - (test_size + val_size)

    #     test_indexs = rand_indices[:test_size]
    #     val_indexs = rand_indices[test_size:(test_size + val_size)]
    #     train_indexs = rand_indices[(test_size + val_size):]

    #     return test_indexs, val_indexs, train_indexs

    # def _split_data_edge(self, num_edges, node_map, edges_lists, test_split=5, val_split=10):
    #     rand_indices = np.random.permutation(num_edges)
    #     ### 通过边列表来划分训练集，测试集，和验证集
    #     test_edge_size = num_edges // test_split
    #     val_edge_size = num_edges // val_split
    #     train_edge_size = num_edges - (test_edge_size + val_edge_size)

    #     test_edge_indexs = rand_indices[:test_edge_size]
    #     val_edge_indexs = rand_indices[test_edge_size:(test_edge_size + val_edge_size)]
    #     train_edge_indexs = rand_indices[(test_edge_size + val_edge_size):]

    #     #print(test_edge_indexs)

    #     edges_array = np.array(edges_lists)

    #     test_edge = edges_array[test_edge_indexs]
    #     val_edge = edges_array[val_edge_indexs]
    #     train_edge = edges_array[train_edge_indexs]

    #     test_indexs  = np.empty(shape=(0,))
    #     val_indexs = np.empty(shape=(0,))
    #     train_indexs = np.empty(shape=(0,))

    #     #print(test_edge)
    #     for idx in range(len(test_edge)):
    #         edges_idx = test_edge[idx]
    #         # node1 = int(edges_idx[0])
    #         # node2 = int(edges_idx[1])
    #         node1 = edges_idx[0]
    #         node2 = edges_idx[1]
    #         node1_index = node_map[str(node1)]
    #         node2_index = node_map[str(node2)]
    #         test_indexs = np.append(test_indexs, int(node1_index))
    #         test_indexs = np.append(test_indexs, int(node2_index))
    #     for idx in range(len(val_edge)):
    #         edges_idx = val_edge[idx]
    #         # node1 = int(edges_idx[0])
    #         # node2 = int(edges_idx[1])
    #         node1 = edges_idx[0]
    #         node2 = edges_idx[1]
    #         node1_index = node_map[str(node1)]
    #         node2_index = node_map[str(node2)]
    #         val_indexs = np.append(val_indexs, int(node1_index))
    #         val_indexs = np.append(val_indexs, int(node2_index))

    #     for idx in range(len(train_edge)):
    #         edges_idx = train_edge[idx]
    #         # node1 = int(edges_idx[0])
    #         # node2 = int(edges_idx[1])
    #         node1 = edges_idx[0]
    #         node2 = edges_idx[1]
    #         node1_index = node_map[str(node1)]
    #         node2_index = node_map[str(node2)]
    #         train_indexs = np.append(train_indexs, int(node1_index))
    #         train_indexs = np.append(train_indexs, int(node2_index))

    #     # 将浮点数数组转换为整数数组
    #     test_indexs = test_indexs.astype(int)
    #     val_indexs = val_indexs.astype(int)
    #     train_indexs = train_indexs.astype(int)

    #     test_indexs = np.unique(test_indexs)
    #     val_indexs = np.unique(val_indexs)
    #     train_indexs = np.unique(train_indexs)

    #     return test_indexs, val_indexs, train_indexs

    # def _split_data_vecnet(self, edges_lists_train, edges_lists_test, node_map, val_split=10):
    #     val_edge_size = len(edges_lists_train) // val_split
    #     edges_train_array = np.array(edges_lists_train)
    #     edges_test_array = np.array(edges_lists_test)

    #     train_edge = edges_train_array[val_edge_size:]
    #     val_edge = edges_train_array[:val_edge_size]
    #     test_edge = edges_test_array

    #     test_indexs  = np.empty(shape=(0,))
    #     val_indexs = np.empty(shape=(0,))
    #     train_indexs = np.empty(shape=(0,))

    #     for idx in range(len(test_edge)):
    #         edges_idx = test_edge[idx]
    #         # node1 = int(edges_idx[0])
    #         # node2 = int(edges_idx[1])
    #         node1 = edges_idx[0]
    #         node2 = edges_idx[1]
    #         node1_index = node_map[str(node1)]
    #         node2_index = node_map[str(node2)]
    #         test_indexs = np.append(test_indexs, int(node1_index))
    #         test_indexs = np.append(test_indexs, int(node2_index))
    #     for idx in range(len(val_edge)):
    #         edges_idx = val_edge[idx]
    #         # node1 = int(edges_idx[0])
    #         # node2 = int(edges_idx[1])
    #         node1 = edges_idx[0]
    #         node2 = edges_idx[1]
    #         node1_index = node_map[str(node1)]
    #         node2_index = node_map[str(node2)]
    #         val_indexs = np.append(val_indexs, int(node1_index))
    #         val_indexs = np.append(val_indexs, int(node2_index))

    #     for idx in range(len(train_edge)):
    #         edges_idx = train_edge[idx]
    #         # node1 = int(edges_idx[0])
    #         # node2 = int(edges_idx[1])
    #         node1 = edges_idx[0]
    #         node2 = edges_idx[1]
    #         node1_index = node_map[str(node1)]
    #         node2_index = node_map[str(node2)]
    #         train_indexs = np.append(train_indexs, int(node1_index))
    #         train_indexs = np.append(train_indexs, int(node2_index))

    #     # 将浮点数数组转换为整数数组
    #     test_indexs = test_indexs.astype(int)
    #     val_indexs = val_indexs.astype(int)
    #     train_indexs = train_indexs.astype(int)

    #     test_indexs = np.unique(test_indexs)
    #     val_indexs = np.unique(val_indexs)
    #     train_indexs = np.unique(train_indexs)
    #     # print(f"test_indexs: {test_indexs}")
    #     # print(f"val_indexs: {val_indexs}")
    #     # print(f"train_indexs: {train_indexs}")
    #     return test_indexs, val_indexs, train_indexs



