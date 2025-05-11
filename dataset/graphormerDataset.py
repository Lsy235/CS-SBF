# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List
import torch
import numpy as np
import copy
from functools import lru_cache
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos
# import algos

from .graphDataset import MultiGraph

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from typing import Any

# class MultiGraph(Data):
#     def __init__(
#         self,
#         x: OptTensor = None,
#         edge_index: OptTensor = None,
#         edge_attr: OptTensor = None,
#         attn_bias: OptTensor = None,
#         attn_edge_type: OptTensor = None,
#         spatial_pos: OptTensor = None,
#         in_degree: OptTensor = None,
#         out_degree: OptTensor = None,
#         edge_input: OptTensor = None,
#         label:OptTensor = None,
#         y:OptTensor = None,
#         idx:OptTensor = None,
#         *args,
#         **kwargs
#     ):
#
#         super().__init__(*args, **kwargs)
#         self.x = x
#         self.edge_index = edge_index
#         self.edge_attr = edge_attr
#         self.y = y
#         self.idx = idx
#         self.label = label
#         self.attn_bias = attn_bias,
#         self.attn_edge_type = attn_edge_type,
#         self.spatial_pos = spatial_pos,
#         self.in_degree = in_degree,
#         self.out_degree = out_degree,
#         self.edge_input = edge_input,
#
#     def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         if key == "edge_index":
#             return self.x.size(0)
#         if key == "edge_index_b":
#             return self.x_b.size(0)
#         else:
#             return super().__inc__(key, value, *args, **kwargs)
#
#     def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         if key == "seq_feat":
#             return None
#         if key == "img_feat":
#             return None
#         if key == "struct_feat":
#             return None
#         else:
#             res = super(MultiGraph, self).__cat_dim__(key, value, *args, **kwargs)
#             # print(f"cat_dim: {res}")
#             return res


class GraphormerPYGDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
    ):
        self.dataset = dataset
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 5, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        self.__indices__ = None

    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data

import pandas as pd
from utils.collators import myCollator
from utils.wrapper import preprocess_item
# def convert_to_single_emb(x, offset: int = 512):
#     feature_num = x.size(1) if len(x.size()) > 1 else 1
#     feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.int32)
#     x = x + feature_offset
#     return x
#
#
# def preprocess_item(item):
#     edge_attr, edge_index, x = item['edge_attr'], item['edge_index'], item['x']
#     N = x.size(0)
#     x = convert_to_single_emb(x)
#
#     # node adj matrix [N, N] bool
#     adj = torch.zeros([N, N], dtype=torch.bool)
#     adj[edge_index[0, :], edge_index[1, :]] = True
#
#     # edge feature here
#     if len(edge_attr.size()) == 1:
#         edge_attr = edge_attr[:, None]
#     attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.int32)
#     attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
#             convert_to_single_emb(edge_attr) + 1
#     )
#
#     shortest_path_result, path = algos.floyd_warshall(adj.numpy())
#     max_dist = np.amax(shortest_path_result)
#     attn_edge_type = attn_edge_type.to(dtype=torch.int8)
#     edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
#     spatial_pos = torch.from_numpy((shortest_path_result)).long()
#     attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
#
#     # # combine
#     # item.x = x
#     # item.attn_bias = attn_bias
#     # item.attn_edge_type = attn_edge_type
#     # item.spatial_pos = spatial_pos
#     # item.in_degree = adj.long().sum(dim=1).view(-1)
#     # item.out_degree = item.in_degree  # for undirected graph
#     # edge_input = torch.from_numpy(edge_input)
#     # edge_input = edge_input.to(dtype=torch.int16)
#     # item.edge_input = edge_input
#
#     edge_input = torch.from_numpy(edge_input)
#     edge_input = edge_input.to(dtype=torch.int16)
#     item = MultiGraph(
#         x=x,
#         edge_attr=item['edge_attr'],
#         edge_index=item['edge_index'],
#         y=item['y'],
#         label=item['label'],
#         idx=item['idx'],
#         attn_bias=attn_bias,
#         attn_edge_type = attn_edge_type,
#         spatial_pos = spatial_pos,
#         in_degree = adj.long().sum(dim=1).view(-1),
#         out_degree = adj.long().sum(dim=1).view(-1), # for undirected graph
#         edge_input = edge_input,
#     )
#
#     return item, item.y

import networkx as nx
class GraphormerDataset(GraphormerPYGDataset):
    def __init__(
        self, root, dataset, mode, transform=None, pre_transform=None, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024
    ):
        self.root = root
        self.dataName = dataset
        self.mode = mode
        # print(self.root)
        self.rawDir = os.path.join(self.root, "granuarRaw")

        self.data = pd.read_csv(os.path.join(self.root, "granuarRaw", f"{self.mode}Y.csv"))
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

        # self.flags = [x.split(".")[0]+".jpg" for x in os.listdir(os.path.join(self.rawDir, self.mode))]
        # self.data = self.data[self.data['name'].isin(self.flags)]
        self.dataList = self.data['name'].tolist()
        self.labels = self.data['label'].tolist()
        # print(f"{m} data len:", len(dataList))
        # print(f"{m} label len:", len(labels))
        assert len(self.dataList) == len(self.labels)
        if (min(self.labels) > 0):
            self.labels = (np.array(self.labels)-1).tolist()
        print(f"label: max {max(self.labels)}, min {min(self.labels)}")
        self.printFlag = True

    def calculate_angle(self, v1, v2):
        """计算两个向量之间的角度"""
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))  # 弧度
        return np.degrees(angle)

    def getGraph(self, raw_file, label, idx):
        G = torch.load(os.path.join(self.rawDir, self.mode, raw_file))
        # correctOrNot = []
        # extract features
        node_features = []
        for node, data in G.nodes(data=True):
            # node_features.append(data['feature'][:4] + data['feature'][5:9]) # 9
            node_features.append(data['feature'][:4] + data['feature'][5:9]+ data['feature'][15:16]) # 9
        node_features = torch.tensor(node_features, dtype=torch.int32)
        # print(f"node_features min: {node_features.min()}, max: {node_features.max()}")
        # minX = min(np.array(node_features)[:,0])
        # maxX = max(np.array(node_features)[:,0])
        # minY = min(np.array(node_features)[:,1])
        # maxY = max(np.array(node_features)[:,1])
        # minRX = min(np.array(node_features)[:, 2])
        # maxRX = max(np.array(node_features)[:, 2])
        # minRY = min(np.array(node_features)[:, 3])
        # maxRY = max(np.array(node_features)[:, 3])
        # correctOrNot.append([minX, maxX, minY, maxY, minRX, maxRX, minRY, maxRY])

        # extract edge
        edge_index = []
        edge_features_int = []
        edge_features_float = []
        edge_features = []
        for edge in G.edges(data=True):
            edge_index.append([edge[0], edge[1]])
            # edge_features_int.append([edge[2]['feature'][12]+edge[2]['feature'][13], edge[2]['feature'][12]-abs(edge[2]['feature'][1]-edge[2]['feature'][7])+edge[2]['feature'][13]-abs(edge[2]['feature'][0]-edge[2]['feature'][6])]) # 19
            edge_features_int.append(edge[2]['feature'][12:14]+edge[2]['feature'][15:19]) # 19
            edge_features_float.append(edge[2]['feature'][19:23] + edge[2]['feature'][25:])
            edge_features.append(edge[2]['feature'][12:14]+edge[2]['feature'][15:19]+edge[2]['feature'][19:23] + edge[2]['feature'][25:])

        edge_attr_b = []
        edge_index_b = []
        bond_angle_graph = nx.Graph()
        for edge in G.edges():
            pos_u, pos_v = np.array([G.nodes[edge[0]]['feature'][0], G.nodes[edge[0]]['feature'][1]]), np.array([G.nodes[edge[1]]['feature'][0], G.nodes[edge[1]]['feature'][1]])
            for w in G.neighbors(edge[0]):
                if(w != edge[1]):
                    pos_w = np.array([G.nodes[w]['feature'][0], G.nodes[w]['feature'][1]])
                    vector_1 = pos_v - pos_u
                    vector_2 = pos_w - pos_u
                    angle = np.arctan2(vector_2[1], vector_2[0]) - np.arctan2(vector_1[1], vector_1[0])
                    edge_attr_b.append(angle)
                    edge_index_b.append([edge[0], w])
                    bond_angle_graph.add_edge(edge[0], w, angle=angle)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_features_int = torch.tensor(edge_features_int, dtype=torch.int32)
        # print(f"edge_features_int max: {edge_features_int.max()}")
        # edge_features_float = torch.tensor(edge_features_float, dtype=torch.float)
        edge_features = torch.tensor(edge_features, dtype=torch.int32)
        # correctOrNot.append([node_features.shape[0], edge_index.shape[1], edge_features_int.shape[0]])

        x_b = torch.ones([edge_index.shape[1], 1])
        edge_index_b = torch.tensor(edge_index_b)
        edge_attr_b = torch.tensor(edge_attr_b)

        # build graph object
        # data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_int, y=torch.tensor([label]), idx=idx, label=label)
        data = MultiGraph(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            # edge_attr_a_float=edge_features_float,
            x_b=x_b,
            edge_index_b=edge_index_b,
            edge_attr_b=edge_attr_b,
            label=torch.tensor([label]),
            y=torch.tensor([label]),
            idx=torch.tensor([idx]),
        )
        # data = {
        #     "x":node_features,
        #     "edge_index":edge_index,
        #     "edge_attr":edge_features,
        #     "label":label,
        #     "y":label,
        #     "idx":idx,
        # }
        # print(type(data))
        return data
        # return correctOrNot

    def __getitem__(self, idx):
        if isinstance(idx, int):
            raw_file = self.dataList[idx].split(".")[0] + ".pt"
            item = self.getGraph(raw_file, self.labels[idx], idx)
            # correctOrNot = self.getGraph(raw_file, self.labels[idx], idx)
            # item = preprocess_item(item)
            # print(item.x)
            # print(type(item))
            # print(f"edge_features: {item.edge_attr[:10, :]}")
            if(self.printFlag):
                # print(
                #     f"node_features:{item.x.size()}, "
                #     f"edge_index:{item.edge_index.size()}, "
                #     f"edge_features:{item.edge_attr.size()},"
                #     f"attn_bias:{item.attn_bias.size()},"
                #     f"attn_edge_type:{item.attn_edge_type.size()},"
                #     f"spatial_pos:{item.spatial_pos.size()},"
                #     f"out_degree:{item.out_degree.size()},"
                #     f"edge_input:{item.edge_input.size()},"
                # )
                print(
                    f"node_features:{item.x.size()}, "
                    f"edge_index:{item.edge_index.size()}, "
                    f"edge_features:{item.edge_attr.size()},"
                    f"x_b:{item.x_b.size()},"
                    f"edge_index_b:{item.edge_index_b.size()},"
                    f"edge_attr_b:{item.edge_attr_b.size()},"
                    f"label:{item.label.size()},"
                    f"idx:{item.idx.size()},"
                )
                self.printFlag=False
            return item
            # return correctOrNot
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return len(self.dataList)

    # def collater(self, samples):
    #     return collator(
    #         samples,
    #         max_node=self.max_node,
    #         multi_hop_max_dist=self.multi_hop_max_dist,
    #         spatial_pos_max=self.spatial_pos_max,
    #     )

import torch_geometric
from torch_geometric.data import Batch

def custom_collate(data_list):
    return Batch.from_data_list(data_list)
class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, collate_fn=None,
        shuffle=False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, **kwargs)

class CustomDataLoader2(torch_geometric.loader.DataLoader):
    def __init__(self, dataset, batch_size=1, collate_fn=None, follow_batch=None,
        shuffle=False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, follow_batch=follow_batch, **kwargs)

if __name__ == "__main__":
    graphPath = "D:\\Documents\\Python\\myDriveData\\CK+48\\graph"
    data_name = "CK"
    allGrapgDataset = GraphormerDataset(
        root=graphPath,
        dataset=data_name,
        mode="train",
    )
    print(f"train dataset len: {len(allGrapgDataset)}")

    trGraphLoader = CustomDataLoader2(
        allGrapgDataset,
        batch_size=1,
        follow_batch=["x"],
        shuffle=False,
        # pin_memory=True,
        # collate_fn=myCollator,
    )
    correctOrNots = []
    for i, correctOrNot in enumerate(trGraphLoader):
        # print(f"graph.x:{graph['x'].shape}, graph.y:{graph['y'].shape}")
        # print(torch.cat(correctOrNot[0]).numpy().tolist())
        correctOrNots.append(torch.cat(correctOrNot[0]).numpy().tolist())
        # break
    correctOrNots = pd.DataFrame(correctOrNots, columns=["nodeFeaturesShape0", "edgeIndexShape1", "edgeAttrShape0"])
    for x in correctOrNots.columns:
        print(correctOrNots[x].describe())
    """
    train dataset len: 11043
count    11043.000000
mean         0.514896
std          2.175265
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max         52.000000
Name: minX, dtype: float64
count    11043.000000
mean        98.638776
std          1.596786
min         36.000000
25%         99.000000
50%         99.000000
75%         99.000000
max         99.000000
Name: maxX, dtype: float64
count    11043.000000
mean         0.287150
std          1.705082
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max         94.000000
Name: minY, dtype: float64
count    11043.000000
mean        98.476954
std          2.113148
min         45.000000
25%         99.000000
50%         99.000000
75%         99.000000
max         99.000000
Name: maxY, dtype: float64
count    11043.000000
mean         0.034139
std          1.882101
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max        142.000000
Name: minRX, dtype: float64
count    11043.000000
mean        23.648012
std         14.027268
min          5.000000
25%         15.000000
50%         20.000000
75%         27.000000
max        142.000000
Name: maxRX, dtype: float64
count    11043.000000
mean         0.029159
std          1.630364
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max        135.000000
Name: minRY, dtype: float64
count    11043.000000
mean        19.943041
std         11.686027
min          4.000000
25%         13.000000
50%         17.000000
75%         23.000000
max        135.000000
Name: maxRY, dtype: float64
    """
    """
    CK224
    train dataset len: 784
count    784.000000
mean     335.707908
std       83.751641
min      154.000000
25%      278.750000
50%      344.500000
75%      396.000000
max      512.000000
Name: nodeFeaturesShape0, dtype: float64
count     784.000000
mean      370.964286
std       341.858557
min        77.000000
25%       170.750000
50%       245.500000
75%       453.000000
max      3598.000000
Name: edgeIndexShape1, dtype: float64
    """
    """
    CK24
    count    784.000000
mean     231.040816
std       40.723167
min      105.000000
25%      207.000000
50%      240.500000
75%      261.000000
max      324.000000
Name: nodeFeaturesShape0, dtype: float64
count    784.000000
mean     122.658163
std       36.658418
min       55.000000
25%      100.000000
50%      116.000000
75%      137.000000
max      422.000000
Name: edgeIndexShape1, dtype: float64
    """