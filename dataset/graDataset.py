import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.typing import OptTensor, NodeType, EdgeType
from typing import Any
import networkx as nx

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos
# import algos

class MultiGraph(Data):
    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        x_b: OptTensor = None,
        edge_index_b: OptTensor = None,
        edge_attr_b: OptTensor = None,
        label:OptTensor = None,
        y:OptTensor = None,
        idx:OptTensor = None,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.idx = idx
        self.label = label
        # self.x_b = x_b
        # self.edge_index_b = edge_index_b
        # self.edge_attr_b = edge_attr_b

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index":
            return self.x.size(0)
        if key == "edge_index_b":
            return self.x_b.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "seq_feat":
            return None
        if key == "img_feat":
            return None
        if key == "struct_feat":
            return None
        else:
            return super(MultiGraph, self).__cat_dim__(key, value, *args, **kwargs)

class GraphDataset(InMemoryDataset):
    def __init__(self, root, dataset, mode, transform=None, pre_transform=None):
        self.root = root
        self.dataset = dataset
        self.mode = mode
        self.rawDir = os.path.join(self.root, "raw")

        super(GraphDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])


        # select data
        if self.mode == "train":
            print(self.processed_paths[0])
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.mode == "val":
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.mode == "test":
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            print(f"no {self.mode}")
            import sys
            sys.exit(0)

    @property
    def raw_file_names(self):
        # return origin filename
        return os.listdir(os.path.join(self.root, 'raw'))

    @property
    def processed_file_names(self):
        # return ['data.pt']
        return [
            "{}_train.pt".format(self.dataset),
            "{}_val.pt".format(self.dataset),
            "{}_test.pt".format(self.dataset),
        ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def download(self):
        # load data
        pass

    def process(self):
        for i, m in enumerate(['train', 'val', 'test']):
            self.data = pd.read_csv(os.path.join(self.rawDir, f"{m}Y.csv"))
            data_list = []
            dataList = self.data['name'].tolist()
            labels = self.data['label'].tolist()
            print(f"{m} data len:",len(dataList))
            print(f"{m} label len:",len(labels))
            assert len(dataList) == len(labels)

            if (min(labels) > 0):
                labels = (np.array(labels)-1).tolist()
            lNum = 0
            for j, raw_file in enumerate(dataList):
                raw_file = raw_file.split(".")[0] + ".pt"
                G = torch.load(os.path.join(self.rawDir, m, raw_file))
                # extract features
                node_features = []
                for node, data in G.nodes(data=True):
                    # 节点特征
                    node_features.append(data['feature'][:4] + data['feature'][5:9] + data['feature'][15:16])  # 9
                node_features = torch.tensor(node_features, dtype=torch.int32)

                # extract edge
                edge_index = []
                edge_features_int = []
                edge_features_float = []
                edge_features = []
                for edge in G.edges(data=True):
                    # 边特征
                    edge_index.append([edge[0], edge[1]])
                    edge_features_int.append(edge[2]['feature'][12:14] + edge[2]['feature'][15:19])  # 19
                    edge_features_float.append(edge[2]['feature'][19:23] + edge[2]['feature'][25:])
                    edge_features.append(
                        edge[2]['feature'][12:14] + edge[2]['feature'][15:19] + edge[2]['feature'][19:23] + edge[2]['feature'][25:]
                    )

                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                edge_features_int = torch.tensor(edge_features_int, dtype=torch.float)
                edge_features_float = torch.tensor(edge_features_float, dtype=torch.float)
                edge_features = torch.tensor(edge_features, dtype=torch.float)

                label = labels[j]

                # build graph object
                data = MultiGraph(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_features,
                    label=torch.tensor([label]),
                    y=torch.tensor([label]),
                    idx=torch.tensor([j]),
                )

                data_list.append(data)
                lNum+=1
                print(f"lNum No.{lNum}, "
                      f"node_features:{node_features.shape}, "
                      f"edge_index:{edge_index.shape}, "
                      f"edge_features:{edge_features.shape},"
                    )
            print(f"{m} lNum: {lNum}")

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

import torch_geometric

# if __name__ == "__main__":
#     graphPath = r"D:\Documents\Python\myDriveData\CK+48\graph"
#     data_name = "CK"
#     allGraphDataset = GraphDataset(
#         root=graphPath,
#         dataset=data_name,
#         mode="train",
#     )
#     allGraphLoader = torch_geometric.loader.DataLoader(
#         allGraphDataset,
#         batch_size=4,
#         num_workers=1,
#         follow_batch=["x"],
#         shuffle=False,
#         pin_memory=True,
#     )
#     for step, graph in enumerate(allGraphLoader):
#         print(
#             f"node_features:{graph.x.size()}, "
#             f"edge_index:{graph.edge_index.size()}, "
#             f"edge_features:{graph.edge_attr.size()},"
#             f"label:{graph.label.size()},"
#             f"idx:{graph.idx.size()},"
#         )
#         break