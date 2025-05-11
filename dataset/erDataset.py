import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.typing import OptTensor, NodeType, EdgeType
from typing import Any
from torch.utils.data.sampler import Sampler

class MultiGraph(Data):
    def __init__(
        self,
        x_a: OptTensor = None,
        edge_index_a: OptTensor = None,
        edge_attr_a: OptTensor = None,
        x_b: OptTensor = None,
        edge_index_b: OptTensor = None,
        edge_attr_b: OptTensor = None,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.x_a = x_a
        self.edge_index_a = edge_index_a
        self.edge_attr_a = edge_attr_a
        #
        # self.x_b = x_b
        # self.edge_index_b = edge_index_b
        # self.edge_attr_b = edge_attr_b

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index_a":
            return self.x_a.size(0)
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
            res = super(MultiGraph, self).__cat_dim__(key, value, *args, **kwargs)
            # print(f"cat_dim: {res}")
            return res

class GraphDataset(InMemoryDataset):
    def __init__(self, root, dataset, mode, transform=None, pre_transform=None):
        self.root = root
        self.dataset = dataset
        self.mode = mode
        if(dataset == "RAF-DB"):
            self.df = pd.read_csv(
                os.path.join(self.root, "EmoLabel/list_patition_label.txt"),
                sep=" ",
                header=None,
                names=["name", "label"],
            )
        elif(dataset == "Oulu"):
            self.df = pd.read_csv(
                # os.path.join(self.root, "AllWeakRenameTogetherYLabel.csv"),
                "/home/ubuntu/try3Graph/data/AllWeakRenameTogetherYLabel.csv"
            )
        elif (dataset == "SFEW"):
            self.df = pd.read_csv(
                # os.path.join(self.root, "AllWeakRenameTogetherYLabel.csv"),
                "/home/ubuntu/try3Graph/data/data/alignedYLabel.csv"
            )
        elif (dataset == "CK"):
            self.df = pd.read_csv(
                # os.path.join(self.root, "AllWeakRenameTogetherYLabel.csv"),
                "/home/ubuntu/try3Graph/data/data2/alignedYLabel.csv"
            )
        elif (dataset == "FERPlus"):
            self.df = pd.read_csv(
                # os.path.join(self.root, "AllWeakRenameTogetherYLabel.csv"),
                "/home/ubuntu/try3Graph/data/alignedYLabel.csv"
            )
        elif (dataset == "fer2013"):
            # 0-8
            self.df = pd.read_csv(os.path.join(self.root, "granuarRaw", self.mode + "Y.csv"))
            self.df = self.df[self.df['label'] != 9]
            self.label = self.df.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            self.dataNames = self.df['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(self.dataNames):
                partPath = name.split(".")[0] + ".pt"
                path = os.path.join(self.root, "granuarRaw", self.mode, partPath)
                self.file_paths.append(path)
        else:
            self.df = None



        super(GraphDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])


        # select data
        if self.mode == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.mode == "val":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        # return origin filename
        return os.listdir(os.path.join(self.root, 'granuarRaw'))

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
        assert len(self.file_paths) == len(self.label)
        data_list = []
        lNum = 0
        for j, raw_file in enumerate(self.file_paths):
            # for j in range(len(dataList)-1):
            # raw_file = str(j) + ".pt"
            # if(".pt" not in raw_file):
            #     print(raw_file)
            #     continue
            # load .pt file
            G = torch.load(raw_file)

            # build mapping
            # str_to_int = {node: int(node) for node in G.nodes}

            # build new graph
            # G_int = nx.Graph()
            # for node, data in G.nodes(data=True):
            #     G_int.add_node(str_to_int[node], **data)
            # for edge in G.edges:
            #     G_int.add_edge(str_to_int[edge[0]], str_to_int[edge[1]], feature=[i+1 for i in range(3)])

            # extract features
            node_features = []
            for node, data in G.nodes(data=True):
                node_features.append(data['feature'])
            features = torch.tensor(node_features, dtype=torch.float)
            node_features = torch.where(
                torch.isnan(features),
                torch.full_like(features, 0.0),
                features)

            # extract edge
            edge_index = []
            edge_features_int = []
            edge_features_float = []
            for edge in G.edges(data=True):
                edge_index.append([int(edge[0]), int(edge[1])])
                edge_features_int.append(edge[2]['feature'][:])
                edge_features_float.append(edge[2]['feature'][-1])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_features_int = torch.tensor(edge_features_int, dtype=torch.float)
            edge_features_float = torch.tensor(edge_features_float, dtype=torch.float)

            l = self.label[j]
            # build graph object
            # data = Data(x=node_features, edge_index=edge_index, y=label[j])
            data = MultiGraph(
                x_a=node_features,
                edge_index_a=edge_index,
                edge_attr_a_int=edge_features_int,
                edge_attr_a_float=edge_features_float,
                label=l,
            )
            data_list.append(data)
            lNum += 1
            # print(
            #     f"node_features:{data.x_a.size()}, "
            #     f"edge_index:{data.edge_index_a.size()}, "
            #     f"edge_features:{data.edge_attr_a_int.size()},"
            #     # f"label:{item.label.size()},"
            #     # f"y:{item.y.size()},"
            # )
        print(f"lNum: {lNum}")

        data, slices = self.collate(data_list)
        if (self.mode == "train"):
            torch.save((data, slices), self.processed_paths[0])
        elif (self.mode == "val"):
            torch.save((data, slices), self.processed_paths[1])
        else:
            torch.save((data, slices), self.processed_paths[2])

class FixedOrderSampler(Sampler):
    def __init__(self, data_source, indices):
        self.data_source = data_source
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class ERDataset(data.Dataset):
    def __init__(self, data_path, phase, dataset = "RAF-DB", transform=None, mosaic=False):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        self.dataset = dataset

        if (dataset == "RAF-DB"):
            # 1-7
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.label = self.data.loc[:, "label"].values
            if(min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                path = os.path.join(self.data_path, "images\\aligned", partPath)
                self.file_paths.append(path)

            if(mosaic):
                self.data2 = pd.read_csv(os.path.join(self.data_path, "images", "mosaic", "mosaicY.csv"))
                self.label2 = self.data2.loc[:, "label"].values
                if (min(self.label2) > 0):
                    self.label2 = self.label2 - 1
                self.label = np.concatenate((self.label, self.label2), axis=0)
                dataNames = self.data2['name'].tolist()
                for id_, name in enumerate(dataNames):
                    path = os.path.join(self.data_path, "images\\mosaic", name)
                    self.file_paths.append(path)
        elif (dataset == "SFEW"):
            # 0-6
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.label = self.data.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                # partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                partPath = "\\".join(name.split("__"))
                path = os.path.join(self.data_path, "images", partPath)
                self.file_paths.append(path)
        elif (dataset == "CK"):
            # 0-6
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase+"Y.csv"))
            self.label = self.data.loc[:, "label"].values
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                partPath = name.split("__")[0]+"\\"+name.split("__")[1]
                path = os.path.join(self.data_path, "images", partPath)
                self.file_paths.append(path)
        elif (dataset == "AffectNet"):
            # 1-8
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.label = self.data.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                # partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                partPath = "\\".join(name.split("__"))
                path = os.path.join(self.data_path, "images", partPath)
                self.file_paths.append(path)
        elif (dataset == "Oulu"):
            # 0-5
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.label = self.data.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                # partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                partPath = "\\".join(name.split("__"))
                path = os.path.join(self.data_path, "images", partPath)
                self.file_paths.append(path)
        elif (dataset == "AffectNet-7"):
            # 0-6
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.label = self.data.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                # partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                partPath = "\\".join(name.split("__"))
                path = os.path.join(self.data_path, "images", partPath)
                self.file_paths.append(path)
        elif (dataset == "CAER-S"):
            # 1-7
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.label = self.data.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                # partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                partPath = "\\".join(name.split("__"))
                path = os.path.join(self.data_path, "images", partPath)
                self.file_paths.append(path)
        elif (dataset == "fer2013"):
            # 0-8
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.data = self.data[self.data['label'] != 9]
            self.label = self.data.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                # partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                partPath = name
                path = os.path.join(self.data_path, "images\\aligned", partPath)
                self.file_paths.append(path)
        elif (dataset == "SAMM"):
            # 1-7
            self.data = pd.read_csv(os.path.join(self.data_path, "graph", "granuarRaw", self.phase + "Y.csv"))
            self.label = self.data.loc[:, "label"].values
            if (min(self.label) > 0):
                self.label = self.label - 1
            dataNames = self.data['name'].tolist()
            self.file_paths = []
            for id_, name in enumerate(dataNames):
                # partPath = name.split(".")[0] + "_aligned." + name.split(".")[1]
                partPath = "\\".join(name.split("__"))
                path = os.path.join(self.data_path, "images", partPath)
                self.file_paths.append(path)
        elif(dataset == "CAM"):
            df = None



    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # print(f"file path: {self.file_paths[idx]}")
        path = self.file_paths[idx]
        image = cv2.imread(path)
        label = self.label[idx]
        # print(path)
        image = Image.fromarray(image[:, :, ::-1])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

