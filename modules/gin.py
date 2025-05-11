import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool, MessagePassing

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, args, outImgDim=64, readout='add'):
        super(GIN, self).__init__()

        self.args = args
        self.conv1 = GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(num_node_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        )
        self.conv2 = GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        )
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, self.args.num_classes)

        # 设置 readout 类型
        if readout == 'add':
            self.readout = global_add_pool
            # self.readout = torch_scatter.scatter_mean
        elif readout == 'mean':
            self.readout = global_mean_pool
            # self.readout = torch_scatter.scatter_add
        elif readout == 'max':
            self.readout = global_max_pool
            # self.readout = torch_scatter.scatter_max
        else:
            raise ValueError(f'Unknown readout type: {readout}')

    def forward(self, data):
        # process graph
        x, edge_index, batch = data.x, data.edge_index, data.x_batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 使用 readout 聚合节点特征
        x = self.readout(x, batch)
        out = self.fc1(x)
        # out = self.normalize(x)
        return out

    def normalize(self, x):
        tensor_min = self.args.normalMin
        tensor_max = self.args.normalMax
        out = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) * (
                    tensor_max - tensor_min) + tensor_min
        return out