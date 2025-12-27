import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, x, edge_index):
        # Layer 1
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hidden Layers with Residual Connections
        for i in range(1, len(self.convs) - 1):
            x_new = self.convs[i](x, edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual connection

        # Output Layer
        x = self.convs[-1](x, edge_index)
        
        return x