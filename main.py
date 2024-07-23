import numpy as np
import os
from glob import glob
import json
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.metrics import f1_score
from tqdm import tqdm
from seaborn import heatmap

def load_data(node_features, node_labels, graph_file):
    node_features = np.load(node_features)
    node_labels = np.load(node_labels)
    graph = json.load(open(graph_file))

    G = nx.node_link_graph(graph)

    multi_graph = nx.MultiGraph()

    for node_id, node in G.nodes(data=True):
        multi_graph.add_node(node_id, **node)
        node_feat = node_features[node_id]
        multi_graph.nodes[node_id]['x'] = node_feat

        node_label = node_labels[node_id]
        multi_graph.nodes[node_id]['y'] = node_label

    for edge in G.edges(data=True):
        edge_weight = edge[2]['weight']
        passive, gate, source, drain = edge_weight & 0b1000, edge_weight & 0b0100, edge_weight & 0b0010, edge_weight & 0b0001

        if passive:
            multi_graph.add_edge(edge[0], edge[1], key='P')
        if gate:
            multi_graph.add_edge(edge[0], edge[1], key='G')
        if source:
            multi_graph.add_edge(edge[0], edge[1], key='S')
        if drain:
            multi_graph.add_edge(edge[0], edge[1], key='D')

    data = HeteroData()
    data['node'].x = torch.tensor(node_features, dtype=torch.float)
    data['node'].y = torch.tensor(node_labels, dtype=torch.float)

    passive_edge_index = []
    gate_edge_index = []
    source_edge_index = []
    drain_edge_index = []

    for u, v, key in multi_graph.edges(keys=True):
        if key == 'P':
            passive_edge_index.append([u, v])
        if key == 'G':
            gate_edge_index.append([u, v])
        if key == 'S':
            source_edge_index.append([u, v])
        if key == 'D':
            drain_edge_index.append([u, v])

    data['node', 'P', 'node'].edge_index = torch.tensor(passive_edge_index, dtype=torch.long).t().contiguous()
    data['node', 'G', 'node'].edge_index = torch.tensor(gate_edge_index, dtype=torch.long).t().contiguous()
    data['node', 'S', 'node'].edge_index = torch.tensor(source_edge_index, dtype=torch.long).t().contiguous()
    data['node', 'D', 'node'].edge_index = torch.tensor(drain_edge_index, dtype=torch.long).t().contiguous()

    return data

ota_or_rf = "RF"

train_data = load_data(f'circuit_data/{ota_or_rf}_data/processed/train_feats.npy', f'circuit_data/{ota_or_rf}_data/processed/train_labels.npy', f'circuit_data/{ota_or_rf}_data/processed/train_graph.json')
valid_data = load_data(f'circuit_data/{ota_or_rf}_data/processed/valid_feats.npy', f'circuit_data/{ota_or_rf}_data/processed/valid_labels.npy', f'circuit_data/{ota_or_rf}_data/processed/valid_graph.json')
test_data = load_data(f'circuit_data/{ota_or_rf}_data/processed/test_feats.npy', f'circuit_data/{ota_or_rf}_data/processed/test_labels.npy', f'circuit_data/{ota_or_rf}_data/processed/test_graph.json')

train_data = train_data.to('cuda')
valid_data = valid_data.to('cuda')
test_data = test_data.to('cuda')

class HeteroGraphSAGE(nn.Module):
    def __init__(self, n_layers, hidden_channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layers):  # Number of layers
            conv = HeteroConv({
                ('node', 'D', 'node'): SAGEConv((-1, -1), hidden_channels),
                ('node', 'S', 'node'): SAGEConv((-1, -1), hidden_channels),
                ('node', 'G', 'node'): SAGEConv((-1, -1), hidden_channels),
                ('node', 'P', 'node'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['node'])

final_train_losses = []
final_valid_losses = []
final_train_f1s = []
final_valid_f1s = []

for n_layers in range(2, 8):
    for hidden_channels in [8, 16, 32, 64, 128, 256]:
        model = HeteroGraphSAGE(n_layers, hidden_channels, train_data['node'].y.size(1))
        model = model.to('cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(600)):
            model.train()
            optimizer.zero_grad()
            out = model(train_data.x_dict, train_data.edge_index_dict)
            train_loss = criterion(out, train_data['node'].y)
            train_loss.backward()
            optimizer.step()

            # Print F1 score
            pred = out.argmax(dim=1).cpu().detach().numpy()
            target = np.argmax(train_data['node'].y.cpu().detach().numpy(), axis=1)

            train_f1 = f1_score(target, pred, average = 'micro')

            # Validation
            model.eval()
            out = model(valid_data.x_dict, valid_data.edge_index_dict)
            valid_loss = criterion(out, valid_data['node'].y)

            pred = out.argmax(dim=1).cpu().detach().numpy()
            target = np.argmax(valid_data['node'].y.cpu().detach().numpy(), axis=1)

            valid_f1 = f1_score(target, pred, average = 'micro')

        final_train_losses.append(train_loss.item())
        final_valid_losses.append(valid_loss.item())
        final_train_f1s.append(train_f1)
        final_valid_f1s.append(valid_f1)

        print(f'Layers: {n_layers}, Hidden Channels: {hidden_channels}, Train Loss: {final_train_losses[-1]:.4f}, Train F1: {(final_train_f1s[-1]*100):.2f}%, Valid Loss: {final_valid_losses[-1]:.4f}, Valid F1: {(final_valid_f1s[-1]*100):.2f}%')

# Plot a heatmap of the results
final_train_losses = np.array(final_train_losses).reshape(6, 6)
final_valid_losses = np.array(final_valid_losses).reshape(6, 6)
final_train_f1s = np.array(final_train_f1s).reshape(6, 6)
final_valid_f1s = np.array(final_valid_f1s).reshape(6, 6)

fig, axs = plt.subplots(2, 2)

heatmap(final_train_losses, ax=axs[0, 0], cmap='viridis', square=True, annot=True)
axs[0, 0].set_title('Train Loss')

heatmap(final_valid_losses, ax=axs[0, 1], cmap='viridis', square=True, annot=True)
axs[0, 1].set_title('Valid Loss')

heatmap(final_train_f1s, ax=axs[1, 0], cmap='viridis', square=True, annot=True)
axs[1, 0].set_title('Train F1 Score')

heatmap(final_valid_f1s, ax=axs[1, 1], cmap='viridis', square=True, annot=True)
axs[1, 1].set_title('Valid F1 Score')

axs[0, 0].set_xticklabels([8, 16, 32, 64, 128, 256])
axs[0, 0].set_yticklabels([2, 3, 4, 5, 6, 7])
axs[0, 0].set_xlabel('Hidden Channels')
axs[0, 0].set_ylabel('Layers')

axs[0, 1].set_xticklabels([8, 16, 32, 64, 128, 256])
axs[0, 1].set_yticklabels([2, 3, 4, 5, 6, 7])
axs[0, 1].set_xlabel('Hidden Channels')
axs[0, 1].set_ylabel('Layers')

axs[1, 0].set_xticklabels([8, 16, 32, 64, 128, 256])
axs[1, 0].set_yticklabels([2, 3, 4, 5, 6, 7])
axs[1, 0].set_xlabel('Hidden Channels')
axs[1, 0].set_ylabel('Layers')

axs[1, 1].set_xticklabels([8, 16, 32, 64, 128, 256])
axs[1, 1].set_yticklabels([2, 3, 4, 5, 6, 7])
axs[1, 1].set_xlabel('Hidden Channels')
axs[1, 1].set_ylabel('Layers')

plt.show()


# n_layers = 4
# hidden_channels = 64
# num_classes = train_data['node'].y.size(1)

# model = HeteroGraphSAGE(n_layers, hidden_channels, num_classes)
# model = model.to('cuda')

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()
# # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 25)

# train_losses = []
# train_f1s = []

# valid_losses = []
# valid_f1s = []

# for epoch in range(500):
#     model.train()
#     optimizer.zero_grad()
#     out = model(train_data.x_dict, train_data.edge_index_dict)
#     loss = criterion(out, train_data['node'].y)
#     loss.backward()
#     optimizer.step()


#     # Print F1 score
#     pred = out.argmax(dim=1).cpu().detach().numpy()
#     target = np.argmax(train_data['node'].y.cpu().detach().numpy(), axis=1)

#     f1 = f1_score(target, pred, average="micro")

#     train_losses.append(loss.item())
#     train_f1s.append(f1)

#     # Validation
#     model.eval()
#     out = model(valid_data.x_dict, valid_data.edge_index_dict)
#     valid_loss = criterion(out, valid_data['node'].y)

#     pred = out.argmax(dim=1).cpu().detach().numpy()
#     target = np.argmax(valid_data['node'].y.cpu().detach().numpy(), axis=1)

#     f1 = f1_score(target, pred, average='micro')

#     valid_losses.append(valid_loss.item())
#     valid_f1s.append(f1)

#     # curr_lr = scheduler.get_last_lr()
    
#     print(f'Epoch: {epoch}, Train Loss: {train_losses[-1]:.4f}, Train F1: {(train_f1s[-1]*100):.2f}%, Valid Loss: {valid_losses[-1]:.4f}, Valid F1: {(valid_f1s[-1]*100):.2f}%')
#     # scheduler.step(valid_loss)

# fig, axs = plt.subplots(2)
# axs[0].plot(train_losses)
# axs[0].plot(valid_losses)
# axs[0].set_title('Training Loss')
# axs[0].legend(['Train', 'Valid'])
# axs[1].plot(train_f1s)
# axs[1].plot(valid_f1s)
# axs[1].set_title('Training F1 Score')
# axs[1].legend(['Train', 'Valid'])

# # Set axs1 y-axis to between 0 and 1
# axs[1].set_ylim([0, 1])

# plt.show()