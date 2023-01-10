import matplotlib.pyplot as plt
from rdkit import Chem
import rdkit
from torch_geometric.datasets import MoleculeNet
from rdkit.Chem import Draw
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
from torch_geometric.data import DataLoader
import seaborn as sns
import pandas as pd
from tqdm import tqdm
# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")
# for _, item in enumerate(data):
#     print(item)

S = 0
def visualize_molecule(data):
    for i in range(16):
        molecule = Chem.MolFromSmiles(data[i]["smiles"])
        str_ = 'images/molecule'+str(i)+'.png'
        Draw.MolToFile(molecule, str_)

class GCN(nn.Module):
    def __init__(self, embedding_size=64):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)
        # GCN layers
        self.initial_conv = gnn.GCNConv(data.num_features, embedding_size)
        self.conv1 = gnn.GCNConv(embedding_size, embedding_size)
        self.conv2 = gnn.GCNConv(embedding_size, embedding_size)
        self.conv3 = gnn.GCNConv(embedding_size, embedding_size)
        # Output layer
        self.out = gnn.Linear(embedding_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = torch.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gnn.global_max_pool(hidden, batch_index),
                            gnn.global_mean_pool(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden

def train(loader, device, optimizer, model, loss_fn):
    for batch in loader:
      batch.to(device)
      optimizer.zero_grad()
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
      loss = loss_fn(pred, batch.y)
      loss.backward()
      optimizer.step()
    return loss, embedding
def train_fn():
    model = GCN()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    data_size = len(data)
    num_graph_in_batch = 64
    loader = DataLoader(data[:int(data_size * 0.8)],
                        batch_size=num_graph_in_batch, shuffle=True)
    test_loader = DataLoader(data[int(data_size * 0.8):],
                             batch_size=num_graph_in_batch, shuffle=True)

    print("Starting training...")
    losses = []
    for epoch in range(2000):
        loss, h = train(loader, device, optimizer, model, loss_fn)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss {loss}")
    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
    plt_ = sns.lineplot(losses_float)
    plt_.set(xlabel='epoch', ylabel='error')

    plt.savefig('train.png')
    print("Starting testing..")
    testing(test_loader, device, model)

def testing(test_loader, device, model):
    # test_batch = next(iter(test_loader))
    y_real=[]
    y_pred=[]
    with torch.no_grad():
       for test_batch in test_loader:
            test_batch.to(device)
            pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
            y_real += test_batch.y.tolist()
            y_pred += pred.tolist()

       df = pd.DataFrame()
       df["y_real"] = y_real
       df["y_pred"] = y_pred
       df["y_real"] = df["y_real"].apply(lambda row: row[0])
       df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
       plt_ = sns.scatterplot(data=df, x="y_real", y="y_pred")
       plt_.set(xlim=(-7, 2), ylim=(-7, 2), xlabel='y_real', ylabel='y_pred')
       plt.savefig('test.png')

if __name__=='__main__':
    train_fn()