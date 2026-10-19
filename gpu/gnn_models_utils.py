import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.utils import to_dense_batch
import numpy as np
import random
from rdkit import Chem

# ==========================================
# 1. Dataset & DataLoader
# ==========================================

class DrugResponseDataset(Dataset):
    def __init__(self, X_gen, X_ch, labels):
        self.X_genomic = torch.tensor(X_gen, dtype=torch.float32)
        self.X_chem = torch.tensor(X_ch if isinstance(X_ch, np.ndarray) else X_ch.values, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_genomic[idx], self.X_chem[idx], self.y[idx]

def make_loaders(X_gen_train, X_chem_train, y_train, 
                 X_gen_val, X_chem_val, y_val, 
                 X_gen_test, X_chem_test, y_test, 
                 batch_size=128, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        DrugResponseDataset(X_gen_train, X_chem_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        DrugResponseDataset(X_gen_val, X_chem_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        DrugResponseDataset(X_gen_test, X_chem_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader

# ==========================================
# 2. Utils
# ==========================================

def smiles_to_graph(smiles: str) -> Data:
    """
    Converts a SMILES string into a PyTorch Geometric Data object.
    Extracts atom features (nodes) and bond features (edges).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic())
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float32)

    edge_indices = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_features = [
            bond.GetBondTypeAsDouble(),
            int(bond.IsInRing())
        ]

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        edge_attr_list.append(bond_features)
        edge_attr_list.append(bond_features)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def weighted_mse_loss(pred, target):
    weights = 1.0 + 2.0 * torch.exp(-3.0 * target)
    loss = weights * (pred - target) ** 2
    return loss.mean()

def build_pyg_dataset(smiles_array, gen_array, y_array, cache=None, use_cache=True):
    dataset = []
    for i in range(len(smiles_array)):
        smiles = smiles_array[i]
        
        # Entweder aus dem Cache holen oder from scratch bauen
        if use_cache and cache is not None:
            base_data = cache.get(smiles)
            # Wichtig: Wenn er aus dem Cache kommt, müssen wir ihn klonen, 
            # damit wir nicht das Cache-Original überschreiben
            if base_data is not None:
                data = base_data.clone()
            else:
                data = None
        else:
            # Graph direkt neu berechnen (kein Klonen nötig, da es ein neues Objekt ist)
            data = smiles_to_graph(smiles)
            
        if data is not None:
            data.gene = torch.tensor(gen_array[i], dtype=torch.float32).unsqueeze(0)
            data.y_true = torch.tensor([y_array[i]], dtype=torch.float32)
            dataset.append(data)

    return dataset

def set_seed(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  
    #torch.use_deterministic_algorithms(True, warn_only=True)

# ==========================================
# 3. Models
# ==========================================

class ModularTokenEncoder(nn.Module):
    """Encodes disjoint subsets of feature columns into separate functional tokens efficiently."""
    def __init__(self, feature_subsets_indices, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_tokens = len(feature_subsets_indices)
        for i, idx in enumerate(feature_subsets_indices):
            self.register_buffer(f'idx_{i}', torch.as_tensor(idx, dtype=torch.long))

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(idx), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for idx in feature_subsets_indices
        ])

    def forward(self, x):
        token_list = []
        for i, proj in enumerate(self.projections):
            idx = getattr(self, f'idx_{i}')
            sub_x = x[:, idx]
            token_list.append(proj(sub_x))
        return torch.stack(token_list, dim=1)

class GCNDrugEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x
class GATDrugEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, dropout=0.1, heads=2):
        super().__init__()
        
        # Calculate the output dimension per head. 
        # We concatenate the heads, so out_channels * heads must equal hidden_dim.
        out_channels = hidden_dim // heads
        
        # Layer 1: Applies multi-head graph attention.
        # Dropout here is applied to the attention coefficients.
        self.conv1 = GATConv(node_feature_dim, out_channels, heads=heads, dropout=dropout)
        
        # Layer 2: Hidden attention layer.
        self.conv2 = GATConv(hidden_dim, out_channels, heads=heads, dropout=dropout)
        
        # Layer 3: Final layer. We use a single head and disable concatenation 
        # to ensure the output dimension perfectly matches hidden_dim.
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Pass node features and edge indices through the attention layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        
        # Returns the context-aware atom embeddings via attention weights
        return x
class GINDrugEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        # GIN requires a neural network (MLP) for each layer to update node representations.
        # This makes it theoretically as powerful as the Weisfeiler-Lehman graph test.
        
        # MLP for Layer 1
        nn1 = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # train_eps=True allows the model to learn the central node weighting (epsilon)
        self.conv1 = GINConv(nn1, train_eps=True)

        # MLP for Layer 2
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2, train_eps=True)

        # MLP for Layer 3
        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv3 = GINConv(nn3, train_eps=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Pass node features through the GIN layers and MLPs
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        
        # Returns highly expressive atom embeddings 
        return x

class GroundedCrossAttentionModel(nn.Module):
    def __init__(self, gene_token_indices, hidden_dim=64, num_heads=4, dropout=0.1, gnn_type="GCN", molclr_backbone=None, freeze_molclr=True):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.use_molclr = molclr_backbone is not None

        self.gene_encoder = ModularTokenEncoder(gene_token_indices, hidden_dim, dropout)

        if self.use_molclr:
            print("Using pre-trained MolCLR backbone for drug encoding.")
            self.drug_encoder = molclr_backbone
            # --- NEU: Adapter-Schicht, um von 300 auf 64 Dimensionen zu kommen ---
            #self.molclr_proj = nn.Linear(300, hidden_dim)
            # this is the nonlinear version
            self.molclr_proj = nn.Sequential(
                nn.Linear(300, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )

            if freeze_molclr:
                print("Freezing MolCLR backbone parameters.")
                for param in self.drug_encoder.parameters():
                    param.requires_grad = False
        else:
            if gnn_type == "GCN":
                self.drug_encoder = GCNDrugEncoder(node_feature_dim=4, hidden_dim=hidden_dim, dropout=dropout)
            elif gnn_type == "GAT":
                self.drug_encoder = GATDrugEncoder(node_feature_dim=4, hidden_dim=hidden_dim, dropout=dropout)
            elif gnn_type == "GIN":
                self.drug_encoder = GINDrugEncoder(node_feature_dim=4, hidden_dim=hidden_dim, dropout=dropout)
            else:
                raise ValueError(f"Unknown GNN-Typ: {gnn_type}")

        self.cross_attn_gene_to_drug = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_drug_to_gene = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_gene = nn.LayerNorm(hidden_dim)
        self.norm_drug = nn.LayerNorm(hidden_dim)

        # OPTIMIERUNG 2: Feed-Forward Networks (Transformer Architektur)
        '''self.ffn_gene = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ffn_drug = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm_gene_2 = nn.LayerNorm(hidden_dim)
        self.norm_drug_2 = nn.LayerNorm(hidden_dim)'''

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            #nn.Sigmoid() # only with MinMax Scaler
        )

        self.last_attn_weights = None

    def forward(self, x_gene, drug_batch, save_attention=False):
        gene_tokens = self.gene_encoder(x_gene)

        if self.use_molclr:
            # preprocessing
            drug_batch.x = drug_batch.x.long()
            drug_batch.edge_attr = drug_batch.edge_attr.long()

            drug_batch.x[:, 0] = torch.clamp(drug_batch.x[:, 0], min=0, max=119)
            drug_batch.x[:, 1] = torch.clamp(drug_batch.x[:, 1], min=0, max=2)
            drug_batch.edge_attr[:, 0] = torch.clamp(drug_batch.edge_attr[:, 0], min=0, max=3)
            drug_batch.edge_attr[:, 1] = torch.clamp(drug_batch.edge_attr[:, 1], min=0, max=2)

            x_drug_nodes, _ = self.drug_encoder(drug_batch)
            x_drug_nodes = self.molclr_proj(x_drug_nodes)
        
        else:
            x_drug_nodes = self.drug_encoder(drug_batch.x, drug_batch.edge_index)

        drug_tokens, valid_mask = to_dense_batch(x_drug_nodes, drug_batch.batch)
        padding_mask = ~valid_mask

        # cross attention
        gene_attn_out, gene_attn_weights = self.cross_attn_gene_to_drug(
            query=gene_tokens, key=drug_tokens, value=drug_tokens,
            key_padding_mask=padding_mask
        )
        
        drug_attn_out, drug_attn_weights = self.cross_attn_drug_to_gene(
            query=drug_tokens, key=gene_tokens, value=gene_tokens
        )

        if save_attention:
            self.last_attn_weights = {
                'gene_to_drug': gene_attn_weights.detach().cpu().numpy(),
                'drug_to_gene': drug_attn_weights.detach().cpu().numpy()
            }

        gene_context = self.norm_gene(gene_tokens + gene_attn_out)
        drug_context = self.norm_drug(drug_tokens + drug_attn_out)

        gene_pooled = gene_context.mean(dim=1)
        
        num_real_atoms = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        drug_pooled = (drug_context * valid_mask.unsqueeze(-1)).sum(dim=1) / num_real_atoms
        
        fused = torch.cat([gene_pooled, drug_pooled], dim=1)

        return self.regressor(fused).squeeze(-1)