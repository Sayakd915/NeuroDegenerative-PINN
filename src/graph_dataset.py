import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
import warnings
warnings.filterwarnings("ignore")

class BrainNetworkDataset(Dataset):
    """
    Constructs PyTorch Geometric graphs from TADPOLE FreeSurfer parcellations.
    Builds the structural connectome using Population Co-Atrophy.
    """
    def __init__(self, tadpole_csv, dict_csv):
        super(BrainNetworkDataset, self).__init__()
        print("Initializing Brain Network Dataset...")
        
        self.df = pd.read_csv(tadpole_csv, low_memory=False)
        self.df_dict = pd.read_csv(dict_csv)
        
        volume_cols_df = self.df_dict[self.df_dict['TEXT'].str.contains('Volume', na=False, case=False)]
        vol_fldnames = volume_cols_df['FLDNAME'].tolist()
        
        all_node_cols = [c for c in vol_fldnames if c in self.df.columns]
        
        self.df = self.df[['PTID', 'Month'] + all_node_cols].copy()
        for col in all_node_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        missing_per_col = self.df[all_node_cols].isna().mean()
        self.node_cols = missing_per_col[missing_per_col < 0.9].index.tolist()
        self.num_nodes = len(self.node_cols)
        print(f"Kept {self.num_nodes} valid brain regions (dropped empty columns).")
        
        self.df = self.df[['PTID', 'Month'] + self.node_cols]
        
        self.df[self.node_cols] = self.df.groupby('PTID')[self.node_cols].transform(lambda x: x.ffill().bfill())
        
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=self.node_cols)
        print(f"Successfully rescued longitudinal visits. {len(self.df)} valid visits remaining (out of {initial_len}).")
        
        print("Computing Population Co-Atrophy Connectome...")
        volumes_only = self.df[self.node_cols].values
        
        corr_matrix = np.corrcoef(volumes_only.T) 
        
        self.adj_matrix = np.where(corr_matrix > 0.4, corr_matrix, 0.0)
        np.fill_diagonal(self.adj_matrix, 0) 
        
        self.edge_index, self.edge_weight = self._build_graph_edges(self.adj_matrix)
        self.L_matrix = self._compute_laplacian(self.adj_matrix)
        
        self.patient_ids = self.df['PTID'].unique()

    def _build_graph_edges(self, adj_matrix):
        """Converts an adjacency matrix into PyG COO edge format."""
        sources, targets, weights = [], [], []
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adj_matrix[i, j] > 0:
                    sources.append(i)
                    targets.append(j)
                    weights.append(adj_matrix[i, j])
                    
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weight

    def _compute_laplacian(self, adj_matrix):
        """Computes the normalized Graph Laplacian: L = I - D^(-1/2) * A * D^(-1/2)"""
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        d_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(degree_matrix)) + 1e-8)) 
        
        identity = np.eye(self.num_nodes)
        laplacian = identity - np.dot(d_inv_sqrt, np.dot(adj_matrix, d_inv_sqrt))
        
        return torch.tensor(laplacian, dtype=torch.float32)

    def len(self):
        return len(self.patient_ids)

    def get(self, idx):
        """Returns the full longitudinal trajectory for a specific patient as a Graph Sequence."""
        patient_id = self.patient_ids[idx]
        patient_data = self.df[self.df['PTID'] == patient_id].sort_values('Month')
        
        x_seq = torch.tensor(patient_data[self.node_cols].values, dtype=torch.float32).unsqueeze(2)
        
        x_seq = x_seq / 10000.0 
        
        times_seq = torch.tensor(patient_data['Month'].values, dtype=torch.float32)
        
        data = Data(x=x_seq[0], edge_index=self.edge_index, edge_attr=self.edge_weight)
        
        data.y_seq = x_seq       
        data.times_seq = times_seq 
        data.L_matrix = self.L_matrix 
        
        return data

if __name__ == "__main__":
    dataset = BrainNetworkDataset('data/TADPOLE_D1_D2.csv', 'data/TADPOLE_D1_D2_Dict.csv')
    torch.save([dataset[i] for i in range(len(dataset))], 'data/processed_brain_graphs.pt')
    sample_graph = dataset[0]
    print("\nPyG Graph Constructed")
    print(f"Nodes (Brain Regions): {sample_graph.x.size(0)}")
    print(f"Edges (White Matter Links): {sample_graph.edge_index.size(1)}")
    print(f"Patient Visits recorded: {sample_graph.times_seq.size(0)}")
    print("Graph data saved to data/processed_brain_graphs.pt")