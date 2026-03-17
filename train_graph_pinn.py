import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np 
import os
from src.model import HeirarchicalGraphPINN
from src.plot_graph_pinn import plot_regional_trajectories, plot_loss_convergence, plot_connectome_adjacency

def train_graph_pinn():
    print("\n Initializing Graph PINN")
    
    try:
        dataset = torch.load("data/processed_brain_graphs.pt", weights_only=False)
        print("\n Dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(" Applying Per-Node Biological Scaling...")
    all_x = torch.stack([g.x for g in dataset]) 
    node_maxes = all_x.max(dim=0)[0] 
    node_maxes[node_maxes == 0] = 1.0
    
    for g in dataset:
        g.x = g.x / node_maxes
        g.y_seq = g.y_seq / node_maxes

    model = HeirarchicalGraphPINN(num_node_features=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    mse_loss = nn.MSELoss()

    lambda_physics = 0.1
    epochs = 30

    random.seed(42)
    indices = list(range(len(dataset)))

    history_data_loss = []
    history_physics_loss = []

    model.train()
    for epoch in range(epochs):
        epoch_data_loss = 0
        epoch_physics_loss = 0
        random.shuffle(indices)

        valid_patients = 0

        for idx in indices:
            graph = dataset[idx]
            x_baseline = graph.x
            y_seq = graph.y_seq
            times = graph.times_seq
            L_matrix = graph.L_matrix
            edge_index = graph.edge_index
            edge_weight = graph.edge_attr

            if len(times) < 2: continue

            valid_patients += 1
            optimizer.zero_grad()

            patient_data_loss = 0
            patient_physics_loss = 0

            for i, t_val in enumerate(times):
                t_val_norm = t_val / 100.0
                t_tensor = torch.tensor([[t_val_norm]], dtype=torch.float32)
                
                v_pred = model(x_baseline, t_tensor, edge_index, edge_weight)
                patient_data_loss += mse_loss(v_pred, y_seq[i])
                
                max_time_norm = (times[-1] + 5.0) / 100.0
                t_colloc = torch.rand((1, 1), requires_grad=True) * max_time_norm
                patient_physics_loss += model.compute_physics_loss(x_baseline, t_colloc, edge_index, edge_weight, L_matrix)
            
            total_loss = (patient_data_loss / len(times)) + (lambda_physics * (patient_physics_loss / len(times)))
            total_loss.backward()
            optimizer.step()

            epoch_data_loss += (patient_data_loss.item() / len(times))
            epoch_physics_loss += (patient_physics_loss.item() / len(times))

            if valid_patients >= 50:
                break

        avg_data = epoch_data_loss / valid_patients
        avg_physics = epoch_physics_loss / valid_patients
        history_data_loss.append(avg_data)
        history_physics_loss.append(avg_physics)

        print(f"Epoch [{epoch+1}/{epochs}] | Graph MSE: {avg_data:.6f} | Physics Loss: {avg_physics:.6f}")
    
    print("\n Training completed")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/graph_pinn_model.pth")
    print("\n Model saved to models/graph_pinn_model.pth")

    plot_loss_convergence(history_data_loss, history_physics_loss)
    
    num_nodes = graph.x.size(0)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    edges = graph.edge_index.numpy()
    weights = graph.edge_attr.numpy()
    for i in range(edges.shape[1]):
        adj_matrix[edges[0, i], edges[1, i]] = weights[i]
    plot_connectome_adjacency(adj_matrix, num_nodes=50)

    sample_graph = next(g for g in dataset if len(g.times_seq) >= 4)
    node_indices = [10, 50, 100]
    region_names = ['Region A', 'Region B', 'Region C']
    plot_regional_trajectories(model, sample_graph, node_indices, region_names, patient_id="Sample")
    
    print("Plots saved as PDF files in root directory")

if __name__ == "__main__":
    train_graph_pinn()