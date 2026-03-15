import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

def plot_regional_trajectories(model, graph_data, node_indices, region_names, patient_id):
    """
    Plots the longitudinal forecasting of specific brain regions for a single patient.
    """
    model.eval()
    times = graph_data.times_seq.numpy()
    actual_volumes = graph_data.y_seq.numpy()
    
    max_time = times[-1] if len(times) > 0 else 24.0
    continuous_t = torch.linspace(0, max_time * 1.3, steps=100).unsqueeze(1)
    
    with torch.no_grad():
        pred_volumes = []
        for t_val in continuous_t:
            t_tensor = t_val.unsqueeze(0)
            v_pred = model(graph_data.x, t_tensor, graph_data.edge_index, graph_data.edge_attr)
            pred_volumes.append(v_pred.numpy())
        pred_volumes = np.array(pred_volumes)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(node_indices))
    
    for idx, (node_idx, color, name) in enumerate(zip(node_indices, colors, region_names)):
        ax.scatter(times, actual_volumes[:, node_idx, 0], color=color, s=80, edgecolor='black', zorder=3, label=f'{name} (Actual)')
        ax.plot(continuous_t.numpy(), pred_volumes[:, node_idx, 0], color=color, linewidth=2.5, zorder=2, label=f'{name} (PI-GNN)')

    ax.set_title(f"Spatiotemporal Forecasting (Patient {patient_id})", fontweight='bold')
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Normalized Regional Volume")
    
    ax.axvspan(max_time, max_time * 1.3, color='gray', alpha=0.15, label='Zero-Shot Forecasting')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.savefig(f"A_Star_Regional_Trajectory_PT_{patient_id}.pdf", format='pdf')
    plt.close()

def plot_connectome_adjacency(adj_matrix, num_nodes=50):
    """
    Visualizes the structural brain network (Population Co-Atrophy Graph).
    Only plots a subset of nodes for visual clarity in the paper.
    """
    subset_adj = adj_matrix[:num_nodes, :num_nodes]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(subset_adj, cmap="YlGnBu", cbar_kws={'label': 'Structural Edge Weight'}, 
                xticklabels=False, yticklabels=False, ax=ax)
    
    ax.set_title("Brain Structural Connectome (Graph Laplacian Core)", fontweight='bold')
    ax.set_xlabel(f"Brain Regions (Subset of {num_nodes})")
    ax.set_ylabel(f"Brain Regions (Subset of {num_nodes})")
    
    plt.savefig("A_Star_Connectome_Heatmap.pdf", format='pdf')
    plt.close()

def plot_loss_convergence(data_losses, phys_losses):
    """
    Dual-axis plot showing the harmonized convergence of Data MSE and Physics Residual.
    """
    epochs = range(1, len(data_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Training Epochs', fontweight='bold')
    ax1.set_ylabel('Data Loss (MSE)', color=color1, fontweight='bold')
    ax1.plot(epochs, data_losses, color=color1, linewidth=2.5, label='Structural MSE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Physics Loss (Network Diffusion)', color=color2, fontweight='bold')
    ax2.plot(epochs, phys_losses, color=color2, linewidth=2.5, linestyle='--', label='ODE Residual')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle('PI-GNN Optimization Dynamics', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.savefig("A_Star_Loss_Convergence.pdf", format='pdf')
    plt.close()