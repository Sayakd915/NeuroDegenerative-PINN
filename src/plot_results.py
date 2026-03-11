import matplotlib.pyplot as plt
import torch

def plot_baseline_failures(epoch_losses, model, sample_x, sample_y, time_steps, feature_name="nWBV", dataset_name="dataset"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='red', label="LSTM Baseline")
    ax1.set_title("Training Convergence: Data Scarcity Plateau", fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Mean Squared Error (MSE)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    model.eval()
    with torch.no_grad():
        predictions = model(sample_x.unsqueeze(0)).squeeze(0).numpy()
    
    actuals = sample_y.numpy()
    times = time_steps.numpy()

    ax2.scatter(times, actuals, color='black', s=80, label="Actual Clinical Visits", zorder=3)
    ax2.plot(times, predictions, color='red', linestyle='--', linewidth=2, label="LSTM Prediction")
    
    ax2.set_title(f"Patient Trajectory: Predicted {feature_name}", fontweight='bold')
    ax2.set_xlabel("Time Step (Delay / Months)")
    ax2.set_ylabel(f"Normalized Score")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_baseline_failure_plot.png", dpi=300)
    print(f"Saved baseline failure plot for {dataset_name}.")



def plot_pinn_trajectory(model, patient_x, actual_t, actual_y, feature_name, is_atrophy=True):

    
    model.eval()
    
    max_time = actual_t.max().item() if len(actual_t) > 0 else 1.0
    continuous_t = torch.linspace(0, max_time * 1.2, steps=100).unsqueeze(1) 
    
    if patient_x.nelement() > 0:
        continuous_x = patient_x.unsqueeze(0).repeat(100, 1)
    else:
        continuous_x = torch.empty((100, 0))
    
    with torch.no_grad():
        continuous_predictions = model(continuous_x, continuous_t).numpy()
    
    actual_t_np = actual_t.numpy()
    actual_y_np = actual_y.numpy()
    continuous_t_np = continuous_t.numpy().flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    if is_atrophy:
        ax.scatter(actual_t_np, actual_y_np, color='black', s=80, label="Actual Clinical Visits", zorder=3)
        ax.plot(continuous_t_np, continuous_predictions[:, 0], color='blue', linewidth=2.5, 
                label="PINN Continuous Trajectory (dV/dt = -kV)")
    else:
        colors = ['blue', 'green', 'purple']
        for i in range(actual_y_np.shape[1]):
            ax.scatter(actual_t_np, actual_y_np[:, i], color=colors[i], s=60, marker='o', 
                       label=f"Actual {feature_name}_{i+1}")
            ax.plot(continuous_t_np, continuous_predictions[:, i], color=colors[i], linewidth=2.5, linestyle='--',
                    label=f"PINN {feature_name}_{i+1} Trajectory (dP/dt = kP)")

    title = f"OASIS-2: Structural Atrophy (PINN)" if is_atrophy else f"AMP-PD: Kinetic Growth (PINN)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (Months / Delay)")
    ax.set_ylabel(f"Normalized Score ({feature_name})")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.axvspan(max_time, max_time * 1.2, color='gray', alpha=0.2, label='Forecasting Zone')
    
    plt.tight_layout()
    filename = f"{'OASIS' if is_atrophy else 'AMP-PD'}_PINN_Success.png"
    plt.savefig(filename, dpi=300)
    print(f"Graph saved as '{filename}'")


def plot_coupled_cjd_trajectory(model, patient_x, actual_t, actual_y):
    model.eval()

    max_time = actual_t.max().item() if len(actual_t) > 0 else 24.0
    continuous_t = torch.linspace(0, max_time * 1.2, steps=100).unsqueeze(1)
    continuous_x = patient_x.unsqueeze(0).repeat(100, 1)

    with torch.no_grad():
        continuous_predictions = model(continuous_x, continuous_t).numpy()
        learned_k = model.get_patient_k(patient_x.unsqueeze(0)).item()
    
    actual_t_np = actual_t.numpy()
    actual_y_np = actual_y.numpy()
    continuous_t_np = continuous_t.numpy().flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(actual_t_np, actual_y_np[:,0], color='blue', s=80, label="Actual PrPC")
    ax.plot(continuous_t_np, continuous_predictions[:,0], color='blue', linewidth=2.5, label="PINN PrPC trajectory (dM/dt) ")
    ax.scatter(actual_t_np, actual_y_np[:,1], color='red', s=80, label="Actual PrPSc")
    ax.plot(continuous_t_np, continuous_predictions[:,1], color='red', linewidth=2.5, label="PINN PrPSc trajectory (dP/dt) ")
    ax.set_title(f"Coupled CJD Progression (PINN learned k = {learned_k:.4f})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Protein Concentration")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.axvspan(max_time, max_time * 1.2, color='gray', alpha=0.2, label='Forecasting Zone')

    plt.tight_layout()
    plt.savefig("CJD_Coupled_PINN_Success.png", dpi=300)
    print("Graph saved as 'CJD_Coupled_PINN_Success.png'")