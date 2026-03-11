import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import NeuroLongitudinalDataset, collate_fn
from src.model import CoupledCJDPINN
from src.plot_results import plot_coupled_cjd_trajectory

def train_cjd_pinn():
    print("Training Coupled PINN for CJD data")

    try:
        dataset = NeuroLongitudinalDataset(
            csv_file='data/synthetic_cjd.csv',
            id_col='Patient_ID',
            time_col='Visit_Month',
            feature_cols=['Age', 'Genotype_Num'],
            target_cols=['PrPC_Normal', 'PrPSc_Burden']
        )

    except FileNotFoundError:
        print("Error: CJD csv not found")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = CoupledCJDPINN(input_size=2)
    optimizer= optim.Adam(model.parameters(), lr=0.002)
    mse_loss = nn.MSELoss(reduction='none')

    lambda_physics = 0.5
    epochs = 20

    model.train()
    for epoch in range(epochs):
        epoch_data_loss = 0
        epoch_phys_loss = 0

        for x_seq, y_seq, times_seq, mask in dataloader:
            for batch_idx in range(x_seq.size(0)):
                valid_len = mask[batch_idx].sum().item()
                if valid_len == 0: continue

                x_patient = x_seq[batch_idx, :valid_len]
                y_patient = y_seq[batch_idx, :valid_len]
                t_patient = times_seq[batch_idx, :valid_len].unsqueeze(1)

                optimizer.zero_grad()

                predictions = model(x_patient, t_patient)
                data_loss = torch.mean(mse_loss(predictions, y_patient))
                phys_loss = model.compute_physics_loss(x_patient, t_patient)

                total_loss = data_loss + (lambda_physics * phys_loss)

                total_loss.backward()
                optimizer.step()

                epoch_data_loss += data_loss.item()
                epoch_phys_loss += phys_loss.item()

        num_batches = len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Data MSE: {epoch_data_loss/num_batches:.4f} | Physics Loss: {epoch_phys_loss/num_batches:.6f}")

    model.eval()
    sample_x_seq, sample_y_seq, sample_t_seq, mask = next(iter(dataloader))
    valid_len = mask[0].sum().item()

    patient_x = sample_x_seq[0,0]
    actual_t = sample_t_seq[0, :valid_len]
    actual_y = sample_y_seq[0, :valid_len]

    plot_coupled_cjd_trajectory(model, patient_x, actual_t, actual_y)

if __name__ == "__main__":
    train_cjd_pinn()