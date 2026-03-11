import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import NeuroLongitudinalDataset, collate_fn
from src.model import AtrophyPINN, KineticPINN
from src.plot_results import plot_pinn_trajectory

def train_pinn_model(config_name, model, dataloader, epochs=20, lambda_physics=0.5):
    print(f"Starting training for {config_name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss(reduction='none')

    model.train()
    for epoch in range(epochs):
        epoch_data_loss = 0
        epoch_phys_loss = 0

        for x_seq, y_seq, time_seq, mask in dataloader :
            for batch_idx in range(x_seq.size(0)):
                valid_len = mask[batch_idx].sum().item()
                if valid_len == 0: continue

                x_patient = x_seq[batch_idx, :valid_len]
                y_patient = y_seq[batch_idx, :valid_len]
                t_patient = time_seq[batch_idx, :valid_len].unsqueeze(1)

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
        print(f"Epoch [{epoch+1}/{epochs}] | Data MSE: {epoch_data_loss/num_batches:.4f} | Physics Loss : {epoch_phys_loss/num_batches:.6f}")
    
    return model

def main():
    try:
        oasis_dataset = NeuroLongitudinalDataset(
            csv_file='data/oasis_longitudinal.csv',
            id_col='Subject ID',
            time_col='Visit',
            feature_cols=['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'MR Delay'],
            target_cols=['nWBV']
        )
        oasis_loader = DataLoader(oasis_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        atrophy_model = AtrophyPINN(input_size=7)
        trained_atrophy_model = train_pinn_model("OASIS-2 (AtrophyPINN)", atrophy_model, oasis_loader, epochs=15)
        
        sample_x_seq, sample_y_seq, sample_t_seq, mask = next(iter(oasis_loader))
        valid_len = mask[0].sum().item()
        
        patient_x = sample_x_seq[0, 0] 
        actual_t = sample_t_seq[0, :valid_len]
        actual_y = sample_y_seq[0, :valid_len, 0] 
        
        plot_pinn_trajectory(trained_atrophy_model, patient_x, actual_t, actual_y, feature_name="nWBV", is_atrophy=True)

    except FileNotFoundError:
        print("Oasis csv not found.")

    try:
        amp_dataset = NeuroLongitudinalDataset(
            csv_file='data/train_clinical_data.csv',
            id_col='patient_id',
            time_col='visit_month',
            feature_cols=['visit_month'],
            target_cols=['updrs_1', 'updrs_2', 'updrs_3']
        )
        amp_loader = DataLoader(amp_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        kinetic_model = KineticPINN(input_size=2, output_size=3)
        trained_kinetic_model = train_pinn_model("AMP-PD (KineticPINN)", kinetic_model, amp_loader, epochs=15)
        
        sample_x_seq, sample_y_seq, sample_t_seq, mask = next(iter(amp_loader))
        valid_len = mask[0].sum().item()
        
        patient_x = sample_x_seq[0, 0] if sample_x_seq.size(2) > 0 else torch.empty(0) 
        actual_t = sample_t_seq[0, :valid_len]
        actual_y = sample_y_seq[0, :valid_len] 
        
        plot_pinn_trajectory(trained_kinetic_model, patient_x, actual_t, actual_y, feature_name="UPDRS", is_atrophy=False)

    except FileNotFoundError:
        print("AMP-PD csv not found.")

if __name__ == "__main__":
    main()