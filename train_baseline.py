import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import NeuroLongitudinalDataset, collate_fn
from src.model import BaselineLSTM
from src.plot_results import plot_baseline_failures

def train_baseline(config_name, dataloader, input_size, output_size, epochs=15):
    print(f"Training baseline model for {config_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineLSTM(input_size,hidden_size=32,num_layers=1,output_size=output_size).to(device)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs_losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x,y,times,mask in dataloader:
            optimizer.zero_grad()
            predictions = model(x)

            loss = criterion(predictions,y)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = (loss * mask_expanded).sum() / mask_expanded.sum()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        epochs_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, epochs_losses
    
def main():
    try:
        oasis_dataset = NeuroLongitudinalDataset(
            csv_file="data/oasis_longitudinal.csv",
            id_col='Subject ID',
            time_col='Visit',
            feature_cols=['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'MR Delay'],
            target_cols=['nWBV', 'CDR']
        )

        oasis_loader = DataLoader(
            oasis_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )

        trained_model, losses = train_baseline(
            "OASIS-2",
            oasis_loader,
            input_size=6,
            output_size=2,
        )

        sample_x, sample_y, sample_times = oasis_dataset[2]
        plot_baseline_failures(
            losses,
            trained_model,
            sample_x,
            sample_y[:,0],
            sample_times,
            feature_name="nWBV",
            dataset_name="OASIS"
        )

    except FileNotFoundError:
        print("Oasis csv file not found")

    try:
        amp_dataset = NeuroLongitudinalDataset(
            csv_file='data/train_clinical_data.csv',
            id_col='patient_id',
            time_col='visit_month',
            feature_cols=['visit_month'],
            target_cols=['updrs_1', 'updrs_2', 'updrs_3']
        )

        amp_loader = DataLoader(
            amp_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )

        trained_model, losses = train_baseline(
            "AMP-PD",
            amp_loader,
            input_size=1,
            output_size=3,
        )

        sample_x, sample_y, sample_times = amp_dataset[2]
        plot_baseline_failures(
            losses,
            trained_model,
            sample_x,
            sample_y[:, 0],
            sample_times,
            feature_name="updrs_1",
            dataset_name="AMP-PD"
        )

    except FileNotFoundError:
        print("AMP-PD csv file not found")

if __name__ == "__main__":
    main()