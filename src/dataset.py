import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

class NeuroLongitudinalDataset(Dataset):
    def __init__(self, csv_file, id_col, time_col, feature_cols, target_cols):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.drop_duplicates(subset=[id_col, time_col], keep='last')
        self.id_col = id_col
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        for col in self.feature_cols + self.target_cols:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        if 'M/F' in self.df.columns:
            self.df['M/F'] = self.df['M/F'].map({'M': 1, 'F': 0}).fillna(0)

        self.scaler_x = StandardScaler()
        self.df[self.feature_cols] = self.scaler_x.fit_transform(self.df[self.feature_cols])

        self.subjects = self.df[self.id_col].unique()
        self.grouped = self.df.groupby(self.id_col)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        patient_data = self.grouped.get_group(subject_id).sort_values(by=self.time_col)

        x =  torch.tensor(patient_data[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(patient_data[self.target_cols].values, dtype=torch.float32)
        time_steps = torch.tensor(patient_data[self.time_col].values, dtype=torch.float32)

        return x, y, time_steps

def collate_fn(batch):
    """Pads patient sequences to match the longest visit history in the current batch."""
    xs, ys, times = zip(*batch)
    
    # Pad features and targets
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    
    # THE FIX: We must also pad the time sequences into a 2D tensor
    t_padded = pad_sequence(times, batch_first=True, padding_value=0.0)
    
    # Create a boolean mask so the loss function ignores padded zero-visits
    lengths = torch.tensor([len(x) for x in xs])
    mask = torch.arange(x_padded.size(1))[None, :] < lengths[:, None]
    
    # Return t_padded instead of the raw times tuple
    return x_padded, y_padded, t_padded, mask