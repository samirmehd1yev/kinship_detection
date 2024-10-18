import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, triplet_df, app):
        self.triplet_df = triplet_df
        self.app = app

    def __len__(self):
        return len(self.triplet_df)

    def __getitem__(self, idx):
        row = self.triplet_df.iloc[idx]
        anchor_img = self.app.get(row["Anchor"])  # Preprocess Anchor image
        positive_img = self.app.get(row["Positive"])  # Preprocess Positive image
        negative_img = self.app.get(row["Negative"])  # Preprocess Negative image
        label = row["ptype"]  # Get label (if needed)
        return anchor_img, positive_img, negative_img, label
