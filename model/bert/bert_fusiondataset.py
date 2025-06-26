import torch
from torch.utils.data import Dataset
class BertFusionDataset(Dataset):
    def __init__(self, input_ids, attention_mask, linguistic_feats, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.linguistic_feats = torch.tensor(linguistic_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'linguistic': self.linguistic_feats[idx],
            'labels': self.labels[idx]
        }