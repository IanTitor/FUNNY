import h5py
import numpy as np
import torch as pt
from torch.utils.data import Dataset
from tqdm import tqdm


pchange = lambda x: x.diff(dim=0) / x[:-1]
scale = lambda x: pt.log(9 * x.abs() + 1) * x.sign()
iscale = lambda x: (pt.exp(x) - 1) / 9


class StockDataset(Dataset):
    def __init__(self, h5_file_path, seq_length=1000, seq_stride=1):
        self.seq_length = seq_length
        self.seq_stride = seq_stride
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.valid_keys = []
        self.seq_counts = []
        # Precompute valid datasets and their number of sequences
        for key in self.h5_file.keys():
            n = self.h5_file[key].shape[0]
            if n >= seq_length:
                count = (n - seq_length) // seq_stride + 1
                self.valid_keys.append(key)
                self.seq_counts.append(count)
        # cumulative counts to map global index to dataset and offset
        self.cum_seq_counts = np.cumsum(self.seq_counts)

    def __len__(self):
        return int(self.cum_seq_counts[-1])
    
    def __getitem__(self, index):
        # Find which dataset this index belongs to using binary search (no python loop)
        dataset_idx = np.searchsorted(self.cum_seq_counts, index, side='right')
        local_index = (index if dataset_idx == 0 else index - self.cum_seq_counts[dataset_idx - 1]) * self.seq_stride
        
        key = self.valid_keys[dataset_idx]
        data = self.h5_file[key]
        # Retrieve the sequence slice without looping
        seq = pt.tensor(data[local_index: local_index + self.seq_length], dtype=pt.float)

        seq[:, -2] += 1e-12
        seq[1:, :-1] = scale(pchange(seq[:, :-1]))
        seq[:, -1] /= 1800  # 1800 is 30 mins
        seq[:, -1] -= seq[1, -1]
        return seq[1:].nan_to_num(0)

if __name__ == "__main__":
    ds = StockDataset("../data/dataset.h5", 1024, 512)
    c = []
    for i in tqdm(range(len(ds)), desc="Testing...", unit="sample"):
        c.append(ds[i].isnan().any())
    print(f"{sum(c)} failed")
