import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset


def load_set(name):
    with np.load(name) as file:
        chunks = file['chunks']
        names = np.concatenate([file[f'names_{chunk_id}'] for chunk_id in chunks])
        fea = np.concatenate([file[f'fea_{chunk_id}'] for chunk_id in chunks], 0)
        lab = np.concatenate([file[f'lab_{chunk_id}'] for chunk_id in chunks])
        end_index = np.concatenate([file[f'end_index_{chunk_id}'] for chunk_id in chunks])
    return names, fea, lab, end_index


class TimitTrainSet(TensorDataset):
    def __init__(self):
        _, fea, lab, _ = load_set('dataset_tr.npz')
        super().__init__(torch.from_numpy(fea), torch.from_numpy(lab))

        self.num_fea = fea.shape[1]
        self.num_out = int(lab.max() - lab.min() + 1)

    def get_loader(self, batch_size):
        return DataLoader(self, batch_size, shuffle=True, pin_memory=True)


class TimitTestSet(Dataset):
    def __init__(self):
        pass
