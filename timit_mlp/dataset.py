import numpy as np
import torch
from bisect import bisect
from torch.utils.data import TensorDataset, DataLoader, Dataset, BatchSampler


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
    def __init__(self, subset):
        super().__init__()
        names, self.fea, self.lab, end_index = load_set(f'dataset_{subset}.npz')
        self.names = list(names)
        self.end_index = list(end_index)

    def __len__(self):
        return self.fea.shape[0]

    def __getitem__(self, item):
        name_ind = bisect(self.end_index, item)
        return self.names[name_ind], torch.from_numpy(self.fea[item]), torch.from_numpy(self.lab[item])

    def get_loader(self):
        return DataLoader(self, batch_sampler=TimitTestDataSampler(self))


class TimitTestDataSampler(BatchSampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.names)

    def __iter__(self):
        inds = list(self.dataset.end_index)
        index_pairs = zip([0] + inds, inds + [-1])
        for name, (start_ind, end_ind) in zip(self.dataset.names, index_pairs):
            yield list(range(start_ind, end_ind))
