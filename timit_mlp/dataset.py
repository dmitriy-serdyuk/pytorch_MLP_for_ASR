import numpy as np
from bisect import bisect
from torch.utils.data import DataLoader, Dataset, BatchSampler


def load_set(name):
    with np.load(name) as file:
        chunks = file['chunks']
        names = np.concatenate([file[f'names_{chunk_id}'] for chunk_id in chunks])
        fea = np.concatenate([file[f'fea_{chunk_id}'] for chunk_id in chunks], 0)
        lab = np.concatenate([file[f'lab_{chunk_id}'] for chunk_id in chunks])
        end_indexes = [file[f'end_index_{chunk_id}'] for chunk_id in chunks]
        end_indexes_lens = [len(e) for e in end_indexes]
        shifts = [0] + list(np.cumsum(end_indexes_lens))[:-1]
        end_index = np.concatenate(
            [end_ind + shift for shift, end_ind in zip(shifts, end_indexes)])
    return names, fea, lab, end_index


class TimitSet(Dataset):
    def __init__(self, subset):
        super().__init__()
        names, self.fea, self.lab, end_index = load_set(f'dataset_{subset}.npz')
        self.names = list(names)
        self.end_index = list(end_index)

        self.num_fea = self.fea.shape[1]
        self.num_out = int(self.lab.max() - self.lab.min() + 1)

    def __len__(self):
        return self.fea.shape[0]

    def __getitem__(self, item):
        name_ind = bisect(self.end_index, item)
        return self.names[name_ind], self.fea[item], self.lab[item]

    def get_test_loader(self):
        return DataLoader(self, batch_sampler=TimitTestDataSampler(self))

    def get_train_loader(self, batch_size):
        return DataLoader(self, batch_size, shuffle=True, pin_memory=True)


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
