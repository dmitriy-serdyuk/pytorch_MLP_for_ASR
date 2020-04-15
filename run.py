# pytorch_speechMLP 
# Mirco Ravanelli 
# Montreal Institute for Learning Algoritms (MILA)
# University of Montreal 

# January 2018

# Description: 
# This code implements with pytorch a basic MLP  for speech recognition. 
# It exploits an interface to  kaldi for feature computation and decoding. 
# How to run it:
# python run.py --cfg TIMIT_MLP_mfcc.cfg

import kaldi_io
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import timeit
import os
from os.path import expandvars
from shutil import copyfile
from timit_mlp.data_io import load_counts, read_opts
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, N_hid, drop_rate, use_batchnorm, num_classes):
        super(MLP, self).__init__()
        self.N_hid = N_hid
        self.use_batchnorm = use_batchnorm

        # list initialization
        self.hidden = nn.ModuleList([])
        self.droplay = nn.ModuleList([])
        self.bnlay = nn.ModuleList([])
        self.criterion = nn.CrossEntropyLoss()

        curr_in_dim = input_dim
        for i in range(N_hid):
            fc = nn.Linear(curr_in_dim, hidden_dim)
            fc.weight = torch.nn.Parameter(
                torch.Tensor(hidden_dim, curr_in_dim).uniform_(-np.sqrt(0.01 / (curr_in_dim + hidden_dim)),
                                                               np.sqrt(0.01 / (curr_in_dim + hidden_dim))))
            fc.bias = torch.nn.Parameter(torch.zeros(hidden_dim))
            curr_in_dim = hidden_dim
            self.hidden.append(fc)
            self.droplay.append(nn.Dropout(p=drop_rate))
            if use_batchnorm:
                self.bnlay.append(nn.BatchNorm1d(hidden_dim, momentum=0.05))

        self.fco = nn.Linear(curr_in_dim, num_classes)
        self.fco.weight = torch.nn.Parameter(torch.zeros(num_classes, curr_in_dim))
        self.fco.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, lab):
        out = x
        for i in range(self.N_hid):
            fc = self.hidden[i]
            drop = self.droplay[i]

            if self.use_batchnorm:
                batchnorm = self.bnlay[i]
                out = drop(F.relu(batchnorm(fc(out))))
            else:
                out = drop(F.relu(fc(out)))

        out = self.fco(out)
        pout = F.log_softmax(out, dim=1)
        pred = pout.max(dim=1)[1]
        err = torch.sum((pred != lab.long()).float())
        loss = self.criterion(out, lab.long())  # note that softmax is included in nn.CrossEntropyLoss()
        return loss, err, pout, pred


def load_data(options):
    tr_fea_scp = options.tr_fea_scp.split(',')
    batch_size = int(options.batch_size)

    tr_fea = np.concatenate([np.load(f'dataset_tr_{chunk_id}.npz')['fea']
                             for chunk_id in range(len(tr_fea_scp))], 0)
    tr_lab = np.concatenate([np.load(f'dataset_tr_{chunk_id}.npz')['lab']
                             for chunk_id in range(len(tr_fea_scp))])
    train_dataset = TensorDataset(torch.from_numpy(tr_fea), torch.from_numpy(tr_lab))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)

    num_fea = tr_fea.shape[1]
    num_out = int(tr_lab.max() - tr_lab.min() + 1)

    dev_fea = np.load(f'dataset_dev.npz')['fea']
    dev_lab = np.load(f'dataset_dev.npz')['lab']
    dev_dataset = TensorDataset(torch.from_numpy(dev_fea), torch.from_numpy(dev_lab))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, pin_memory=True)

    te_fea = np.load(f'dataset_te.npz')['fea']
    te_lab = np.load(f'dataset_te.npz')['lab']
    test_dataset = TensorDataset(torch.from_numpy(te_fea), torch.from_numpy(te_lab))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    return train_loader, dev_loader, test_loader, num_fea, num_out


def main():
    # Reading options in cfg file
    options = read_opts()

    # Reading training data options
    tr_fea_scp = options.tr_fea_scp.split(',')

    # Reading count file from kaldi
    count_file = options.count_file

    # reading architectural options
    hidden_dim = int(options.hidden_dim)
    N_hid = int(options.N_hid)
    drop_rate = float(options.drop_rate)
    use_batchnorm = bool(int(options.use_batchnorm))
    seed = int(options.seed)
    use_cuda = bool(int(options.use_cuda))

    # reading optimization options
    N_ep = int(options.N_ep)
    batch_size = int(options.batch_size)
    lr = float(options.lr)
    halving_factor = float(options.halving_factor)
    improvement_threshold = float(options.improvement_threshold)
    save_gpumem = int(options.save_gpumem)

    # Create output folder
    if not os.path.exists(options.out_folder):
        os.makedirs(options.out_folder)

    # copy cfg file into the output folder
    copyfile(options.cfg, options.out_folder + '/conf.cfg')

    # Setting torch seed
    torch.manual_seed(seed)

    # Creating the res file
    res_file = open(options.out_folder + '/results.res', "w")

    train_loader, dev_loader, test_loader, num_fea, num_out = load_data(options)

    net = MLP(num_fea, hidden_dim, N_hid, drop_rate, use_batchnorm, num_out)
    if use_cuda:
        net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for ep in range(1, N_ep + 1):
        # ---TRAINING LOOP---#
        err_sum = 0.0
        loss_sum = 0.0
        n_train_batches_tot = 0
        N_ex_tr_tot = 0
        start_epoch = timeit.default_timer()

        for inp, lab in train_loader:
            net.train()
            if use_cuda:
                inp = inp.to('cuda', dtype=torch.float)
                lab = lab.to('cuda')
            optimizer.zero_grad()

            loss, err, pout, pred = net(inp, lab)
            loss.backward()
            optimizer.step()

            # Loss accumulation
            loss_sum = loss_sum + loss.item()
            err_sum = err_sum + err.item()
            n_train_batches_tot += 1
            N_ex_tr_tot += batch_size

        # Average Loss
        loss_tr = loss_sum / n_train_batches_tot
        err_tr = err_sum / N_ex_tr_tot

        end_epoch = timeit.default_timer()

        dev = np.load('dataset_dev.npz')
        dev_name = dev['names']
        dev_end_index = dev['end_index']
        dev_fea = torch.from_numpy(dev['fea'])
        dev_lab = torch.from_numpy(dev['lab'])
        # ---EVALUATION OF DEV---#
        beg_snt = 0
        err_sum = 0.0
        loss_sum = 0.0
        n_dev_snt = len(dev_name)
        net.eval()
        # Reading dev-set sentence by sentence
        for i in range(n_dev_snt):
            end_snt = dev_end_index[i]
            inp = Variable(dev_fea[beg_snt:end_snt], volatile=True)
            lab = Variable(dev_lab[beg_snt:end_snt], volatile=True)

            if save_gpumem and use_cuda:
                inp = inp.to('cuda', dtype=torch.float)
                lab = lab.to('cuda')

            loss, err, pout, pred = net(inp, lab)
            loss_sum = loss_sum + loss.data
            err_sum = err_sum + err.data

            beg_snt = dev_end_index[i]

        loss_dev = loss_sum / n_dev_snt
        err_dev = (err_sum / dev_fea.shape[0]).cpu().numpy()

        # Learning rate annealing (if improvement on dev-set is small)
        lr_ep = lr
        if ep == 1:
            err_dev_prev = err_dev
        else:
            err_dev_new = err_dev
            if ((err_dev_prev - err_dev_new) / err_dev_new) < improvement_threshold:
                lr_ep = lr
                lr = lr * halving_factor
            err_dev_prev = err_dev_new

        # set the next epoch learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---EVALUATION OF TEST---#
        test_set = np.load('dataset_te.npz')
        te_name = test_set['names']
        te_end_index = test_set['end_index']
        te_fea = torch.from_numpy(test_set['fea'])
        te_lab = torch.from_numpy(test_set['lab'])

        beg_snt = 0
        err_sum = 0.0
        loss_sum = 0.0
        n_te_snt = len(te_name)
        net.eval()

        if ep == N_ep:
            # set folder for posteriors ark
            post_file = kaldi_io.open_or_fd(options.out_folder + '/pout_test.ark', 'wb')
            counts = load_counts(count_file)

        for i in range(n_te_snt):
            end_snt = te_end_index[i]
            inp = Variable(te_fea[beg_snt:end_snt], volatile=True)
            lab = Variable(te_lab[beg_snt:end_snt], volatile=True)

            if save_gpumem and use_cuda:
                inp = inp.to('cuda', dtype=torch.float)
                lab = lab.to('cuda')

            loss, err, pout, pred = net(inp, lab)

            if ep == N_ep:
                # writing the ark containing the normalized posterior probabilities (needed for kaldi decoding)
                kaldi_io.write_mat(post_file, pout.data.cpu().numpy() - np.log(counts / np.sum(counts)), te_name[i])

            loss_sum = loss_sum + loss.data
            err_sum = err_sum + err.data

            beg_snt = te_end_index[i]

        loss_te = loss_sum / n_te_snt
        err_te = err_sum / te_fea.shape[0]

        print(
            f'epoch {ep} training_cost={loss_tr}, training_error={err_tr}, '
            f'dev_error={err_dev}, test_error={err_te}, learning_rate={lr_ep}, '
            f'execution_time(s)={end_epoch - start_epoch}')
        res_file.write(
            f'epoch {ep} training_cost={loss_tr}, training_error={err_tr}, '
            f'dev_error={err_dev}, test_error={err_te}, learning_rate={lr_ep}, '
            f'execution_time(s)={end_epoch - start_epoch}')

    post_file.close()
    res_file.close()
    # Model Saving
    torch.save(net.state_dict(), expandvars(options.out_folder) + '/model.pkl')

    # If everything went fine, you can run the kaldi phone-loop decoder:
    # cd kaldi_decoding_scripts
    # ./decode_dnn_TIMIT.sh /home/mirco/kaldi-trunk/egs/timit/s5/exp/tri3/graph /home/mirco/kaldi-trunk/egs/timit/s5/data/test/ /home/mirco/kaldi-trunk/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali /home/mirco/pytorch_exp/TIMIT_MLP_fmllr/decoding_test cat /home/mirco/pytorch_exp/TIMIT_MLP_fmllr/pout_test.ark


if __name__ == "__main__":
    main()
