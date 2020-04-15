# pytorch_speechMLP 
# Mirco Ravanelli 
# Montreal Institute for Learning Algoritms (MILA)
# University of Montreal 

# January 2018

# Description: 
# This code implements with pytorch a basic MLP  for speech recognition. 
# It exploits an interface to  kaldi for feature computation and decoding. 
# How to run it:
# python run.py --cfg TIMIT_MLP_mfcc.toml

import kaldi_io
import numpy as np
import torch
import timeit
import os
from os.path import expandvars
from shutil import copyfile
from timit_mlp.data_io import load_counts, read_opts
from torch import optim

from timit_mlp.dataset import TimitTrainSet
from timit_mlp.model import MLP


def main():
    # Reading options in cfg file
    options = read_opts()

    # Reading count file from kaldi
    count_file = options.data.count_file

    # reading architectural options
    seed = options.seed

    # reading optimization options
    num_epochs = options.optimization.num_epochs
    lr = options.optimization.lr
    halving_factor = options.optimization.halving_factor
    improvement_threshold = options.optimization.improvement_threshold

    # Create output folder
    if not os.path.exists(options.out_folder):
        os.makedirs(options.out_folder)

    # copy cfg file into the output folder
    copyfile(options.cfg, options.out_folder + '/conf.cfg')

    # Setting torch seed
    torch.manual_seed(seed)

    # Creating the res file
    res_file = open(options.out_folder + '/results.res', "w")

    batch_size = options.optimization.batch_size

    train_dataset = TimitTrainSet()
    train_loader = train_dataset.get_loader(batch_size)

    net = MLP(input_dim=train_dataset.num_fea,
              num_classes=train_dataset.num_out,
              options=options.architecture)
    net.to(options.device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for ep in range(1, num_epochs + 1):
        # ---TRAINING LOOP---#
        err_sum = 0.0
        loss_sum = 0.0
        n_train_batches_tot = 0
        N_ex_tr_tot = 0
        start_epoch = timeit.default_timer()

        for inp, lab in train_loader:
            net.train()
            inp = inp.to(options.device, dtype=torch.float)
            lab = lab.to(options.device)
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
            inp = dev_fea[beg_snt:end_snt].to(options.device, dtype=torch.float)
            lab = dev_lab[beg_snt:end_snt].to(options.device)

            with torch.no_grad():
                loss, err, pout, pred = net(inp, lab)
            loss_sum = loss_sum + loss.item()
            err_sum = err_sum + err.item()

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

        if ep == num_epochs:
            # set folder for posteriors ark
            post_file = kaldi_io.open_or_fd(options.out_folder + '/pout_test.ark', 'wb')
            counts = load_counts(count_file)

        for i in range(n_te_snt):
            end_snt = te_end_index[i]
            inp = te_fea[beg_snt:end_snt].to(options.device, dtype=torch.float)
            lab = te_lab[beg_snt:end_snt].to(options.device)

            with torch.no_grad():
                loss, err, pout, pred = net(inp, lab)

            if ep == num_epochs:
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
