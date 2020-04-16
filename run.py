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

from olympus.datasets import Dataset, SplitDataset, DataLoader
from olympus.utils import option, new_seed

from timit_mlp.dataset import TimitTestDataSampler
from timit_mlp.model import MLP


def test(net, loader, device, *, write_posts=False, out_folder=None, count_file=None):
    if not write_posts:
        assert out_folder is None and count_file is None
    if write_posts:
        assert out_folder is not None and count_file is not None

    if write_posts:
        # set folder for posteriors ark
        post_file = kaldi_io.open_or_fd(out_folder + '/pout_test.ark', 'wb')
        counts = load_counts(count_file)

    errs = []
    losses = []
    lens = []
    net.eval()
    # Reading dev-set sentence by sentence
    for name, fea, lab in loader:
        inp = fea.to(device, dtype=torch.float)
        lab = lab.to(device)

        with torch.no_grad():
            loss, err, pout, pred = net(inp, lab)

        if write_posts:
            # writing the ark containing the normalized posterior probabilities (needed for kaldi decoding)
            kaldi_io.write_mat(
                post_file,
                pout.data.to('cpu').numpy() - np.log(counts / np.sum(counts)),
                name[0])

        losses.append(loss.item())
        errs.append(err.item())
        lens.append(inp.shape[0])

    avg_loss = sum(losses) / len(losses)
    avg_err = sum(errs) / sum(lens)

    if write_posts:
        post_file.close()

    return avg_loss, avg_err


def main():
    # Reading options in cfg file
    options = read_opts()

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
        os.makedirs(options.out_folder + '/te')
        os.makedirs(options.out_folder + '/dev')

    # copy cfg file into the output folder
    copyfile(options.cfg, options.out_folder + '/conf.cfg')

    # Setting torch seed
    torch.manual_seed(seed)

    # Creating the res file
    res_file = open(options.out_folder + '/results.res', "w")

    batch_size = options.optimization.batch_size

    dataset = Dataset('timit', path=f'{options.out_folder}/../')
    splits = SplitDataset(dataset, split_method='original')
    loader = DataLoader(
        splits,
        sampler_seed=new_seed(sampler=1),
        batch_size=batch_size
    )

    train_loader = loader.train()
    dev_loader = loader.valid(batch_sampler=TimitTestDataSampler())
    test_loader = loader.test(batch_sampler=TimitTestDataSampler())

    input_size, num_classes = loader.get_shapes()

    net = MLP(input_dim=input_size,
              num_classes=num_classes,
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

        for _, inp, lab in train_loader:
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

        # ---EVALUATION OF DEV---#
        loss_dev, err_dev = test(
            net, dev_loader, options.device,
            **(dict(write_posts=True,
                    out_folder=options.out_folder + '/te/',
                    count_file=options.data.count_file)
               if ep == num_epochs else {}))

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
        loss_te, err_te = test(
            net, test_loader, options.device,
            **(dict(write_posts=True,
                    out_folder=options.out_folder + '/test/',
                    count_file=options.data.count_file)
               if ep == num_epochs else {}))

        print(
            f'epoch {ep} training_cost={loss_tr}, training_error={err_tr}, '
            f'dev_error={err_dev}, test_error={err_te}, learning_rate={lr_ep}, '
            f'execution_time(s)={end_epoch - start_epoch}')
        res_file.write(
            f'epoch {ep} training_cost={loss_tr}, training_error={err_tr}, '
            f'dev_error={err_dev}, test_error={err_te}, learning_rate={lr_ep}, '
            f'execution_time(s)={end_epoch - start_epoch}')
    res_file.close()
    # Model Saving
    torch.save(net.state_dict(), expandvars(options.out_folder) + '/model.pkl')

    decode('dev', output_dir=options.out_folder)
    decode('test', output_dir=options.out_folder)


def decode(subset, output_dir):
    import subprocess
    subprocess.Popen(
        f'./decode_dnn_TIMIT.sh '
        f'$KALDI_ROOT/egs/timit/s5/exp/tri3/graph '
        f'$KALDI_ROOT/egs/timit/s5/data/{subset}/ '
        f'$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali '
        f'../{output_dir}/decoding_{subset} '
        f'"cat ../{output_dir}/{subset}/pout_test.ark"',
        cwd='./kaldi_decoding_scripts/',
        shell=True)


if __name__ == "__main__":
    main()
