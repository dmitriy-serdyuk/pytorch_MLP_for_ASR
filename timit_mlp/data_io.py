import kaldi_io
import numpy as np
import configparser
from os.path import expandvars
from optparse import OptionParser


def load_dataset(fea_scp, fea_opts, lab_folder, lab_opts, left, right):
    fea = {k: m for k, m
           in kaldi_io.read_mat_ark(
               f'ark:copy-feats scp:{fea_scp} ark:- |{fea_opts}')}
    lab = {k: v for k, v
           in kaldi_io.read_vec_int_ark(
               f'gunzip -c {lab_folder}/ali*.gz | {lab_opts} {lab_folder}/final.mdl ark:- ark:-|')
           if k in fea}  # Note that I'm copying only the alignments of the loaded fea
    # This way I remove all the features without an alignment (see log file in alidir "Did not Succeded")
    fea = {k: v for k, v in fea.items()
           if k in lab}

    count = 0
    end_snt = 0
    end_index = []
    snt_name = []
    for k in sorted(fea.keys(), key=lambda k: len(fea[k])):
        if count == 0:
            count = 1
            fea_conc = fea[k]
            lab_conc = lab[k]
            end_snt = end_snt + fea[k].shape[0] - left
        else:
            fea_conc = np.concatenate([fea_conc, fea[k]], axis=0)
            lab_conc = np.concatenate([lab_conc, lab[k]], axis=0)
            end_snt = end_snt + fea[k].shape[0]

        end_index.append(end_snt)
        snt_name.append(k)

    end_index[-1] = end_index[-1] - right

    return snt_name, fea_conc, lab_conc, end_index


def context_window(fea, left, right):
    N_row = fea.shape[0]
    N_fea = fea.shape[1]
    frames = np.empty((N_row - left - right, N_fea * (left + right + 1)))

    for frame_index in range(left, N_row - right):
        right_context = fea[frame_index + 1:frame_index + right + 1].flatten()  # right context
        left_context = fea[frame_index - left:frame_index].flatten()  # left context
        current_frame = np.concatenate([left_context, fea[frame_index], right_context])
        frames[frame_index - left] = current_frame

    return frames


def load_chunk(fea_scp, fea_opts, lab_folder,
               lab_opts, left, right):
    # open the file
    data_name, data_set, data_lab, end_index = load_dataset(
        fea_scp, fea_opts, lab_folder, lab_opts, left, right)

    # Context window
    data_set = context_window(data_set, left, right)

    # mean and variance normalization
    data_set = (data_set - np.mean(data_set, axis=0)) / np.std(data_set, axis=0)

    # Label processing
    data_lab = data_lab - data_lab.min()
    if right > 0:
        data_lab = data_lab[left:-right]
    else:
        data_lab = data_lab[left:]

    return data_name, data_set, data_lab, end_index


def load_counts(class_counts_file):
    with open(expandvars(class_counts_file)) as f:
        row = f.readline().strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
    return counts


def read_opts():
    parser = OptionParser()
    parser.add_option("--cfg")  # Mandatory
    options, args = parser.parse_args()

    cfg_file = options.cfg
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # DATA
    options.out_folder = config.get('data', 'out_folder')
    options.tr_fea_scp = config.get('data', 'tr_fea_scp')
    options.tr_fea_opts = config.get('data', 'tr_fea_opts')
    options.tr_lab_folder = config.get('data', 'tr_lab_folder')
    options.tr_lab_opts = config.get('data', 'tr_lab_opts')

    options.dev_fea_scp = config.get('data', 'dev_fea_scp')
    options.dev_fea_opts = config.get('data', 'dev_fea_opts')
    options.dev_lab_folder = config.get('data', 'dev_lab_folder')
    options.dev_lab_opts = config.get('data', 'dev_lab_opts')

    options.te_fea_scp = config.get('data', 'te_fea_scp')
    options.te_fea_opts = config.get('data', 'te_fea_opts')
    options.te_lab_folder = config.get('data', 'te_lab_folder')
    options.te_lab_opts = config.get('data', 'te_lab_opts')

    options.count_file = config.get('data', 'count_file')

    # ARCHITECTURE
    options.hidden_dim = config.get('architecture', 'hidden_dim')
    options.N_hid = config.get('architecture', 'N_hid')
    options.drop_rate = config.get('architecture', 'drop_rate')
    options.use_batchnorm = config.get('architecture', 'use_batchnorm')
    options.cw_left = config.get('architecture', 'cw_left')
    options.cw_right = config.get('architecture', 'cw_right')
    options.seed = config.get('architecture', 'seed')
    options.device = config.get('architecture', 'device')

    options.N_ep = config.get('optimization', 'N_ep')
    options.lr = config.get('optimization', 'lr')
    options.halving_factor = config.get('optimization', 'halving_factor')
    options.improvement_threshold = config.get('optimization', 'improvement_threshold')
    options.batch_size = config.get('optimization', 'batch_size')
    options.save_gpumem = config.get('optimization', 'save_gpumem')

    return options
