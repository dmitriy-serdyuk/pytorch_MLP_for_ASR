import numpy as np

from timit_mlp.data_io import load_chunk, read_opts


def dump_chunk(name, data_options, left, right):
    all_data = {}
    for chunk_id, scp in enumerate(data_options.fea_scp):
        names, fea, lab, end_index = load_chunk(
            scp, data_options.fea_opts, data_options.lab_folder, data_options.lab_opts,
            left, right)
        all_data[f'names_{chunk_id}'] = names
        all_data[f'fea_{chunk_id}'] = fea
        all_data[f'lab_{chunk_id}'] = lab
        all_data[f'end_index_{chunk_id}'] = end_index
    np.savez(f'dataset_{name}.npz', **all_data,
             chunks=list(range(len(data_options.fea_scp))))


def dump_features(options):
    # Dump features
    left, right = options.data.cw_left, options.data.cw_right
    dump_chunk("tr", options.tr_data, left, right)
    dump_chunk("te", options.te_data, left, right)
    dump_chunk("dev", options.dev_data, left, right)


def main():
    options = read_opts()
    dump_features(options)


if __name__ == "__main__":
    main()
