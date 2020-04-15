import numpy as np

from data_io import load_chunk, read_opts


def dump_chunk(name, scp, fea_opts, lab_folder, lab_opts, left, right):
    names, fea, lab, end_index = load_chunk(
        scp, fea_opts, lab_folder, lab_opts,
        left, right)
    np.savez(f'dataset_{name}.npz',
             names=names, fea=fea, lab=lab, end_index=end_index)


def dump_features(options):
    # Dump features
    left, right = int(options.cw_left), int(options.cw_right)
    for chunk_id, scp in enumerate(options.tr_fea_scp.split(',')):
        # Reading training chunk
        dump_chunk(
            f"tr_{chunk_id}",
            scp, options.tr_fea_opts, options.tr_lab_folder, options.tr_lab_opts,
            left, right)

    dump_chunk(
        f"te",
        options.te_fea_scp, options.te_fea_opts, options.te_lab_folder, options.te_lab_opts,
        left, right)

    dump_chunk(
        f"dev",
        options.dev_fea_scp, options.dev_fea_opts, options.dev_lab_folder, options.dev_lab_opts,
        left, right)


def main():
    options = read_opts()
    dump_features(options)


if __name__ == "__main__":
    main()
