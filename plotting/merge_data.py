import argparse
import pickle


class MergeParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('out', type=str)
        self.add_argument('--file', nargs='+', type=str)


if __name__ == '__main__':
    args = MergeParser().parse_args()

    datas = []
    for fname in args.file:
        with open(fname, 'rb') as f:
            datas += pickle.load(f)

    with open(args.out, 'wb') as f:
        pickle.dump(datas, f)
