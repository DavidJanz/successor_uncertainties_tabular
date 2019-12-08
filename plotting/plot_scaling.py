import argparse
import pickle

import matplotlib.pyplot as plt
import mpl_latex
import numpy as np
import tabular_analytic

_limits = {'UBE': 25.5, 'BDQN': 25.5}

scatter_size = 2
x_lines = np.arange(4, 180, 0.05)
y_unif = [tabular_analytic.median_solve_time_tree(xs) for xs in x_lines]

ymax_lookup = {True: 60000, False: 5000}


def get_line(scaling, offset, log=False, ymax=50000):
    x_scaling = x_lines
    y_scaling = np.log10(x_scaling) * scaling + offset

    if log:
        x_scaling = np.log10(x_lines)
        ymax = np.log10(ymax)
    else:
        y_scaling = 10 ** y_scaling

    x, y = np.array([(xs, ys) for xs, ys in zip(x_scaling, y_scaling) if ys < ymax]).transpose()

    return x, y


class ScalingParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--file', nargs='+', type=str)
        self.add_argument('--loglog', action='store_true')
        self.add_argument('--print', action='store_true')
        self.add_argument('--max', type=int, default=5000)
        self.add_argument('--ymax', type=int, default=5000)
        self.add_argument('--show', action='store_true')
        self.add_argument('--scaling', type=float, default=0.0)
        self.add_argument('--offset', type=float, default=0.0)


def setup_axis(axs, title):
    axs.grid(True, color='gray', linestyle=':', linewidth=0.3)

    axs.spines['top'].set_visible(False)
    # axs.spines['bottom'].set_visible(False)
    axs.spines['right'].set_visible(False)
    if title:
        axs.set_title(title)


def make_scaling_plot(axs, data, title, first=False, loglog=False, plot_lines=False):
    setup_axis(axs, title)

    x, y, c = data
    if plot_lines:
        x = x[1:]
        y = y[1:]
        c = c[1:]
    x_highest = np.max(x)

    if plot_lines:
        x_scaling, y_scaling = get_line(3, -0.5, log=loglog)
        axs.plot(x_scaling, y_scaling, c='tab:orange', linewidth=1.0, linestyle='--', zorder=1,
                 label='Bootstrap+Prior fit')

        x_scaling, y_scaling = get_line(2.5, -0.95, log=loglog)
        axs.plot(x_scaling, y_scaling, c='tab:blue', linewidth=1.0, linestyle='--', zorder=1,
                 label='Successor Uncertainties fit')
    else:
        axs.set_ylim(0, 5000 + 200)
        axs.set_xlim(0, int(x_highest * 1.05))
        axs.set_yticks([0, 2500, 5000])

    if loglog:
        y = np.array(y)
        for i, (xs, ys, cs) in enumerate(zip(x, y, c)):
            if not i:
                axs.scatter(np.log10([xs]), np.log10([ys]), color='k' if plot_lines else cs, s=scatter_size, zorder=2,
                            label="Successor Uncertainties data")
            else:
                axs.scatter(np.log10([xs]), np.log10([ys]), color='k' if plot_lines else cs, s=scatter_size, zorder=2)
        axs.set_xlim(np.log10(20), np.log10(200))
        axs.set_ylim(2.5, 5)

    else:
        axs.plot(x_lines, y_unif, color='k', alpha=1.0, linestyle='--', zorder=1, linewidth=0.5)

        for i, (xs, ys, cs) in enumerate(zip(x, y, c)):
            if not i:
                axs.scatter([xs], [ys], color='k' if plot_lines else cs, zorder=2, s=scatter_size,
                            label="Successor Uncertainties data")
            else:
                axs.scatter([xs], [ys], color='k' if plot_lines else cs, zorder=2, s=scatter_size)

    if plot_lines and not loglog:
        axs.set_ylim(0, 60e3)
        axs.set_yticks([0, 25e3, 50e3])
        axs.set_xlim(20, 200)

    axs.spines['top'].set_visible(False)
    # axs.spines['bottom'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.set_xlabel(f'{"log10 " if loglog else ""}problem scale $L$')
    if first:
        axs.set_ylabel(f'{"log10 " if loglog else ""}median learning time')


def get_color(entries, max):
    success = [e < max for e in entries]
    if all(success):
        c = 'tab:blue'
    elif any(success):
        c = 'tab:orange'
    else:
        c = 'tab:red'

    return min(np.median(entries), max), c


def read_pickle_scaling(file_name, ymax):
    with open(file_name, 'rb') as f:
        r = pickle.load(f)

    rd = {}

    for (n, e) in r:
        if n in rd:
            rd[n] += [e]
        else:
            rd[n] = [e]

    e_medians = [(n, *get_color(e_list, max=ymax)) for n, e_list in sorted(rd.items())]

    x, y, c = zip(*sorted(e_medians))
    return (x, y, c), rd


def list_format(lst):
    lstr = '{:6}, '
    lstr = (len(lst) * lstr)[:-1]
    return lstr.format(*lst)


if __name__ == '__main__':
    args = ScalingParser().parse_args()

    titles = ['BDQN',
              'UBE',
              'Bootstrap+Prior \n (1x compute)',
              'Bootstrap+Prior \n (25x compute)',
              'SU\n (1x compute)']

    datas, datas_raw = [], []
    for f in args.file:
        data, data_raw = read_pickle_scaling(f, ymax_lookup[args.loglog])
        datas.append(data)
        datas_raw.append(data_raw)

    if args.print:
        for data_raw, file in zip(datas_raw, args.file):
            print(f"File: {file.split('/')[-1].split('.')[0]}\n")
            for n, e_list in sorted(data_raw.items()):
                print(f"{n:3}: {list_format(sorted(e_list))}")
            print("\n")
    elif args.loglog:
        fig, axs = mpl_latex.newfig(1.00, ncols=2, nrows=1, ratio=0.25, sharey=False)

        make_scaling_plot(axs[0], datas[0], title=None, first=True,
                          plot_lines=True)
        make_scaling_plot(axs[1], datas[0], title=None, first=False,
                          loglog=True, plot_lines=True)

        axs[1].legend(loc=4, fontsize='small', frameon=False, markerfirst=False)
        plt.tight_layout()
        mpl_latex.savefig(fig, 'scaling_loglog')

        if args.show:
            plt.show(block=True)

    else:
        ncols = len(datas)
        fig, axs = mpl_latex.newfig(1.00, ncols=ncols, nrows=1, ratio=0.25, sharey=True)
        if ncols == 1:
            axs = [axs]
        titles = [f.split('/')[-1] for f in args.file]
        for i, (a, d, t) in enumerate(zip(axs, datas, titles)):
            make_scaling_plot(a, d, t, first=i == 0)
        plt.tight_layout()
        plt.show(block=True)
        mpl_latex.savefig(fig, 'scaling')
        if args.show:
            plt.show(block=True)
