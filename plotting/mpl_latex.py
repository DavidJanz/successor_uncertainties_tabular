import os

import matplotlib as mpl
import numpy as np

the_textwidth = 397.4
golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0


def figsize(width, ratio=golden_ratio):
    fig_width_pt = the_textwidth  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt * width  # width in inches
    fig_height = fig_width * ratio  # height in inches
    return [fig_width, fig_height]


pgf_with_latex = {  # setup matplotlib to use latex for output
    # "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    # "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "font.size": 5,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    # "pgf.preamble": [
    #     r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
    # r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    # ]
}

mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

plt.style.use(['tableau-colorblind10'])


# plt.style.use(['seaborn-white'])


def newfig(width, ratio=None, **kwargs):
    if ratio:
        s = figsize(width, ratio)
    else:
        s = figsize(width)
    return plt.subplots(**kwargs, figsize=s)


def savefig(fig, filename):
    # fig.savefig(f'figs/{filename}.pgf', bbox_inches='tight')
    os.makedirs('figs', exist_ok=True)
    fig.savefig(f'figs/{filename}.pdf', bbox_inches='tight')


_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colour_local = _colours[0]
colour_sf = _colours[1]
colour_eps = _colours[2]
colour_uni = _colours[3]
colour_ube = _colours[5]


def make_legend(axs):
    handles, labels = axs.get_legend_handles_labels()
    handles_dict = {l: h for l, h in zip(labels, handles)}
    # sort both labels and handles by labels
    handles, labels = zip(*[(label, handles_dict[label]) for label in
                            ('SU', 'UBE', 'BDQN', 'Uniform')])
    leg = axs.legend(labels, handles, prop={'size': 6}, markerfirst=False)
    leg.get_frame().set_linewidth(0.0)
