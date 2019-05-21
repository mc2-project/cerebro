#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
# import seaborn as sns
import sys
import datetime
import glob
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FixedFormatter
# %matplotlib inline

import warnings
import matplotlib.cbook
import matplotlib as mpl
import random

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,mathptmx}']

warnings.filterwarnings('ignore')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams.update({'figure.max_open_warning': 0})
mpl.rcParams.update({"font.size": 34,"figure.autolayout": True})
mpl.rc("axes", edgecolor="0.8")
plt.rcParams["font.family"] = "Times New Roman"

legendsize = 40

red = "#a93226"
blue = "#2874a6"
green = "#1e8449"
purple = '#800080'

def plot_dpi_ictf():
    # plt.plot(nrules, nb_vals, marker='o', linestyle='-', color=red)
    # plt.plot(nrules, sb_vals, marker='o', linestyle='-', color=green2)

    x = [100, 300, 600, 900, 1200]
    y1 = [9181.09, 24593.7, 43146.3, 57361.9, 66653.8]
    y2 = [21832.2, 43854.4, 64580.2, 73689.8, 79508.2]

    plt.plot(x, y1, linestyle='--', marker='x', markersize=7, color=red, label="Flat Linear")
    plt.plot(x, y2, linestyle=':', marker='o', markersize=7, color=blue, label="Hierarchical Linear")

    axes = plt.gca()
    axes.set_ylim([-100, 90500])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,4), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("\# mult gates/s",fontsize=15)
    plt.yticks(np.arange(0,90500,15000), fontsize=12)
    plt.xlabel("inter-continent bandwidth (MBps)", fontsize=15)
    xts = [200, 400, 600, 800, 1000, 1200]
    plt.xticks(xts, ["$200$", "$400$", "$600$", "$800$", "$1000$", "$1200$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_7a_6_6.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()

