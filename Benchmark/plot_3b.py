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

def plot_dpi_ictf():
    # plt.plot(nrules, nb_vals, marker='o', linestyle='-', color=red)
    # plt.plot(nrules, sb_vals, marker='o', linestyle='-', color=green2)

    x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y1 = [3761.61, 36012.521, 68263.432, 100514.343, 132765.254, 165016.165, 197267.076, 229517.987, 261768.898, 294019.809, 326270.72]
    y2 = [3761.61, 9630.1722, 15498.7344, 21367.2966, 27235.8588, 33104.421, 38972.9832, 44841.5454, 50710.1076, 56578.6698, 62447.232]

    y3 = [14442.4, 14442.4, 14442.4, 14442.4, 14442.4, 14442.4, 14442.4, 14442.4, 14442.4, 14442.4, 14442.4]

    plt.plot(x, y1, linestyle='--', marker='x', markersize=7, color=red, label="LHE (d=100)")
    plt.plot(x, y2, linestyle=':', marker='o', markersize=5, color=blue, label="LHE (d=10)")

    plt.plot(x, y3, linestyle='-.', marker='s', markersize=5, color=green, label="SWHE")

    axes = plt.gca()
    axes.set_ylim([-10, 355000])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,4), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("\# avg mult gates/s",fontsize=15)
    plt.yticks(np.arange(0,355000,70000), fontsize=12)
    plt.xlabel("\% of vectorized mult gates in 100Mbps network", fontsize=15)
    xts = [0, 20, 40, 60, 80, 100]
    plt.xticks(xts, ["$0$", "$20$", "$40$", "$60$", "$80$", "$100$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_3b_protocols_mixed_protocol.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
