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
    y1 = [23841.7, 121660.666, 219479.632, 317298.598, 415117.564, 512936.53, 610755.496, 708574.462, 806393.428, 904212.394, 1002031.36]
    y2 = [23841.7, 68345.466, 112849.232, 157352.998, 201856.764, 246360.53, 290864.296, 335368.062, 379871.828, 424375.594, 468879.36]

    y3 = [111843, 111843, 111843, 111843, 111843, 111843, 111843, 111843, 111843, 111843, 111843]

    plt.plot(x, y1, linestyle='--', marker='x', markersize=6, color=red, label="SH-LHE (d=100)")
    plt.plot(x, y2, linestyle=':', marker='x', markersize=6, color=blue, label="SH-LHE (d=20)")

    plt.plot(x, y3, linestyle='-.', marker='x', markersize=6, color=green, label="SH-SWHE")

    axes = plt.gca()
    axes.set_ylim([-10, 1005000])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,4), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("\# avg mult gates/s",fontsize=15)
    plt.yticks(np.arange(0,1005000,200000), fontsize=12)
    plt.xlabel("\% of vectorized mult gates in 2Gbps network", fontsize=15)
    xts = [0, 20, 40, 60, 80, 100]
    plt.xticks(xts, ["$0$", "$20$", "$40$", "$60$", "$80$", "$100$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_3a_protocols_mixed_protocol.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
