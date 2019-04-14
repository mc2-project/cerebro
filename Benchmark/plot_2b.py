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

    x = [2, 4, 6, 8, 10, 12]
    y1 = [120.49, 76.4725, 54.5879, 44.1632, 35.8264, 30.7972]
    y2 = [200.428, 111.913, 80.4253, 62.2471, 50.2785, 42.8461]
    # y3 = [579.595, 447.296, 364, 308.474, 263.952, 226.461]

    y4 = [2.095625, 1.721695313, 1.627796875, 1.597226563, 1.502742188, 1.49184375]
    y5 = [4.19125, 3.443390625, 3.25559375, 3.194453125, 3.005484375, 2.9836875]
    # y6 = [49.21757813, 56.22460938, 60.67148438, 61.17304688, 61.50742188, 63.29570313]

    plt.plot(x, y2, linestyle='--', marker='o', markersize=6, color='g', label="Pairwise (50d)")
    plt.plot(x, y1, linestyle=':', marker='o', markersize=6, color='k', label="Pairwise (100d)  ")
    # plt.plot(x, y3, linestyle='-.', marker='o', markersize=6, color=green, label="Pairwise (20d vector)")

    plt.plot(x, y5, linestyle='--', marker='o', markersize=6, color='r', label="Many-to-all (50d)")
    plt.plot(x, y4, linestyle=':', marker='o', markersize=6, color='b', label="Many-to-all (100d)  ")
    # plt.plot(x, y6, linestyle='-.', marker='o', markersize=6, color=green, label="Many-to-all (20d vector)")

    axes = plt.gca()
    axes.set_ylim([-10, 250])

    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("Throughput (triple/s)",fontsize=15)
    plt.yticks(np.arange(0,250,50), fontsize=12)
    plt.xlabel("Number of parties in 100Mbps network", fontsize=15)
    xts = [2, 4, 6, 8, 10, 12]
    plt.xticks(xts, ["$2$", "$4$", "$6$", "$8$", "$10$", "$12$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_2b_protocols_in_diff_parties_vectorsize.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
