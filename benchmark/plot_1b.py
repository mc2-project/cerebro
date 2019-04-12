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
    y1 = [26824, 22037.7, 20835.8, 20444.5, 19235.1, 19095.6]
    y2 = [34573.1, 15369.3, 9598.62, 7089, 5541.65, 4541.09]
    # y3 = [4.44, 4.73, 5.20, 5.74, 6.12]

    plt.plot(x, y1, linestyle=':', marker='o', markersize=6, color=red, label="Many-to-all  ")
    plt.plot(x, y2, linestyle='--', marker='o', markersize=6, color=blue, label="Pairwise")
    # plt.plot(x, y3, linestyle='-.', marker='o', markersize=6, color=green, label="64KB")

    axes = plt.gca()
    # axes.set_xscale("log", basex=2)
    axes.set_ylim([0, 36000])
    # axes.set_xlim([0,700])
	
    plt.legend(ncol=2, columnspacing=0.2, fontsize=15, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0)

    plt.ylabel("Throughput (triple/s)",fontsize=15)
    plt.yticks(np.arange(0,36000,5000), fontsize=12)
    plt.xlabel("Number of parties in 100Mbps network", fontsize=15)
    xts = [2, 4, 6, 8, 10, 12]
    plt.xticks(xts, ["$2$", "$4$", "$6$", "$8$", "$10$", "$12$"], fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(6, 4, forward=True)

    pp = PdfPages('Exp_1b_protocols_in_diff_parties.pdf')
    plt.savefig(pp, format='pdf', bbox_inches='tight', dpi=fig.dpi)
    pp.close()
    plt.show()


if __name__ == "__main__":

    if True:
        plot_dpi_ictf()
